import torch
from torch import nn,optim
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from accelerate import Accelerator
from diffusers import DDPMScheduler
from data import train_dataloader,test_dataloader
from config import config
import numpy as np
import os


def train_unet(unet:nn.Module):
    # 初始化DDPM噪声调度器，设置去噪时间步长
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    # 设置优化器
    optimizer = optim.AdamW(unet.parameters(),lr=config.lr_unet)
    
    lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
    )
    
    # 初始化加速器，用于分布式训练
    accelerator = Accelerator()
    # 使用加速器准备模型、优化器、训练数据加载器和学习率调度器
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    # 初始化指数移动平均（EMA）模型，用于平滑模型参数
    ema_model = EMAModel(model.parameters(),decay=0.999,use_ema_warmup=True)
    
    model.to(config.device)
    model.train()
    for epoch in range(config.num_epochs):
        losses = []
        for step,batch in enumerate(train_dataloader):
            clean_images = batch['pixel_values'].to(config.device)
            labels = batch['class_label'].to(config.device)
            
            origin_labels = labels
            noise = torch.randn(clean_images.shape).to(config.device)
            
            bs = clean_images.shape[0]
            
            timesteps = torch.randint(0,noise_scheduler.config.num_train_timesteps, (bs,), device=config.device).long()
            
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            optimizer.zero_grad()
            noise_pred = model(sample=noisy_images, timestep=timesteps, encoder_hidden_states=None, class_labels=labels)
            noise_pred = noise_pred.sample
            loss = F.mse_loss(noise_pred,noise)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            ema_model.step(model.parameters())
            
            losses.append(loss.item())
            
        # 打印当前epoch的训练损失
        print(f'Epoch: {epoch+1}, Train loss: {np.mean(losses)}')
        torch.save(unet.state_dict(), f=os.path.join(config.output_dir,f'unet_epoch{epoch + 1}.pth'))
        
            
            
            
            
            
    
    
    