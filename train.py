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
from tqdm import tqdm
from utils import * 


def train_unet(unet:nn.Module):
    # 初始化DDPM噪声调度器，设置去噪时间步长
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    # 设置优化器
    optimizer = optim.AdamW(unet.parameters(),lr=config.lr_unet)
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs)
    )
    
    # 初始化加速器，用于分布式训练
    accelerator = Accelerator()
    # 使用加速器准备模型、优化器、训练数据加载器和学习率调度器
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    # 初始化指数移动平均（EMA）模型，用于平滑模型参数
    ema_model = EMAModel(model.parameters(),decay=0.999,use_ema_warmup=True)
    
    # 打印训练设备信息
    print_device(config.device)
        
    model.to(config.device)
    
    model.train()
    for epoch in range(config.num_epochs):
        losses = []
        for step,batch in enumerate(train_dataloader):
            # 获取干净的未加噪的数据并移动到训练设备
            clean_images = batch['pixel_values'].to(config.device)
            
            # 获取数据标签，并移动到训练设备上
            labels = batch['class_label'].to(config.device)
            
            # 保存原始标签数据
            origin_labels = labels
            
            # 生成一份与原始数据形状相同的随机噪声
            noise = torch.randn(clean_images.shape).to(config.device)
            
            # 获取当前批次的样本数据
            bs = clean_images.shape[0]
            
            # 为每个样本随机采样一个时间步长
            timesteps = torch.randint(low = 0,high = noise_scheduler.config.num_train_timesteps, size=(bs,), device=config.device).long()
            
            # 根据每个时间步长的噪声幅度，向干净的潜在表示中添加噪声
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
        
        test_losses = []
        with torch.no_grad():
            # 遍历测试数据
            for step, (test_images,test_labels) in enumerate(tqdm(test_dataloader)):
                test_images = test_images.to(config.device)
                test_labels = test_labels.to(config.device)
                
                test_noise = torch.rand(test_images.shape).to(config.device)
                test_bs = test_images.shape[0]
                
                test_timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                                               size=(test_bs,), device=config.device).long()
                test_noise_images = noise_scheduler.add_noise(test_images,test_noise,test_timesteps)
                
                test_noise_pred = unet(sample = test_noise_images,
                                       timestep = test_timesteps,
                                       encoder_hidden_states = None,
                                       class_labels = test_labels)
                
                test_noise_pred = test_noise_pred.sample
                
                test_loss = F.mse_loss(test_noise_pred,test_noise)
                
                test_losses.append(test_loss.item())
                
            print(f'Epoch: {epoch+1}, Test loss: {np.mean(test_losses)}')
        
    
    draw_loss(train_losses=losses,test_losses=test_losses)
            

    
        
            
            
            
            
            
    
    
    