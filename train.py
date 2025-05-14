from utils import *
from config import *
from models import *
from data import *

import torch
from torch import nn,optim
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from accelerate import Accelerator
from diffusers import DDPMScheduler
import numpy as np
import os
from tqdm import tqdm

def train_vae():
    """
    训练VAE模型
    """
    
    # 初始化操作
    create_path_if_not_exists(config.vae_path)
    create_path_if_not_exists(config.vae_plot_path)    
    create_path_if_not_exists(config.vae_checkpoints_path)
    
    
    vae_optimizer = optim.AdamW(vae.parameters(),lr=config.lr_vae)
    epoch_train_losses = [] # 记录每一轮的训练损失，用于绘制损失曲线
    epoch_test_losses = []
    
    # 开始迭代
    for epoch in range(config.vae_epochs):
        losses = [] # 记录当前轮次的损失，取平均
        
        # ==============训练部分=============
        vae.train()
        
        for step,(images,_) in enumerate(tqdm(train_loader)):
            images = images.to(config.device)
            vae_optimizer.zero_grad()
            
            # vae前向传播部分
            latent_images = vae.encode(images)
            outputs = vae.decode(latent_images['latent_dist'].sample())
            outputs = outputs.sample
            
            # 计算损失
            vae_loss = compute_vae_loss(images, outputs, latent_images['latent_dist'])
            losses.append(vae_loss.item())
            
            vae_loss.backward()
            vae_optimizer.step()
        
        # 输出损失信息和保存每一轮的训练损失
        avg_loss = np.mean(losses)
        epoch_train_losses.append(avg_loss)    
        print(f"VAE Epoch {epoch + 1},Loss: {avg_loss:.4f}")
        
        torch.save(vae.state_dict(),f=os.path.join(config.vae_checkpoints_path,f'vae_epoch{epoch + 1}.pth'))
        
        plot_side_by_side(images, outputs.sample, posterior.latent_dist.sample(), epoch+1)
        
        # ================测试部分====================
        vae.eval()
        with torch.no_grad():
            losses = []
            
            for step, (images, _) in enumerate(tqdm(test_loader)):
                images = images.to(config.device)
                
                # VAE 前向传播
                posterior = vae.encode(images)
                outputs = vae.decode(posterior['latent_dist'].sample())
                outputs = outputs.sample
                
                # 计算测试集的损失
                vae_loss = compute_vae_loss(images,outputs,posterior)
                losses.append(vae_loss.item())
                
            avg_loss = np.mean(losses)
            epoch_test_losses.append(avg_loss)    
        print(f"VAE Epoch {epoch + 1},Test Loss: {avg_loss:.4f}")
        
    
    train_plot_title = 'VAE Training Loss'
    test_plot_title = 'VAE Test Loss'
    plot_loss_curve(epoch_train_losses,train_plot_title,os.path.join(config.vae_plot_path,f'{train_plot_title}.png'))
    plot_loss_curve(epoch_test_losses,test_plot_title,os.path.join(config.vae_plot_path,f'{test_plot_title}.png'))
    draw_loss(epoch_train_losses,epoch_test_losses,save_path=config.vae_plot_path)


def train_unet():
     # 初始化操作
    create_path_if_not_exists(config.unet_path)
    create_path_if_not_exists(config.unet_plot_path)    
    create_path_if_not_exists(config.unet_checkpoints_path)
    
    vae.load_state_dict(torch.load(os.path.join(config.vae_checkpoints_path,f'vae_epoch{config.vae_epochs}.pth')))
    
    # 初始化DDPM噪声调度器，设置去噪时间步长
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    
    # 设置优化器
    optimizer = optim.AdamW(unet.parameters(), lr=config.lr_unet)

    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=(len(train_loader) * 100),
    )
    

    # 初始化加速器，用于分布式训练
    accelerator = Accelerator()
    # 使用加速器准备模型、优化器、训练数据加载器和学习率调度器
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_loader, lr_scheduler
    )
    
    # 初始化指数移动平均（EMA）模型，用于平滑模型参数
    ema_model = EMAModel(model.parameters(), decay=0.9999, use_ema_warmup=True)
 
    # 打印训练设备信息
    print_device(config.device)
        
    model.to(config.device)
    
    model.train()
    for epoch in range(config.num_epochs):
        losses = []
        for step,(clean_images,labels) in enumerate(tqdm(train_loader)):
            # 获取干净的未加噪的数据并移动到训练设备
            clean_images = clean_images.to(config.device)
            
            # 获取数据标签，并移动到训练设备上
            labels = labels.to(config.device)
            
            # 生成一份与原始数据形状相同的随机噪声
            noise = torch.randn(clean_images.shape).to(config.device)
            
            # 获取当前批次的样本数据
            bs = clean_images.shape[0]
            
            # 为每个样本随机采样一个时间步长
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=config.device).long()

            
            # 根据每个时间步长的噪声幅度，向干净的潜在表示中添加噪声
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            optimizer.zero_grad()
            noise_pred =  unet(sample=noisy_images, timestep=timesteps, encoder_hidden_states=None, class_labels=labels)
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
        
        generate(unet,noise_scheduler,epoch)
        
        torch.save(unet.state_dict(), f=os.path.join(config.unet_path,f'unet_epoch{epoch + 1}.pth'))
        
        test_losses = []
        with torch.no_grad():
            # 遍历测试数据
            for step, (test_images,test_labels) in enumerate(tqdm(test_loader)):
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


if __name__ == '__main__':
    train_vae()
    
        
            
            
            
            
            
    
    
    