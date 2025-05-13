from diffusers import DDPMScheduler
from torchvision import transforms
import torch.nn.functional as F
import torch
from torch import nn
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from tqdm import tqdm
from config import config

def print_device(device):
    """
    打印训练设备的信息。
    :param device: 获取的设备
    :return: 无
    """
    print('==============所用训练设备============')
    print(f'设备：{device}')
    if device.type == 'cuda':
        print(f'设备名称为：{torch.cuda.get_device_name(0)}')
    print('====================================')
    print()
    
def create_path_if_not_exists(path):
    os.makedirs(path,exist_ok=True)
    
def draw_loss(train_losses,test_losses,save_path = None):
    
    create_path_if_not_exists(config.plot_path)
    
    plt.figure(figsize=(16,8))
    
    # 训练损失子图
    plt.subplot(1, 2, 1)
    epochs = range(1, len(train_losses) + 1)  # 生成x轴数据（轮次）
    plt.plot(epochs, train_losses)
    plt.title('Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # 测试损失子图
    plt.subplot(1, 2, 2)
    epochs = range(1, len(test_losses) + 1)  # 生成x轴数据（轮次）
    plt.plot(epochs, test_losses)
    plt.title('Testing Losses')  # 修正标题
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.tight_layout()  # 调整布局
    plt.savefig(os.path.join(save_path, 'Train VS Test Loss.png'))
    plt.close()  

def revert_images(imgs: torch.tensor) -> np.array:
    h = imgs.shape[-1]
    imgs = imgs.cpu().detach().numpy()
    min_vals = imgs.min(axis=(1, 2, 3))[:, np.newaxis, np.newaxis, np.newaxis]
    max_vals = imgs.max(axis=(1, 2, 3))[:, np.newaxis, np.newaxis, np.newaxis]

    imgs = ((max_vals - imgs)/(max_vals-min_vals))*255
    if imgs.shape[1] == 1:
        imgs = imgs.astype(int).reshape(-1, h, h)

    return imgs


def plot(recon_imgs, timesteps, epoch):
    create_path_if_not_exists(os.path.join(config.plot_path, f'epoch_{epoch}'))
    recon_imgs = revert_images(recon_imgs.sample)
    fig,axs = plt.subplots(1,2)
    for i in range(2):
        axs[i//2][i%2].imshow(recon_imgs[i], cmap='gray')
        axs[i//2][i%2].axis('off')  
        axs[i//2][i%2].set_title(str(i))
        plt.suptitle(f"Timesteps: {timesteps}")
        plt.savefig(os.path.join(config.plot_path, f'epoch_{epoch}', f'plot {timesteps}.png'))
        plt.clf()
    plt.close()

def generate(unet:nn.Module,
             noise_scheduler:DDPMScheduler,
             epoch:int):
    
    latents = torch.rand((2,1,config.target_size,config.target_size)).to(config.device)
    labels = torch.arange(2).to(config.device)
    for time in tqdm(noise_scheduler.timesteps):
        with torch.no_grad():
            noise_pred = unet(latents,labels,encoder_hidden_states = None).sample()
            recon_imgs = noise_scheduler.step(noise_pred, time, latents).prev_sample
            if time == 999 or time % 100 == 0:
                plot(recon_imgs, time, epoch)

def compute_vae_loss(images, outputs, latent_dist):
    """
    计算 VAE 的重建损失和 KL 散度损失
    """
    recon_loss = F.mse_loss(outputs, images, reduction='mean')
    kl_loss = (latent_dist.kl() / (config.train_batch_size * config.img_size * config.img_size)).mean()
    vae_loss = recon_loss + 0.5 * kl_loss
    return vae_loss

def plot_loss_curve(losses:list,title:str,save_path:str = None):
    """
    绘制并可选保存损失曲线图

    参数:
        losses (list): 每个 epoch 的平均损失
        title (str): 图像标题
        save_path (str or None): 如果提供路径，则保存图像
    """
    plt.figure(figsize=(8,5))
    plt.plot(range(1,len(losses) + 1), losses,marker = 'o')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f'损失图像已保存至{save_path}')
    plt.close()
    
    
def plot_side_by_side(images_y: torch.tensor, images_pred: torch.tensor, 
                      latents: torch.tensor, epoch: int):
    """
    可视化VAE模型的表现和他的潜在空间
    """
    images_y, images_pred = revert_images(images_y), revert_images(images_pred)
    latents = revert_images(latents)
    idx = np.random.randint(0, images_y.shape[0])
    fig, axs = plt.subplots(1, 2)
    
    # Plot input image and Output image
    axs[0].imshow(images_y[idx], cmap='gray')
    axs[0].axis('off')  
    axs[0].set_title("Input")
    
    axs[1].imshow(images_pred[idx], cmap='gray')
    axs[1].axis('off')  
    axs[1].set_title("Output")
    plt.savefig(os.path.join(config.vae_plot_path, f'epoch_{epoch}_input_output.png'))
    plt.clf()

    latent_channels = latents.shape[1]
    fig, axs = plt.subplots(1, 4)

    # Plot the different latent channels
    for i in range(latent_channels):
        axs[i].imshow(latents[idx, i, :, :], cmap='gray')
        axs[i].axis('off')  
        axs[i].set_title(f"Latent channel: {i}", fontsize=8)       
    plt.savefig(os.path.join(config.vae_plots_path, f'epoch_{epoch}_latent_channels.png'))
    plt.clf()
    


if __name__ == '__main__':
    labels = torch.arange(10)
    print(labels)
    print(labels.dtype)