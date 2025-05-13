import torch
import os

class TrainingConfig():
    """
    训练配置类，用于集中管理所有训练相关的参数和路径配置
    """
    def __init__(self):
        # 数据集路径配置
        self.root_path = r"F:\DL_DataSet\MedMNIST\pneumoniamnist" # 原始数据集的根目录路径
        self.unet_model_output_dir = r'./unet_output'  # UNet模型输出目录（用于保存模型、日志等）
        self.plot_path = r'./plots'  # 训练过程可视化图表保存路径
        
        # VAE（变分自编码器）相关路径配置
        self.vae_path = r'./vae'  # VAE模型根目录
        self.vae_plot_path = r'./vae/plot'  # VAE训练过程可视化图表保存路径
        self.vae_checkpoints_path = r'./vae/checkpoints'  # VAE模型检查点保存路径
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 自动选择可用的设备（优先使用GPU）
        # self.device = torch.device('cpu') # 强制使用CPU（调试时使用）
        
        # 模型架构参数
        self.in_channels = 1  # 输入图像的通道数（灰度图为1）
        self.out_channels = 1  # 输出图像的通道数
        self.latent_channels = 8  # 潜在空间的通道数（用于VAE等模型）
        
        # 图像尺寸配置
        self.img_size = 64  # 原始数据集中的图像尺寸
        self.target_size = 64  # 模型处理的目标图像尺寸（通常与img_size相同）
        
        # 训练超参数
        self.train_batch_size = 64  # 训练时的批量大小
        self.num_classes = 2  # 分类任务中的类别数
        self.num_epochs = 100  # 训练的总轮数
        
        # UNet模型优化器参数
        self.lr_unet = 1e-4  # UNet模型的学习率
        self.num_warmup_steps = 500  # 学习率预热步数（逐步增加学习率）
        
        # VAE模型训练参数
        self.lr_vae = 1e-4  # VAE模型的学习率
        self.vae_epochs = 50  # VAE模型的训练轮数
        
        # 扩散模型/去噪相关参数
        self.denoising_timesteps = 1000  # 去噪过程的时间步数（用于扩散模型）

# 创建配置实例
config = TrainingConfig()