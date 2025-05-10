import torch
import os
from utils import create_path_if_not_exists

class TrainingConfig():
    def __init__(self):
        self.root_path = r"F:\DL_DataSet\MedMNIST\pneumoniamnist" # 数据集的根目录
        self.unet_model_output_dir = r'./unet_output'
        
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 选择训练设备
        self.device = torch.device('cpu') # 先用cpu方便调试。
        
        self.img_size = 64 # 数据集图片的分辨率
        self.target_size = 64 # 生成图片的分辨率

        self.train_batch_size = 32
        self.num_classes = 2
        self.num_epochs = 100
        
        self.lr_unet = 1e-4
        self.lr_warmup_steps = 500
        
        self.denoising_timesteps = 1000
        
        
        
        create_path_if_not_exists(self.output_dir)
    

        

config = TrainingConfig()