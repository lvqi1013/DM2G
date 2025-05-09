import torch
import os

class Config():
    def __init__(self):
        self.root_path = r"F:\DL_DataSet\MedMNIST\pneumoniamnist" # 数据集的根目录
        self.img_size = 64 # 数据集图片的分辨率
        self.target_size = 64 # 生成图片的分辨率
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 选择训练设备
        self.device = torch.device('cpu') # 先用cpu方便调试。