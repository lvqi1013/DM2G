import torch
import os
import matplotlib.pyplot as plt
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
    
def draw_loss(train_losses,test_losses):
    
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
    plt.savefig(os.path.join(config.plot_path, 'Train VS Test Loss.png'))
    plt.close()  