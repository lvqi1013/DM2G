import torch
import os

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
    
