from medmnist import PneumoniaMNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import Dataset
from config import TrainingConfig

config = TrainingConfig()


# def process_ds2hug(example):
#     img,label = example
#     return {
#         "pixel_values": img,
#         "class_label": label
#     }

# 图像预处理
transform = transforms.Compose([
    transforms.ToTensor(),             # [0,1]
    transforms.Resize((config.img_size,config.img_size)),       
    transforms.Normalize((0.5,), (0.5,))  # [-1,1]
])



# 加载数据集
train_dataset = PneumoniaMNIST(root=config.root_path,split='train',transform=transform,download=True,size=64)
test_dataset = PneumoniaMNIST(root=config.root_path,split='test',transform=transform,download=True,size=64)


# # 
# train_dataset = [process_ds2hug(i) for i in train_dataset]
# test_dataset= [process_ds2hug(i) for i in test_dataset]

# # 转为 Huggingface Dataset
# train_dataset = Dataset.from_list(train_dataset)
# test_dataset = Dataset.from_list(test_dataset)

train_dataloader = DataLoader(train_dataset,batch_size=config.train_batch_size,shuffle=True)
test_dataloader = DataLoader(test_dataset,batch_size=config.train_batch_size,shuffle=False)
