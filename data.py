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
class IntLabelDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        
    def __len__(self):
        return len(self.original_dataset)
        
    def __getitem__(self, idx):
        img, label = self.original_dataset[idx]
        return img, int(label.item())  # 确保转换为 Python int
    
# 图像预处理
transform = transforms.Compose([
    transforms.ToTensor(),             # [0,1]
    transforms.Resize((config.img_size,config.img_size)),       
    transforms.Normalize((0.5,), (0.5,))  # [-1,1]
])


# 加载数据集
train_dataset = PneumoniaMNIST(root=config.root_path,split='train',transform=transform,download=True,size=64)
test_dataset = PneumoniaMNIST(root=config.root_path,split='test',transform=transform,download=True,size=64)

# 注意事项：这里需要修改medmnist的源码
labels = train_dataset.labels
labels = [int(x) for x in labels.flatten()]
train_dataset.labels = labels

# train_dataset = IntLabelDataset(train_dataset)
# test_dataset = IntLabelDataset(test_dataset)

# # 
# train_dataset = [process_ds2hug(i) for i in train_dataset]
# test_dataset= [process_ds2hug(i) for i in test_dataset]

# # 转为 Huggingface Dataset
# train_dataset = Dataset.from_list(train_dataset)
# test_dataset = Dataset.from_list(test_dataset)

train_loader = DataLoader(train_dataset,batch_size=config.train_batch_size,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=config.train_batch_size,shuffle=False)

# img,label = train_dataset[0]
# print(img.shape)
# print(type(img))
# print(label)
# print(type(label))