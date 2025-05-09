from medmnist import PneumoniaMNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import Dataset
from config import TrainingConfig

config = TrainingConfig()


def process_ds2hug(example):
    img,label = example
    return {
        "pixel_values": img,
        "class_label": label
    }

# 图像预处理
transform = transforms.Compose([
    transforms.ToTensor(),             # [0,1]
    transforms.Resize((config.img_size,config.img_size)),       
    transforms.Normalize((0.5,), (0.5,))  # [-1,1]
])



# 加载数据集
pneumoniaMNIST_train = PneumoniaMNIST(root=config.root_path,split='train',transform=transform,download=True)
pneumoniaMNIST_test = PneumoniaMNIST(root=config.root_path,split='test',transform=transform,download=True)


# 
pneumoniaMNIST_train = [process_ds2hug(i) for i in pneumoniaMNIST_train]
pneumoniaMNIST_test= [process_ds2hug(i) for i in pneumoniaMNIST_test]

# 转为 Huggingface Dataset
train_dataset = Dataset.from_list(pneumoniaMNIST_train)
test_dataset = Dataset.from_list(pneumoniaMNIST_test)

train_dataloader = DataLoader(train_dataset,batch_size=config.train_batch_size,shuffle=True)
test_dataloader = DataLoader(test_dataset,batch_size=config.train_batch_size,shuffle=False)