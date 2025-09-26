#设置数据集
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import Normalize




transform = torchvision.transforms.Compose([
    #transforms.Resize((64, 64)),  # 将图像大小调整到224x224
    torchvision.transforms.ToTensor(),
    Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])
])
Data = torchvision.datasets.CIFAR10('./data', train=True, transform=transform, download=True)
TestData = torchvision.datasets.CIFAR10('./data', train=False, transform=transform, download=True)

class Mydata(Dataset):
    def __init__(self,data,label):
        self.data = data
        self.label = label
    
    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        return self.data[index], self.label[index]






