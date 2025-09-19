#设置数据集
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import Normalize




transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
    #torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
Data = torchvision.datasets.CIFAR10('./data', train=True, transform=transform, download=True)
TestData = torchvision.datasets.CIFAR10('./data', train=False, transform=transform, download=True)

class Mydata(Dataset):
    def __init(self,data,label):
        self.data = data
        self.label = label
    
    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        return super().__getitem__(index)






