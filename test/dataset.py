#设置数据集
import torch
from torch.utils.data import Dataset




class Mydata(Dataset):
    def __init(self,data,label):
        self.data = data
        self.label = label
    
    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        return super().__getitem__(index)







