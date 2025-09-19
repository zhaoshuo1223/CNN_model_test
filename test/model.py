#定义模型结构


import torch
import torch.nn as nn




__all__ = [
    "MyModel",
    "SimpleNet",
]




class SimpleNet(nn.Module):
    #定义init
    def __init__(self, num_classes = 10):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )


        self.head = nn.Sequential(
            nn.Linear(64*8*8,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    #定义前向传播
    def forward(self,x):
        x = self.backbone(x)
        x = torch.flatten(x,start_dim=1)
        x = self.head(x)
        return x















class MyModel(nn.Module):
    #定义init
    def __init__(self,num_classes = 2, n_in = 10, n_out = 1, n_h = 5):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(n_in, n_h),
            nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Linear(n_h, n_out),
            nn.Sigmoid()
        )

    #定义前向传播
    def forward(self,x):
        x = self.backbone(x)
        x = self.head(x)
        return x




