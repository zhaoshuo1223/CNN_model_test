#定义模型结构

from turtle import forward
import torch
import torch.nn as nn




__all__ = [
    "MyModel",
]

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




