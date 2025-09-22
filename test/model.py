#定义模型结构

from turtle import forward
import torch
import torch.nn as nn




__all__ = [
    "MyModel",
<<<<<<< Updated upstream
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
=======
    "AlexNet",
]




class AlexNet(nn.Module):
    #定义init
    def __init__(self, num_classes = 10):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3,48, kernel_size=11),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(48,128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(128,192,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192,192,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192,128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
        )


        self.head = nn.Sequential(
            nn.Linear(1*1*128,2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048,2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048,num_classes),
>>>>>>> Stashed changes
        )

    #定义前向传播
    def forward(self,x):
        x = self.backbone(x)
<<<<<<< Updated upstream
=======
        x = torch.flatten(x,start_dim=1)
>>>>>>> Stashed changes
        x = self.head(x)
        return x




<<<<<<< Updated upstream
=======
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







>>>>>>> Stashed changes
