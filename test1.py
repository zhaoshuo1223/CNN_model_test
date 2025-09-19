
import torch
import torch.nn as nn
from test.dataset import Data
from torch.utils.data import DataLoader

""" train_data = DataLoader(Data,16)
for i,(x,y) in enumerate(train_data):
    print(x.shape,y) """
tem = torch.randn(16, 3, 32, 32)  # 添加batch维度
""" net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
             """
net = nn.Sequential(
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
y = net(tem)
print(y.shape)

