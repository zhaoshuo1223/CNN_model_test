#定义训练：包括损失函数的计算以及反向传播
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from model import VGGNet
from dataset import Data                                     #Data为张量
from torch.utils.data import DataLoader
import os



model = VGGNet()
net = model.backbone
input = torch.randn(16,3,64,64)
out = net(input)
print(out.shape)