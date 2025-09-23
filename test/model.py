#定义模型结构


import torch
import torch.nn as nn




__all__ = [
    "MyModel",
    "AlexNet",
    "VGGNet",
]

VGGconfigs = {
    # A 数字代表卷积核的数量，'M' 表示池化层
    'vgg11':[64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    # B
    'vgg13':[64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    # D
    'vgg16':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    # E
    'vgg19':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG(nn.Module):
    #定义init
    def __init__(self, backbone, num_classes = 10):
        super().__init__()
        self.backbone = backbone

        self.head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512*7*7, 2048),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Linear(2048, num_classes)
        )

    #定义前向传播
    def forward(self,x):
        x = self.backbone(x)
        x = torch.flatten(x,start_dim=1)
        x = self.head(x)
        return x

def make_backbone(cfg: list):
    layers = []
    in_channels = 3
    for i in cfg:
        if i == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv = nn.Conv2d(in_channels, i, kernel_size=3, padding=1)
            layers += [conv, nn.ReLU(True)]
            in_channels = i
    
    return nn.Sequential(*layers)
def VGGNet(mode_name="vgg16"):
    model = VGG(make_backbone(VGGconfigs[mode_name]))
    return model









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
        )

    #定义前向传播
    def forward(self,x):
        x = self.backbone(x)
        x = torch.flatten(x,start_dim=1)
        x = self.head(x)
        return x




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







