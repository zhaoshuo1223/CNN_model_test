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
    #'vgg16':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg16':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M'],
    # E
    'vgg19':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG(nn.Module):
    #定义init
    def __init__(self, backbone, num_classes = 10):
        super().__init__()
        self.backbone = backbone       

        """ self.head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512*2*2, 2048),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Linear(2048, num_classes)
        ) """
        self.head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256*8*8, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes)
        )
        # 初始化网络权重
        #self._initialize_weights()
        
        # 梯度监控相关属性
        self.gradient_hooks = []
        self.gradient_stats = {}
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)
    
    def _register_gradient_hooks(self):
        """注册梯度监控钩子函数"""
        def gradient_hook(name):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    # 将梯度值放大100倍
                    grad_norm = grad_output[0].norm().item() * 100
                    
                    if name not in self.gradient_stats:
                        self.gradient_stats[name] = {
                            'norms': []
                        }
                    
                    # 只记录梯度范数
                    self.gradient_stats[name]['norms'].append(grad_norm)
            return hook
        
        # 为每个有参数的模块注册钩子
        for name, module in self.named_modules():
            if len(list(module.parameters())) > 0:  # 只对有参数的模块注册钩子
                hook = module.register_backward_hook(gradient_hook(name))
                self.gradient_hooks.append(hook)
    
    def get_gradient_stats(self):
        """获取梯度统计信息"""
        return self.gradient_stats.copy()
    
    def reset_gradient_stats(self):
        """重置梯度统计信息"""
        for name in self.gradient_stats:
            self.gradient_stats[name]['norms'] = []
    
    def print_gradient_summary(self, epoch=None):
        """打印梯度变化摘要"""
        if epoch is not None:
            print(f"\n=== Epoch {epoch} 梯度统计 ===")
        
        for name, stats in self.gradient_stats.items():
            if stats['norms']:  # 如果有梯度数据
                avg_norm = sum(stats['norms']) / len(stats['norms'])
                print(f"{name}: 平均梯度范数 = {avg_norm:.6f}")
    
    def remove_gradient_hooks(self):
        """移除梯度监控钩子"""
        for hook in self.gradient_hooks:
            hook.remove()
        self.gradient_hooks.clear()
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







