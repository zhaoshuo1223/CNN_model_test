#定义模型结构
import torch
import torch.nn as nn




__all__ = [
    "ResNet18"
]

#定义残差块：提高net的下限，使其下限从学习复杂特征转变为学习0.
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        

        self.shortcut = nn.Sequential()
        
        #调整残差通道数
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # 避免就地操作，使用非就地加法
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out




class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # 初始化网络权重
        #self._initialize_weights()
        
         # 梯度监控相关属性
        self.gradient_hooks = []
        self.gradient_stats = {}
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


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
                # 使用register_full_backward_hook避免警告
                hook = module.register_full_backward_hook(gradient_hook(name))
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
