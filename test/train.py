#定义训练：包括损失函数的计算以及反向传播
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from model import ResNet18
from dataset import Data                                     #Data为张量
from torch.utils.data import DataLoader
import os

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# 创建TensorBoard日志目录

log_dir = 'runs/ResNet/test'
save_dir = 'save_model/ResNet18'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# 初始化TensorBoard writer
writer = SummaryWriter(log_dir)


model = ResNet18()
model = model.to(device)  # 将模型移动到GPU

# 记录模型结构到TensorBoard (在注册梯度钩子之前)
dummy_input = torch.randn(1, 3, 32, 32).to(device)
writer.add_graph(model, dummy_input)
# 现在注册梯度监控钩子
model._register_gradient_hooks()

losses = []
#定义优化器SGD，损失函数交叉熵
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
#输入数据处理：转为张量
train_data = DataLoader(Data,16)

print('start:')
global_step = 0

for epoch in range(51):
    epoch_loss = 0.0
    epoch_acc = 0.0
    num_batches = 0
    
    #张量通过模型输出预测值
    for i, (x,y) in enumerate(train_data):
        x, y = x.to(device), y.to(device)
        y_pre = model(x)
        #计算Loss
        loss = criterion(y_pre,y)
        losses.append(loss.item())
        epoch_loss += loss.item()
        
        # 计算准确率
        acc = (y_pre.argmax(1) == y).float().mean()
        epoch_acc += acc.item()
        num_batches += 1
        
        #先对梯度清零，反向传播计算梯度
        optimizer.zero_grad()
        loss.backward()
        #通过优化器，更新权重
        optimizer.step()
        
        # 记录到TensorBoard (每10个batch记录一次)
        if i % 10 == 0:
            writer.add_scalar('Loss/Batch', loss.item(), global_step)
            writer.add_scalar('Accuracy/Batch', acc.item(), global_step)
            
            # 记录学习率
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
        
        global_step += 1
    if epoch % 5 == 0:
        # 临时移除梯度钩子以便保存模型
        model.remove_gradient_hooks()
        torch.save(model, os.path.join(save_dir, "module%s" % epoch))
        # 重新注册梯度钩子
        model._register_gradient_hooks()
        
        
    # 计算epoch平均指标
    avg_loss = epoch_loss / num_batches
    avg_acc = epoch_acc / num_batches
    
    # 记录epoch级别的指标到TensorBoard
    writer.add_scalar('Loss/Epoch', avg_loss, epoch)
    writer.add_scalar('Accuracy/Epoch', avg_acc, epoch)
    
    print(f"第{epoch}轮的acc为{avg_acc:.4f}, loss为{avg_loss:.4f}")
    
    # 每5个epoch记录一次模型权重分布
    if epoch % 5 == 0:
        for name, param in model.named_parameters():
            writer.add_histogram(f'Weights/{name}', param, epoch)
            if param.grad is not None:
                writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

# 记录一些样本图像到TensorBoard
sample_batch = next(iter(train_data))
sample_images, sample_labels = sample_batch
sample_images = sample_images[:8].to(device)  # 取前8张图像
sample_labels = sample_labels[:8].to(device)

# 创建图像网格
img_grid = torchvision.utils.make_grid(sample_images.cpu(), normalize=True, scale_each=True)
writer.add_image('Sample_Images', img_grid, 0)

# 记录预测结果
with torch.no_grad():
    sample_predictions = model(sample_images)
    sample_pred_labels = sample_predictions.argmax(1)

# 关闭writer
writer.close()
print("训练完成！TensorBoard日志保存在:", log_dir)
print("运行 'tensorboard --logdir=runs' 查看可视化结果")




