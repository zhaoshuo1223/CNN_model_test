#定义训练：包括损失函数的计算以及反向传播
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from model import MyModel,AlexNet
from dataset import Data                                     #Data为张量
from torch.utils.data import DataLoader
import os


# 创建TensorBoard日志目录
log_dir = 'runs/AlexNet'
save_dir = 'save_model/AlexNet'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# 初始化TensorBoard writer
writer = SummaryWriter(log_dir)

model = AlexNet()
losses = []
#定义优化器SGD，损失函数交叉熵
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
#输入数据处理：转为张量
train_data = DataLoader(Data,16)

# 记录模型结构到TensorBoard
dummy_input = torch.randn(1, 3, 32, 32)
writer.add_graph(model, dummy_input)

print('start:')
global_step = 0

for epoch in range(51):
    epoch_loss = 0.0
    epoch_acc = 0.0
    num_batches = 0
    
    #张量通过模型输出预测值
    for i, (x,y) in enumerate(train_data):
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
        torch.save(model,os.path.join(save_dir,"module%s"%epoch))
        
        
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
sample_images = sample_images[:8]  # 取前8张图像
sample_labels = sample_labels[:8]

# 创建图像网格
img_grid = torchvision.utils.make_grid(sample_images, normalize=True, scale_each=True)
writer.add_image('Sample_Images', img_grid, 0)

# 记录预测结果
with torch.no_grad():
    sample_predictions = model(sample_images)
    sample_pred_labels = sample_predictions.argmax(1)

# 关闭writer
writer.close()
print("训练完成！TensorBoard日志保存在:", log_dir)
print("运行 'tensorboard --logdir=runs' 查看可视化结果")




