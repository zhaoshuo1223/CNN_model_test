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

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')
if torch.cuda.is_available():
    print(f'GPU名称: {torch.cuda.get_device_name(0)}')
    print(f'GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')


# 创建TensorBoard日志目录
log_dir = 'runs/VGGNet/test3'
save_dir = 'save_model/VGGNet'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# 初始化TensorBoard writer
writer = SummaryWriter(log_dir)

model = VGGNet()
model = model.to(device)  # 将模型移动到GPU

# 记录模型结构到TensorBoard (在注册梯度钩子之前)
dummy_input = torch.randn(1, 3, 64, 64).to(device)
writer.add_graph(model, dummy_input)

# 现在注册梯度监控钩子
model._register_gradient_hooks()

losses = []
#定义优化器SGD，损失函数交叉熵
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.001)
#输入数据处理：转为张量
train_data = DataLoader(Data,16)

print('start:')
global_step = 0

#开始训练
for epoch in range(1,51):
    epoch_loss = 0.0
    epoch_acc = 0.0
    num_batches = 0
    
    # 重置梯度统计信息
    model.reset_gradient_stats()
    
    #张量通过模型输出预测值
    for i, (x,y) in enumerate(train_data):
        x, y = x.to(device), y.to(device)  # 将数据和标签移动到GPU
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
    
    # 打印梯度统计信息
    model.print_gradient_summary(epoch)
    
    # 记录单epoch内各层平均梯度
    gradient_stats = model.get_gradient_stats()
    layer_names = []
    avg_gradients = []
    
    for name, stats in gradient_stats.items():
        if stats['norms']:  # 如果有梯度数据
            avg_gradient = sum(stats['norms']) / len(stats['norms'])
            layer_names.append(name)
            avg_gradients.append(avg_gradient)
    
    # 创建梯度分布图
    if layer_names and avg_gradients:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(layer_names)), avg_gradients)
        plt.xlabel('Network Layer')
        plt.ylabel('Average Gradient (×100)')
        plt.title(f'Epoch {epoch} - Average Gradient per Layer')
        plt.xticks(range(len(layer_names)), layer_names, rotation=45, ha='right')
        plt.tight_layout()
        
        # 将图像添加到TensorBoard
        writer.add_figure(f'Gradient_Distribution/Epoch_{epoch}', plt.gcf(), epoch)
        plt.close()
    
    print(f"第{epoch}轮的acc为{avg_acc:.4f}, loss为{avg_loss:.4f}")
    
    # 移除其他梯度记录，只保留梯度分布图

# 记录一些样本图像到TensorBoard
sample_batch = next(iter(train_data))
sample_images, sample_labels = sample_batch
sample_images = sample_images[:8].to(device)  # 取前8张图像并移动到GPU
sample_labels = sample_labels[:8]

# 创建图像网格（需要将GPU上的tensor移回CPU用于显示）
img_grid = torchvision.utils.make_grid(sample_images.cpu(), normalize=True, scale_each=True)
writer.add_image('Sample_Images', img_grid, 0)

# 记录预测结果
with torch.no_grad():
    sample_predictions = model(sample_images)
    sample_pred_labels = sample_predictions.argmax(1)

# 清理梯度监控钩子
model.remove_gradient_hooks()

# 关闭writer
writer.close()
print("训练完成！TensorBoard日志保存在:", log_dir)
print("运行 'tensorboard --logdir=runs' 查看可视化结果")




