#定义训练：包括损失函数的计算以及反向传播
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
<<<<<<< Updated upstream
from model import MyModel




model = MyModel()
=======
from torch.utils.tensorboard import SummaryWriter
from model import AlexNet
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
>>>>>>> Stashed changes
losses = []
#定义优化器SGD，损失函数MSE
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
#输入数据处理：转为张量
n_in, n_h, n_out, batch_size = 10, 5, 1, 10 
x = torch.randn(batch_size, n_in)
y = torch.tensor([[1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]])

print('start:')
for epoch in range(50):
#张量通过模型输出预测值
    y_pre = model(x)
#计算Loss
    loss = criterion(y_pre,y)
    losses.append(loss.item())
    print(f'epoch:{epoch},loss:{loss.item():.4f}')
#先对梯度清零，反向传播计算梯度
    optimizer.zero_grad()
    loss.backward()

#通过优化器，更新权重
    optimizer.step()

#可视化
plt.figure(figsize=(8, 5))
plt.plot(range(1, 51), losses, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid()
plt.show()

# 可视化预测结果与实际目标值对比
y_pred_final = model(x).detach().numpy()  # 最终预测值
y_actual = y.numpy()  # 实际值

plt.figure(figsize=(8, 5))
plt.plot(range(1, batch_size + 1), y_actual, 'o-', label='Actual', color='blue')
plt.plot(range(1, batch_size + 1), y_pred_final, 'x--', label='Predicted', color='red')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.grid()
plt.show()




