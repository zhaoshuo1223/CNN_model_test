
from dataset import TestData
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.nn import functional as F

module_path = r"C:\Users\zhaoshuo\Desktop\cv_test\save_model\SimpleNet\module50"

test_loader = DataLoader(TestData,16)
module = torch.load(module_path,weights_only=False)

acces = []
test_loss = 0
correct = 0
with torch.no_grad():
    for data,label in test_loader:
        output=module(data)
        test_loss+=F.cross_entropy(output,label).item()
        pred=output.argmax(dim=1)
        correct+=pred.eq(label.view_as(pred)).sum().item()
    test_loss/=len(test_loader.dataset)
    print('Test_average_loss:{:.4f},Accuracy:{:3f}\n'.format(
        test_loss,100*correct/len(test_loader.dataset)
    ))
    acc=100*correct/len(test_loader.dataset)
    acces.append(acc)
sum = 0
for i in range(len(acces)):
    sum += acces[i]
acc = sum / len(acces)
print(f'acc:{acc}')