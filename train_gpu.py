import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

from model import *

# 定义训练的设备
# 使用GPU训练需要转移三样东西到GPU中，1、模型  2、数据(数据转移是在每次训练或测试中转移一个批量)  3、损失函数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
# 输出有两种写法，第一种比较常用
print("训练集的长度为： {}".format(train_data_size))
print("测试集的长度为：", test_data_size)

# 利用DataLoader加载数据
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# 创建网络模型
model = Ycy()
# 模型和损失函数不用另外赋值，只有数据需要另外赋值
# model = model.to(device)
model.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

# 优化器
lr = 0.01
# lr = 1e-2
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数 --其实就是epoch的循环值i
# total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tensorboard
# writer = SummaryWriter(log_dir='./logs')

for i in range(epoch):
    print("--------第 {} 轮训练开始--------".format(i+1))

    # 训练步骤开始
    model.train() # 当模型中有Dropout和BatchNorm时才有用
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        y_hat = model(imgs)
        loss = loss_fn(y_hat, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录每次训练的损失
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}， Loss：{}".format(total_train_step, loss.item()))
            # writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    model.eval() # 当模型中有Dropout和BatchNorm时才有用
    total_test_loss = 0.0
    total_test_accuracy = 0.0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            y_hat = model(imgs)
            loss = loss_fn(y_hat, targets)
            total_test_loss = total_test_loss + loss.item()

            # 精确率
            accuracy = (y_hat.argmax(dim=1) == targets).sum().item()
            total_test_accuracy = total_test_accuracy + accuracy

    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_test_loss / test_data_size))
    # writer.add_scalar("test_loss", total_test_loss, i)
    # writer.add_scalar("test_acc", total_test_loss / test_data_size, i)

    # 保存每一轮的模型
    torch.save(model, "./pths/model_{}.pth".format(i+1))
    print("第 {} 轮模型已保存！".format(i+1))

# writer.close()
