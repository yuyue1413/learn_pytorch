import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader

import nn_seq

dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=1)

loss = nn.CrossEntropyLoss()
ycy = nn_seq.Ycy()
optim = torch.optim.SGD(ycy.parameters(), lr=0.01, )


# 进行epoch次总学习
for epoch in range(20):
    running_loss = 0.0
    # 对数据进行一次学习
    for data in dataloader:
        imgs, targets = data
        outputs = ycy(imgs)
        result_loss = loss(outputs, targets)
        # 反向传播前要梯度清零，不然会累加
        optim.zero_grad()
        # 前向传播forward，反向传播backward，是对损失值作反向传播，通过计算损失值的梯度，将其存在权重的gred属性中
        result_loss.backward()
        # 反向传播后使用优化器根据梯度更新权重
        optim.step()
        running_loss = running_loss + result_loss

    print(running_loss)