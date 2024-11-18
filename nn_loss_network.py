import torchvision
from torch import nn
from torch.utils.data import DataLoader

import nn_seq

dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=1)

loss = nn.CrossEntropyLoss()
ycy = nn_seq.Ycy()
for data in dataloader:
    imgs, targets = data
    outputs = ycy(imgs)
    result_loss = loss(outputs, targets)
    # 前向传播forward，反向传播backward，是对损失值作反向传播，通过计算损失值的梯度，将其存在权重的gred属性中，然后使用优化器根据梯度更新权重
    result_loss.backward()
    print("ok")