import torch
from torch import nn
from torch.nn import Conv2d, Flatten, MaxPool2d, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


class Ycy(nn.Module):
    def __init__(self):
        super(Ycy, self).__init__()
        # self.conv1 = Conv2d(3, 32, 5, padding=2)
        # self.maxpool1 = MaxPool2d(2)
        # self.conv2 = Conv2d(32, 32, 5, padding=2)
        # self.maxpool2 = MaxPool2d(2)
        # self.conv3 = Conv2d(32, 64, 5, padding=2)
        # self.maxpool3 = MaxPool2d(2)
        # self.flatten = Flatten()
        # self.linear1 = Linear(1024, 64)
        # self.linear2 = Linear(64, 10)

        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

writer = SummaryWriter("logs")
ycy = Ycy()

input = torch.randn(64, 3, 32, 32)
output = ycy(input)

writer.add_graph(ycy, input)
writer.close()
