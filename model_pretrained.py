import torch
import torchvision
from torch import nn

# train_data = torchvision.datasets.ImageNet("./dataset_imagenet", split="train", download=True,
#                                            transform=torchvision.transforms.ToTensor())

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)
resnet = torchvision.models.resnet50(pretrained=True)

train_data = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)

# 如何改动经典的网络模型，有两种方式，第一种是在后面加层，第二种是改动这个模型
# 1、在后面添加层
# 1)在最后加一层
# vgg16_true.add_module("add_linear", nn.Linear(1000, 10))
# 2)在某个Sequential的最后添加,如在classifier层的最后添加
vgg16_true.classifier.add_module("add_linear", nn.Linear(1000, 10))

# 2、直接修改模型
vgg16_false.classifier[6] = nn.Linear(4096, 10)

# print(vgg16_false)


# 模型保存和加载
# 保存方式一,保存模型结构+模型参数
torch.save(vgg16_true, "./pths/vgg16_method1.pth")
# 加载模型
model1 = torch.load("./pths/vgg16_method1.pth")


# 保存方式二，保存模型参数（官方推荐）
torch.save(vgg16_true.state_dict(), "./pths/vgg16_method2.pth")
# 加载模型，需要先创建一样的模型，因为保存的是字典格式的参数
model2 = torchvision.models.vgg16(pretrained=False)
model2.classifier.add_module("add_linear", nn.Linear(1000, 10))

# 直接给模型加载这个字典
model2.load_state_dict(torch.load("./pths/vgg16_method2.pth"))
print(model2)


# 陷阱
# 第一种保存方式有一个陷阱，如果是自定义的模型，在加载pth文件时，需要有这个模型的类（比如import这个模型类），才可以直接写model = torch.load()，没有这个类则会报错

