import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# 与transform联动
dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# 参数说明
# root:会在当前路径创建一个文件夹，把数据集存进去
# train:表示是否是训练集
# transform:就是要对数据集进行的转换，这里只有ToTensor一个转换
# download表示是否要从网络上下载,如果有了，就不会下载了，所以设为True就行
train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=True)

writer = SummaryWriter(log_dir='./logs')

print(test_set[0])
print(test_set.classes)
print(test_set.classes[3])
for i in range(10):
    img, target = test_set[i]
    writer.add_image('dataset_transform', img, i)




writer.close()