import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备的测试数据集
test_set = torchvision.datasets.CIFAR10("./dataset", False, transform=torchvision.transforms.ToTensor(), download=True)

# 常用参数解释
# dataset表示使用到的数据集
# batch_size表示批量大小，默认为1
# shuffle是否打乱，默认不打乱
# num_workers表示读取数据集时使用几个进程，默认为0，表示使用主进程
# drop_last表示最后剩下不足一个批量是是否丢弃，比如批量大小为64，最后还有20张，False则保留，并把这20张作为最后一个批量
test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

# for i in range(4):
#     img, label = test_set[i]
#     print(img.shape)
#     print(label)

writer = SummaryWriter("logs")
step = 0
for data in test_loader:
    imgs, labels = data
    writer.add_images("test_data", imgs, step)
    step = step + 1

writer.close()
