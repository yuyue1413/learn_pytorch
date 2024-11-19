from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# 配置tensorboard
writer = SummaryWriter("logs")

img = Image.open('data/test/Cat/21.jpg')

# ToTenser使用
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image('Totensor', img_tensor)

# Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image('Normalize', img_norm)

# Resize
print(img.size)
trans_resize = transforms.Resize((111, 111))
img_resize = trans_resize(img)
# img PIL -> resize -> img_resize PIL
writer.add_image('Resize', trans_totensor(img_resize), 0)
print(img_resize)

# Compose - resize -第二种resize  等比缩放，就是不改变原图的宽高比
trans_resize_2 = transforms.Resize(111)
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])  # 按照列表顺序执行，先执行trans_resize_2
# PIL -> PIL -> tensor
img_resize_2 = trans_compose(img)
writer.add_image('Resize', img_resize_2, 1)

# RandomCrop 随机裁剪，从原图中随机裁剪111*50大小
trans_random = transforms.RandomCrop(111, 50)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image('RandomCrop', img_crop, i)




writer.close()
