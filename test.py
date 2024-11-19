import torchvision.transforms
from PIL import Image
from model import *

img_path = "./data/dog.png"
image = Image.open(img_path).convert('RGB')
print(image)

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()
])
image = transforms(image)
# 这一步容易遗忘，模型是需要一个batch_size的，所以需要给图片添加一个维度
image = torch.reshape(image, (1, 3, 32, 32))

# 将模型映射到cpu上，map就是映射的意思
model = torch.load("pths/model_10.pth", map_location=torch.device('cpu'))
print(model)

# 加载的模型在GPU上，有两种方式，可以将模型放到CPU中，也可以将输入放到GPU中，保持一致
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# image = image.to(device)

# 26、27行最好写上，不写也不会报错，但是这是个好的习惯，如果模型中有Dropout就会报错
model.eval()
with torch.no_grad():
    output = model(image)
print(output)
print(output.argmax(1))