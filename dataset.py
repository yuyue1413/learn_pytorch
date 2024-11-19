from torch.utils.data import Dataset
from PIL import Image
import os

class MyDate(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)


    def __getitem__(self, index):
        img_name = self.img_path[index]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)

root_dir = 'F:\\Coding\\learn_pytorch\\data\\train'
cat_label_dir = 'Cat'
dog_label_dir = 'Dog'
cat_dataset = MyDate(root_dir, cat_label_dir)
dog_dataset = MyDate(root_dir, dog_label_dir)

train_dataset = cat_dataset + dog_dataset
img, label = train_dataset[0]
img.show()
print(label)