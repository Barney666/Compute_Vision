from __future__ import division
import torch
import torch.nn as nn
import os
from torchvision import models
from torchvision import datasets
from torchvision import transforms  # 包含resize、crop等常见的data augmentation操作
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import argparse  # python标准库里面用来处理命令行参数的库
import torchvision
import matplotlib.pyplot as plt
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 在安装了cuda的设备上可进行gpu运算
os.environ["OMP_NUM_THREADS"] = "1"

data_dir = "./"  # 文件路径
model_name = "resnet"
num_classes = 80
batch_size = 128  # 数据打乱后分散成多少个batch
num_epochs = 50
feature_extract = True
input_size = 224

model = torch.load('model/my_vgg16.pkl')


class My_Dataset(Dataset):  # 继承Dataset
    def __init__(self, root_dir, transform=None):  # __init__是初始化该类的一些基础参数
        self.root_dir = root_dir  # 文件目录
        self.transform = transform  # 变换
        self.images = os.listdir(self.root_dir)  # 目录里的所有文件

    def __len__(self):  # 返回整个数据集的大小
        return len(self.images)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        image_index = self.images[index]  # 根据索引index获取该图片
        img_path = os.path.join(self.root_dir, image_index)  # 获取索引为index的图片的路径名test/0.jpg
        img = Image.open(img_path).convert('RGB')  # 读取该图片
        name = img_path.split('/')[-1].strip()

        return self.transform(img), name  # 返回该样本


trans = transforms.Compose([
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 验证集处理，重构/中心裁剪/转化为tensor/标准化
])

test_data = DataLoader(My_Dataset("test/", trans), batch_size=batch_size, shuffle=True, num_workers=4)

result = {}
for image, name in test_data:
    image = image.to(device)  # 将输入和参数转至GPU加速运算
    outputs = model(image)
    preds = outputs.argmax(dim=1).data.cpu().numpy().tolist()
    for i in range(len(preds)):
        result[int(name[i].split('.')[0])] = preds[i]
print(result)
with open("result.txt", "w") as file:
    for i in sorted(result):
        file.write(str(i) + ".jpg " + str(result[i]) + "\n")

