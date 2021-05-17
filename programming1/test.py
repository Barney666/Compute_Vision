from __future__ import division
import torch
import torch.nn as nn
import os
import cv2 as cv
from torchvision import models
from torchvision import datasets
from torchvision import transforms  #包含resize、crop等常见的data augmentation操作
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import argparse  #python标准库里面用来处理命令行参数的库
import torchvision
import matplotlib.pyplot as plt
import copy
os.environ["OMP_NUM_THREADS"] = "1"


data_dir = "./"  #文件路径
model_name = "resnet"
num_classes = 5
batch_size = 5     # 数据打乱后分散成多少个batch
num_epochs = 15
feature_extract = True
input_size = 224
data_transforms = {
     "train":transforms.Compose([
        transforms.RandomResizedCrop(input_size),   # 即先随机采集，然后对裁剪得到的图像缩放为同一大小
        transforms.RandomHorizontalFlip(),          # 以给定的概率随机水平旋转给定的PIL的图像，默认为0.5
        transforms.ToTensor(),
        transforms.Normalize([0.477,0.451,0.403],[0.214,0.211,0.212])  # 训练集图片处理，随机裁剪/随机平移/转化为tensor/标准化
         ]),
     "val":transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.479,0.452,0.405],[0.214,0.210,0.211])  # 验证集处理，重构/中心裁剪/转化为tensor/标准化
         ])
}



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


# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x]) for x in ["val"]}
# dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
#                     batch_size=batch_size, shuffle=True,num_workers=4) for x in ["val"] }
#
# model = models.resnet18(pretrained=True)
# for inputs, lables in dataloaders_dict["val"]:
#     out = model(inputs)
#     preds = out.argmax(1)
#     print(preds.numpy())


trans = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 验证集处理，重构/中心裁剪/转化为tensor/标准化
])


model = models.resnet18(pretrained=True)
#
test_data = DataLoader(My_Dataset("test/", trans), batch_size=batch_size, shuffle=True,num_workers=4)


result = {}
for image, name in test_data:
    # inputs, lables = inputs.to(device), lables.to(device)  # 将输入和参数转至GPU加速运算
    print(image.shape)
    outputs = model(image)
    preds = outputs.argmax(dim=1).numpy().tolist()
    for i in range(batch_size):
        result[name[i]] = preds[i]
print(result)
with open("result.txt", "w") as file:
    for i in sorted(result, cmp=lambda x, y: int(str(x[0])+str(y[0])) < int(str(y[0])+str(x[0]))):
        file.write(str(i) + " " + str(result[i]) + "\n")