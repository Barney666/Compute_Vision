from __future__ import division
import torch
import torch.nn as nn
import os
from torchvision import models
from torchvision import datasets
from torchvision import transforms  # 包含resize、crop等常见的data augmentation操作
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

all_imgs = datasets.ImageFolder(os.path.join(data_dir, "train"), transforms.Compose([
    transforms.RandomResizedCrop(input_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()]))
loader = torch.utils.data.DataLoader(all_imgs, batch_size=batch_size, shuffle=True, num_workers=4)

data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(input_size),  # 即先随机采集，然后对裁剪得到的图像缩放为同一大小
        transforms.RandomHorizontalFlip(),  # 以给定的概率随机水平旋转给定的PIL的图像，默认为0.5
        transforms.ToTensor(),
        transforms.Normalize([0.477, 0.451, 0.403], [0.214, 0.211, 0.212])  # 训练集图片处理，随机裁剪/随机平移/转化为tensor/标准化
    ]),
    "val": transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.479, 0.452, 0.405], [0.214, 0.210, 0.211])  # 验证集处理，重构/中心裁剪/转化为tensor/标准化
    ])
}
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ["train", "val"]}
data_loaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                    batch_size=batch_size, shuffle=True, num_workers=4) for x in ["train", "val"]}

img = next(iter(data_loaders_dict["val"]))[0]  # 读取验证集中数据next(iterator[, default])，不断返回迭代器的下一个对象

unloader = transforms.ToPILImage()  # 从tensor到image的转化
plt.ion()  # 开启交互模式，可动态显示图像


def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)  # remove the fake batch dimension 降维
    image = unloader(image)
    plt.imshow(image)  # 在交互模式下plt.plot(x)或plt.imshow(x)是直接出图像，不需要plt.show()
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # 图像显示指定时间。使用ion()命令开启了交互模式，没有使用ioff()关闭的话，则图像会一闪而过，并不会常留


plt.figure()
imshow(img[31], title='Image')


def set_parameter_requires_grad(model, feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False  # pytorch，是否需要求该参数的梯度


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    if model_name == "resnet":
        #         model_ft = models.resnet18(pretrained=use_pretrained)  # 使用pytorch中预训练的resnet模型

        #         model_ft = models.resnet18(pretrained=False)
        #         pre = torch.load('resnet18-5c106cde.pth')
        #         model_ft.load_state_dict(pre)
        #         set_parameter_requires_grad(model_ft, feature_extract)
        #         num_ftrs = model_ft.fc.in_features  # 提取预训练模型中的固定参数
        #         model_ft.fc = nn.Linear(num_ftrs, num_classes)  # 修改分类类别数

        model_ft = models.vgg16(pretrained=False)
        pre = torch.load('vgg16-397923af.pth')
        model_ft.load_state_dict(pre)

        model_ft.classifier = nn.Sequential(  # 定义自己的分类层
            nn.Linear(512 * 7 * 7, 512),  # 512 * 7 * 7不能改变 ，由VGG16网络决定的，第二个参数为神经元个数可以微调
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, num_classes))
        input_size = 224
    else:
        print("model not implemented")
        return None, None
    return model_ft, input_size


model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)


def train_model(model, dataloaders, loss_fn, optimizer, num_epochs=5):
    best_acc = 0
    best_model_wts = copy.deepcopy(model.state_dict())  # 保留权重参数
    val_acc_history = []  # 验证集准确率列表
    for epoch in range(num_epochs):
        print("No." + str(epoch + 1) + " epoch starts.")
        for phase in ["train", "val"]:
            running_loss = 0
            running_corrects = 0
            if phase == "train":
                model.train()
            else:
                model.eval()
            for inputs, lables in dataloaders[phase]:
                inputs, lables = inputs.to(device), lables.to(device)  # 将输入和参数转至GPU加速运算
                with torch.autograd.set_grad_enabled(phase == "train"):  # with后紧跟的语句会被求值
                    outputs = model(inputs)
                    loss = loss_fn(outputs, lables)
                preds = outputs.argmax(dim=1)
                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds.view(-1) == lables.view(-1)).item()
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)
            print("Phase: {}, loss: {}, acc: {}".format(phase, epoch_loss, epoch_acc))

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "val":
                val_acc_history.append(epoch_acc)

    model.load_state_dict(best_model_wts)
    torch.save(model, 'model/my_vgg16.pkl')
    return model


model_ft = model_ft.to(device)
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=0.005, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()
train_model(model_ft, data_loaders_dict, loss_fn, optimizer, num_epochs=num_epochs)
torch.cuda.empty_cache()