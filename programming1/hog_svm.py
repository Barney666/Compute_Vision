import glob
import platform
import time
from PIL import Image
from skimage.feature import hog
import numpy as np
import os
import joblib
from sklearn.svm import LinearSVC
from sklearn import svm
import shutil
import sys

# 训练集图片的位置
train_image_path = 'train/'
# 测试集图片的位置
test_image_path = 'test/'
# 预测图片的位置
predict_image_path = 'predict/'


train_feat_path = 'train_feat/'
test_feat_path = 'test_feat/'
model_path = 'model/'


# 获得图片列表
def get_image_list(filePath, nameList):
    print('read image from ', filePath)
    img_list = []
    for name in nameList:
        temp = Image.open(os.path.join(filePath, name)).convert('L')
        img_list.append(temp.copy())
        temp.close()
    return img_list


# 变成灰度图片
def rgb2gray(image):
    # gray = image[:, :, 0] * 0.2989 + image[:, :, 1] * 0.5870 + image[:, :, 2] * 0.1140
    try:
        gray = image[:, :, 0] * 0.2989 + image[:, :, 1] * 0.5870 + image[:, :, 2] * 0.1140
    except:
        gray = image[:, :] * 0.2989 + image[:, :] * 0.5870 + image[:, :] * 0.1140
    return gray


# 提取特征并保存
def get_feat(image_list, name_list, label_list, save_path):
    i = 0
    for image in image_list:
        image = np.array(image)         # 转成RGB三维数组
        gray = image / 255.0

        # 这句话根据你的尺寸改改
        fd = hog(gray, orientations=12, block_norm='L1', pixels_per_cell=[8, 8], cells_per_block=[4, 4], visualize=False, transform_sqrt=True)
        fd = np.concatenate((fd, [label_list[i]]))      # 默认按列拼接，fd是1x4800的矩阵，这里直接在最后面加一列
        fd_name = name_list[i] + '.feat'
        fd_path = os.path.join(save_path, fd_name)
        joblib.dump(fd, fd_path)
        print("No." + str(i) + " picture's feat has been extracted.")
        i += 1
    print(save_path + " features are extracted and saved.")


# 获得图片名称与对应的类别 针对val_anno.txt
def get_name_label(file_path):
    print("read label from ", file_path)
    name_list = []
    label_list = []
    with open(file_path, 'r') as f:
        items = f.readlines()
        items = [item.strip().split() for item in items]
    for item in items:
        name_list.append(item[0])
        label_list.append(item[1])
    return name_list, label_list


def extract_feat():
    # train_feat
    for root, dirs, files in os.walk("train"):
        for name in dirs:       # 1-80
            path = os.path.join(root, name)     # train/61
            train_name = os.listdir(path)
            train_label = [name] * len(train_name)
            train_image = get_image_list(train_image_path + name, train_name)
            get_feat(train_image, train_name, train_label, train_feat_path + name)
    # test_feat
    test_name, test_label = get_name_label("val_anno.txt")
    test_image = get_image_list(test_image_path, test_name)
    get_feat(test_image, test_name, test_label, test_feat_path)


def train():
    t0 = time.time()
    correct_number = 0
    total = 0
    features = []
    labels = []
    result_list = []

    for root, dirs, files in os.walk("train_feat"):
        for name in dirs:  # 1-80
            for feat_path in glob.glob(os.path.join(train_feat_path + name, '*.feat')):
                feat = joblib.load(feat_path)
                features.append(feat[:-1])
                labels.append(feat[-1])

    # 同样顺序打乱
    state = np.random.get_state()
    np.random.shuffle(features)
    np.random.set_state(state)
    np.random.shuffle(labels)

    print("Training a Linear LinearSVM Classifier.")
    # print("Training a SVM Classifier.")

    clf = LinearSVC()
    clf.fit(features, labels)

    # 下面的代码是保存模型的
    joblib.dump(clf, model_path + 'model')

    # 下面的代码是加载模型  可以注释上面的代码   直接进行加载模型  不进行训练
    # clf = joblib.load(model_path+'model')

    print("训练之后的模型存放在model文件夹中")

    # exit()

    for feat_path in glob.glob(os.path.join(test_feat_path, '*.feat')):
        total += 1
        image_name = feat_path.split('/')[1].split('.feat')[0]

        data_test = joblib.load(feat_path)
        data_test_feat = data_test[:-1].reshape((1, -1)).astype(np.float64)
        result = clf.predict(data_test_feat)
        result_list.append(image_name + ' ' + result[0] + '\n')

        if int(result[0]) == int(data_test[-1]):
            correct_number += 1

    with open('result.txt', 'w') as f:
        f.writelines(result_list)
    print('每张图片的识别结果存放在result.txt里面')

    rate = float(correct_number) / total
    t1 = time.time()
    print('准确率是： %f' % rate)
    print('耗时是 : %f' % (t1 - t0))



if __name__ == '__main__':
    shutil.rmtree(train_feat_path)
    shutil.rmtree(test_feat_path)
    shutil.rmtree(model_path)

    os.mkdir(train_feat_path)
    for i in range(80):
        os.mkdir(train_feat_path + str(i))
    os.mkdir(test_feat_path)
    os.mkdir(model_path)

    extract_feat()
    train()