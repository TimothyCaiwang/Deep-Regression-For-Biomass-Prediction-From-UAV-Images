# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 20:34:55 2022

@author: caiwangzheng
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from torch.utils import data
import scipy.io as scio
import torchvision.models as models
from ViT_pytorch import ViT
from T2TViT_pytorch import T2TViT

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))
# 对数据进行转换处理


transf2 = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=(8.618, 10.301, 7.600),
                             std=(7.247, 6.129, 4.751))
    ]
)


'''
transf2 = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=(8.618, 10.301, 7.600, 18.453, 35.113, 15.909),
                             std=(7.247, 6.129, 4.751, 11.958, 23.874, 0.236))
        ]
)



transf2 = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=(8.618, 10.301, 7.600, 15.909),
                             std=(7.247, 6.129, 4.751, 0.236))
        ]
)
'''


class Myresnet18(nn.Module):
    def __init__(self, model):  # 此处的model参数是已经加载了预训练参数的模型，方便继承预训练成果
        super(Myresnet18, self).__init__()
        # 取掉model的后两层
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])

        self.Linear_layer = nn.Linear(512, 1, bias=False)

    def forward(self, x):
        x = self.resnet_layer(x)
        x = x.view(x.size(0), -1)
        x = self.Linear_layer(x)
        return x


class Myresnet34(nn.Module):
    def __init__(self, model):  # 此处的model参数是已经加载了预训练参数的模型，方便继承预训练成果
        super(Myresnet34, self).__init__()
        # 取掉model的后两层
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])

        self.Linear_layer = nn.Linear(512, 1, bias=False)

    def forward(self, x):
        x = self.resnet_layer(x)
        x = x.view(x.size(0), -1)
        x = self.Linear_layer(x)
        return x


class Myresnet50(nn.Module):
    def __init__(self, model):  # 此处的model参数是已经加载了预训练参数的模型，方便继承预训练成果
        super(Myresnet50, self).__init__()
        # 取掉model的后两层
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])

        self.Linear_layer = nn.Linear(2048, 1, bias=False)

    def forward(self, x):
        x = self.resnet_layer(x)
        x = x.view(x.size(0), -1)
        x = self.Linear_layer(x)
        return x


class Myvgg16(nn.Module):
    def __init__(self, model):  # 此处的model参数是已经加载了预训练参数的模型，方便继承预训练成果
        super(Myvgg16, self).__init__()
        # 取掉model的后两层
        self.vgg_layer = nn.Sequential(*list(model.children())[:-1])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.Linear_layer = nn.Linear(512, 1, bias=False)

    def forward(self, x):
        x = self.vgg_layer(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.Linear_layer(x)
        return x


class Mydesnet(nn.Module):
    def __init__(self, model):  # 此处的model参数是已经加载了预训练参数的模型，方便继承预训练成果
        super(Mydesnet, self).__init__()
        # 取掉model的后两层
        self.md_layer = nn.Sequential(*list(model.children())[:-1])
        print(self.md_layer)
        self.Linear_layer = nn.Linear(1024, 1, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.md_layer(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.Linear_layer(x)
        return x


class Mymobilenet(nn.Module):
    def __init__(self, model):  # 此处的model参数是已经加载了预训练参数的模型，方便继承预训练成果
        super(Mymobilenet, self).__init__()
        # 取掉model的后两层
        self.mn_layer = nn.Sequential(*list(model.children())[:-1])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # print(self.mn_layer)

        self.Linear_layer = nn.Linear(1280, 1, bias=False)

    def forward(self, x):
        x = self.mn_layer(x)
        x = self.avg_pool(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.Linear_layer(x)
        return x


class Mydatasetpro(data.Dataset):
    # 类初始化
    def __init__(self, img_paths, labels, transform):
        self.imgs = img_paths
        self.labels = labels
        self.transforms = transform

    # 进行切片
    def __getitem__(self, index):  # 根据给出的索引进行切片，并对其进行数据处理转换成Tensor，返回成Tensor
        img = self.imgs[index]
        # img = img[:, :, 0:6]
        label = self.labels[index]
        img_datam = scio.loadmat(img)
        img_datao = img_datam['sample_r_mat']
        # print('xxxxxxx')
        # print(img_data.shape)
        img_data = img_datao[:, :, 0:3]
        # img_data_RGB = img_datao[:, :, 0:3]
        # img_data_DSM = img_datao[:, :, 5]
        # img_data_DSM = np.expand_dims(img_data_DSM, axis=2)
        # img_data = np.concatenate((img_data_RGB, img_data_DSM), axis=2)
        # print('xxxxxxx')
        # print(img_data.shape)
        idata = self.transforms(img_data)
        return idata, label

    # 返回长度
    def __len__(self):
        return len(self.imgs)


def train_and_val(epochs, model, train_loader, val_loader, criterion1, criterion2, optimizer, out_name):
    torch.cuda.empty_cache()
    train_mse = []
    val_mse = []
    train_mae = []
    val_mae = []
    best_mae = 100000000000
    best_mse = 100000000000

    model.to(device)
    fit_time = time.time()
    for e in range(epochs):
        since = time.time()
        running_loss = 0
        train_maelosses = 0
        with tqdm(total=len(train_loader)) as pbar:
            for image, label in train_loader:
                # training phase

                #                 images, labels = data
                #             optimizer.zero_grad()
                #             logits = net(images.to(device))
                #             loss = loss_function(logits, labels.to(device))
                #             loss.backward()
                #             optimizer.step()

                model.train()
                optimizer.zero_grad()
                image = image.to(device)
                label = label.to(device)
                label = label.to(torch.float32)
                # forward
                output = model(image).to(torch.float32)
                mseloss = criterion1(output, label)
                maeloss = criterion2(output, label)
                # predict_t = torch.max(output, dim=1)[1]

                # backward
                mseloss.backward()
                optimizer.step()  # update weight

                running_loss += mseloss.item()
                train_maelosses += maeloss.item()
                pbar.update(1)

        model.eval()
        val_mselosses = 0
        val_maelosses = 0
        # validation loop
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as pb:
                for image, label in val_loader:
                    image = image.to(device)
                    label = label.to(device)
                    output = model(image).to(torch.float32)

                    # loss
                    mseloss = criterion1(output, label)
                    maeloss = criterion2(output, label)
                    # predict_v = torch.max(output, dim=1)[1]

                    val_mselosses += mseloss.item()
                    val_maelosses += maeloss.item()
                    pb.update(1)

            # calculatio mean for each batch
            train_mse.append(running_loss / len(train_loader))
            val_mse.append(val_mselosses / len(val_loader))

            train_mae.append(train_maelosses / len(train_loader))
            val_mae.append(val_maelosses / len(val_loader))
            last_name = out_name + "epoch" + str(e) + ".pth"
            best_name_mae = out_name + "best_mae.pth"
            best_name_mse = out_name + "best_mse.pth"

            # torch.save(model.state_dict(), last_name)
            if best_mae > val_maelosses / len(val_loader):
                best_mae = val_maelosses / len(val_loader)
                torch.save(model.state_dict(), best_name_mae)

            if best_mse > val_mselosses / len(val_loader):
                best_mse = val_mselosses / len(val_loader)
                torch.save(model.state_dict(), best_name_mse)

            print("Epoch:{}/{}..".format(e + 1, epochs),
                  "Train MAE: {:.3f}..".format(train_maelosses / len(train_loader)),
                  "Val MAE: {:.3f}..".format(val_maelosses / len(val_loader)),
                  "Train MSE: {:.3f}..".format(running_loss / len(train_loader)),
                  "Val MSE: {:.3f}..".format(val_mselosses / len(val_loader)),
                  "Time: {:.2f}s".format((time.time() - since)))

    historyr = {'train_mse': train_mse, 'val_mse': val_mse, 'train_mae': train_mae, 'val_mae': val_mae}
    print('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))
    out_loss = out_name + 'loss.npy'
    out_loss_data = [train_mse, val_mse, train_mae, val_mae]
    np.save(out_loss, out_loss_data)
    return historyr


def preict_one_img(img_path, net):
    img_datam = scio.loadmat(img_path)
    img_datao = img_datam['sample_r_mat']
    img_data = img_datao[:, :, 0:6]
    idata = transf2(img_data)
    img = idata.to(device)
    out = net(img) / 100
    return out


Train_imgs_path = []
Train_labels = []

fname1 = 'Train_2020_to_2022_data.txt'
with open(fname1, 'r+', encoding='utf-8') as f:
    for line in f.readlines():
        a = line[:-1].split(' ')
        Train_imgs_path.append(a[0])
        Train_labels.append(float(a[1]))

Val_imgs_path = []
Val_labels = []

fname2 = 'Valid_2020_to_2022_data.txt'
with open(fname2, 'r+', encoding='utf-8') as f:
    for line in f.readlines():
        a = line[:-1].split(' ')
        Val_imgs_path.append(a[0])
        Val_labels.append(float(a[1]))

BATCH_SIZE = 32
train_dataset = Mydatasetpro(Train_imgs_path, Train_labels, transf2)
train_datalodaer = data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_dataset = Mydatasetpro(Val_imgs_path, Val_labels, transf2)
val_datalodaer = data.DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

mse_function = nn.MSELoss()
mae_function = nn.L1Loss()
epoch = 80


model_r18o = models.resnet18(pretrained=False)
model_r18 = Myresnet18(model_r18o)
print(model_r18)
optimizer1 = optim.Adam(model_r18.parameters(), lr=0.0001)  # 设置优化器和学习率
out_name1 = 'resnet18_'
history1 = train_and_val(epoch, model_r18, train_datalodaer, val_datalodaer, mse_function, mae_function, optimizer1, out_name1)
del model_r18o
del model_r18


model_r34o = models.resnet34(pretrained=False)
model_r34 = Myresnet34(model_r34o)
optimizer2 = optim.Adam(model_r34.parameters(), lr=0.0001)  # 设置优化器和学习率
out_name2 = 'resnet34_'
history2 = train_and_val(epoch, model_r34, train_datalodaer, val_datalodaer, mse_function, mae_function, optimizer2, out_name2)
del model_r34o
del model_r34

model_r50o = models.resnet50(pretrained=False)
model_r50 = Myresnet50(model_r50o)
optimizer3 = optim.Adam(model_r50.parameters(), lr=0.0001)  # 设置优化器和学习率
out_name3 = 'resnet50_'
history3 = train_and_val(epoch, model_r50, train_datalodaer, val_datalodaer, mse_function, mae_function, optimizer3, out_name3)
del model_r50o
del model_r50


model_vgg16o = models.vgg16(pretrained=False)
model_vgg16 = Myvgg16(model_vgg16o)
optimizer4 = optim.Adam(model_vgg16.parameters(), lr=0.0001)  # 设置优化器和学习率
out_name4 = 'vgg16_'
history4 = train_and_val(epoch, model_vgg16, train_datalodaer, val_datalodaer, mse_function, mae_function, optimizer4, out_name4)
del model_vgg16o
del model_vgg16


model_dno = models.DenseNet()
model_dn = Mydesnet(model_dno)
optimizer5 = optim.Adam(model_dn.parameters(), lr=0.0001)  # 设置优化器和学习率
out_name5 = 'desnet_'
history5 = train_and_val(epoch, model_dn, train_datalodaer, val_datalodaer, mse_function, mae_function, optimizer5, out_name5)
del model_dno
del model_dn

model_mno = models.mobilenet_v2(pretrained=False)
model_mn = Mymobilenet(model_mno)
optimizer6 = optim.Adam(model_mn.parameters(), lr=0.0001)  # 设置优化器和学习率
out_name6 = 'mobilenet_'
history6 = train_and_val(epoch, model_mn, train_datalodaer, val_datalodaer, mse_function, mae_function, optimizer6, out_name6)
del model_mno
del model_mn

model_ViT = ViT(
    image_size=224,
    patch_size=56,
    num_classes=1,
    dim=1024,
    depth=5,
    heads=8,
    mlp_dim=2048,
    channels=3,
    dropout=0.1,
    emb_dropout=0.1
)
optimizer7 = optim.Adam(model_ViT.parameters(), lr=0.0001)  # 设置优化器和学习率
out_name7 = 'ViT_'
history7 = train_and_val(epoch, model_ViT, train_datalodaer, val_datalodaer, mse_function, mae_function, optimizer7, out_name7)
del model_ViT


model_T2TViT = T2TViT(
    dim=512,
    image_size=224,
    depth=5,
    heads=8,
    mlp_dim=512,
    channels=3,
    num_classes=1,
    t2t_layers=((7, 4), (3, 2), (3, 2))
    # tuples of the kernel size and stride of each consecutive layers of the initial token to token module
)

optimizer8 = optim.Adam(model_T2TViT.parameters(), lr=0.0001)  # 设置优化器和学习率
out_name8 = 'T2TViT_'
history8 = train_and_val(epoch, model_T2TViT, train_datalodaer, val_datalodaer, mse_function, mae_function, optimizer8, out_name8)
del model_T2TViT
