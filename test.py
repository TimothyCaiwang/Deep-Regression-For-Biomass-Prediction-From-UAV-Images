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
import copy
from ViT_pytorch import ViT
from T2TViT_pytorch import T2TViT
import scipy.io as scio

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))
# 对数据进行转换处理
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
        transforms.Normalize(mean=(8.618, 10.301, 7.600, 18.453, 35.113),
                             std=(7.247, 6.129, 4.751, 11.958, 23.874))
    ]
)
'''

transf2 = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=(8.618, 10.301, 7.600),
                             std=(7.247, 6.129, 4.751))
    ]
)


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


def preict_one_img(img_path, net):
    net.to(device)
    img_datam = scio.loadmat(img_path)
    img_datao = img_datam['sample_r_mat']
    img_data = img_datao[:, :, 0:3]
    # img_data_RGB = img_datao[:, :, 0:3]
    # img_data_DSM = img_datao[:, :, 5]
    # img_data_DSM = np.expand_dims(img_data_DSM, axis=2)
    # img_data = np.concatenate((img_data_RGB, img_data_DSM), axis=2)
    idata = transf2(img_data)
    # print(idata.shape)
    idata = idata.view(1, 3, 224, 224)
    img = idata.to(device)
    out = net(img)
    return out


model_r18o = models.resnet18(pretrained=False)
model_r18 = Myresnet18(model_r18o)
pretrain_r18 = 'E:\\Caiwang_ZHENG\\Deep_Regression_Drone_2020_to_2022\\log_RGB\\resnet18_best_mae.pth'

# model_r18.load_state_dict(copy.deepcopy(torch.load("resnet18last.pth", device)))
model_r18.load_state_dict(torch.load(pretrain_r18))
model_r18.eval()

model_r34o = models.resnet34(pretrained=False)
model_r34 = Myresnet18(model_r34o)
pretrain_r34 = 'E:\\Caiwang_ZHENG\\Deep_Regression_Drone_2020_to_2022\\log_RGB\\resnet34_best_mae.pth'

# model_r18.load_state_dict(copy.deepcopy(torch.load("resnet18last.pth", device)))
model_r34.load_state_dict(torch.load(pretrain_r34))
model_r34.eval()

model_r50o = models.resnet50(pretrained=False)
model_r50 = Myresnet50(model_r50o)
pretrain_r50 = 'E:\\Caiwang_ZHENG\\Deep_Regression_Drone_2020_to_2022\\log_RGB\\resnet50_best_mae.pth'

model_r50.load_state_dict(torch.load(pretrain_r50))
model_r50.eval()


model_vgg16o = models.vgg16(pretrained=False)
model_vgg16 = Myvgg16(model_vgg16o)
pretrain_v16 = 'E:\\Caiwang_ZHENG\\Deep_Regression_Drone_2020_to_2022\\log_RGB\\vgg16_best_mae.pth'

model_vgg16.load_state_dict(torch.load(pretrain_v16))
model_vgg16.eval()


model_dno = models.DenseNet()
model_dn = Mydesnet(model_dno)
pretrain_dn = 'E:\\Caiwang_ZHENG\\Deep_Regression_Drone_2020_to_2022\\log_RGB\\desnet_best_mae.pth'

model_dn.load_state_dict(torch.load(pretrain_dn))
model_dn.eval()

model_mno = models.mobilenet_v2(pretrained=False)
model_mn = Mymobilenet(model_mno)
pretrain_mn = 'E:\\Caiwang_ZHENG\\Deep_Regression_Drone_2020_to_2022\\log_RGB\\mobilenet_best_mae.pth'

model_mn.load_state_dict(torch.load(pretrain_mn))
model_mn.eval()

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
pretrain_ViT = 'E:\\Caiwang_ZHENG\\Deep_Regression_Drone_2020_to_2022\\log_RGB\\ViT_best_mae.pth'
model_ViT.load_state_dict(torch.load(pretrain_ViT))
model_ViT.eval()


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
pretrain_T2TViT = 'E:\\Caiwang_ZHENG\\Deep_Regression_Drone_2020_to_2022\\log_RGB\\T2TViT_best_mae.pth'
model_T2TViT.load_state_dict(torch.load(pretrain_T2TViT))
model_T2TViT.eval()


fname3 = 'test_2020_to_2022_data.txt'

gt_bio = []
pre_bio1 = []
pre_bio2 = []
pre_bio3 = []
pre_bio4 = []
pre_bio5 = []
pre_bio6 = []
pre_bio7 = []
pre_bio8 = []


with open(fname3, 'r+', encoding='utf-8') as f:
    for line in f.readlines():
        a = line[:-1].split(' ')
        # print(a[0])
        test_img_path = a[0]
        test_biomass = float(a[1])
        pre_biomass1 = preict_one_img(test_img_path, model_r18).cpu().detach().numpy()
        pre_biomass2 = preict_one_img(test_img_path, model_r34).cpu().detach().numpy()
        pre_biomass3 = preict_one_img(test_img_path, model_r50).cpu().detach().numpy()
        pre_biomass4 = preict_one_img(test_img_path, model_vgg16).cpu().detach().numpy()
        pre_biomass5 = preict_one_img(test_img_path, model_dn).cpu().detach().numpy()
        pre_biomass6 = preict_one_img(test_img_path, model_mn).cpu().detach().numpy()
        pre_biomass7 = preict_one_img(test_img_path, model_ViT).cpu().detach().numpy()
        pre_biomass8 = preict_one_img(test_img_path, model_T2TViT).cpu().detach().numpy()

        gt_bio.append(test_biomass)
        pre_bio1.append(pre_biomass1.squeeze())
        pre_bio2.append(pre_biomass2.squeeze())
        pre_bio3.append(pre_biomass3.squeeze())

        pre_bio4.append(pre_biomass4.squeeze())
        pre_bio5.append(pre_biomass5.squeeze())
        pre_bio6.append(pre_biomass6.squeeze())

        pre_bio7.append(pre_biomass7.squeeze())
        pre_bio8.append(pre_biomass8.squeeze())
scio.savemat('test_pre_gt_RGB_mae.mat', {'pre1': pre_bio1, 'pre2': pre_bio2, 'pre3': pre_bio3, 'pre4': pre_bio4, 'pre5': pre_bio5,
                                         'pre6': pre_bio6, 'pre7': pre_bio7, 'pre8': pre_bio8, 'gt': gt_bio})
