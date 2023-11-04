import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as I
import torch.nn.functional as F

class AlexNet_filter(nn.Module):

    def __init__(self, fmask6, fmask7, fmask8, num_classes=1000):
        super(AlexNet_filter, self).__init__()
        # fmask6 = torch.from_numpy(np.loadtxt(
        #     "/home/mufan/VDisk1/Mufan/Mufan_old/code/nsdmaster1/output/AlexNet/FFA/multisubject/gnet_mpf_evc_Jun-05-2023_1708/feature_mask_layer6_512.txt",
        #     dtype=int, delimiter=","))
        # fmask7 = torch.from_numpy(np.loadtxt(
        #     "/home/mufan/VDisk1/Mufan/Mufan_old/code/nsdmaster1/output/AlexNet/FFA/multisubject/gnet_mpf_evc_Jun-05-2023_1708/feature_mask_layer7_512.txt",
        #     dtype=int, delimiter=","))
        # fmask8 = torch.from_numpy(np.loadtxt(
        #     "/home/mufan/VDisk1/Mufan/Mufan_old/code/nsdmaster1/output/AlexNet/FFA/multisubject/gnet_mpf_evc_Jun-05-2023_1708/feature_mask_layer8_512.txt",
        #     dtype=int, delimiter=","))
        # self.fmask6 = fmask6
        # self.fmask7 = fmask7
        # self.fmask8 = fmask8
        self.fmask6 = nn.Parameter(torch.as_tensor(fmask6), requires_grad=False)
        self.fmask7 = nn.Parameter(torch.as_tensor(fmask7), requires_grad=False)
        self.fmask8 = nn.Parameter(torch.as_tensor(fmask8), requires_grad=False)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.avgpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d((6, 6))
        )
        self.fc6 = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True)
        )
        self.fc7 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True)
        )
        self.fc8 = nn.Sequential(
            nn.Linear(4096, int(num_classes))  # torch.empty to return a tensor without initial
        )

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        y1 = self.avgpool(c5)
        y2 = torch.flatten(y1, 1)
        # y2 = y2.to('cuda:1')
        f6 = self.fc6(y2)
        f7 = self.fc7(f6)
        f8 = self.fc8(f7)
        f6 = torch.unsqueeze(torch.unsqueeze(f6, 2), 3)
        f7 = torch.unsqueeze(torch.unsqueeze(f7, 2), 3)
        f8 = torch.unsqueeze(torch.unsqueeze(f8, 2), 3)
        f6 = torch.index_select(f6, dim=1, index=self.fmask6)
        f7 = torch.index_select(f7, dim=1, index=self.fmask7)
        f8 = torch.index_select(f8, dim=1, index=self.fmask8)
        #select the max varance feature in full conection layer 6,7,8
        # n6 = (f6.cpu().detach().numpy())[:, self.fmask6, :, :]
        # n7 = (f7.cpu().detach().numpy())[:, self.fmask7, :, :]
        # n8 = (f8.cpu().detach().numpy())[:, self.fmask8, :, :]
        # l6 = n6[:, self.fmask6, :, :]
        # l7 = n7[:, self.fmask7, :, :]
        # l8 = n8[:, self.fmask8, :, :]
        # f6 = torch.from_numpy(n6).to('cuda:0')
        # f7 = torch.from_numpy(n7).to('cuda:0')
        # f8 = torch.from_numpy(n8).to('cuda:0')

        return [c1,c2,c3,c4,c5,y1,f6,f7,f8]
        # return [c1,c2,c3,c4,c5,y1]
class alexnet_fmaps_filter(nn.Module):
    def __init__(self, mean, std,fmask6,fmask7,fmask8):
        super(alexnet_fmaps_filter, self).__init__()
        self.mean = nn.Parameter(torch.as_tensor(mean), requires_grad=False)
        self.std = nn.Parameter(torch.as_tensor(std), requires_grad=False)
        self.extractor = AlexNet_filter(fmask6,fmask7,fmask8)

    def forward(self, _x):
        # _x = (_x - self.mean)/self.std
        # _x = (_x - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        # fmaps = self.extractor(_x)
        return self.extractor((_x - self.mean)/self.std)