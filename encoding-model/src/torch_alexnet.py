import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as I
import torch.nn.functional as F

class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
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
        return [c1,c2,c3,c4,c5,y1,f6,f7,f8]
        # return [c1,c2,c3,c4,c5,y1]
class alexnet_fmaps(nn.Module):
    def __init__(self, mean, std):
        super(alexnet_fmaps, self).__init__()
        self.mean = nn.Parameter(torch.as_tensor(mean), requires_grad=False)
        self.std = nn.Parameter(torch.as_tensor(std), requires_grad=False)
        self.extractor = AlexNet()

    def forward(self, _x):
        # _x = (_x - self.mean)/self.std
        # _x = (_x - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        # fmaps = self.extractor(_x)
        return self.extractor((_x - self.mean)/self.std)