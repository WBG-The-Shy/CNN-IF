import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, utils
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms
from pathlib import Path
from torchvision import models
from PIL import Image
import torch.nn.functional as F
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, utils
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms
from pathlib import Path
from PIL import Image
import torch.nn.functional as F

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=[1, 1, 1], padding=[0, 1, 0], first=False) -> None:
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=padding[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),  # to save memory,replace in original place
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=padding[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),  # to save memory,replace in original place
            nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=stride[2], padding=padding[2], bias=False),
            nn.BatchNorm2d(out_channels * 4)
        )

        # shortcut part
        # if the dimention is not same, it need to high dim
        self.shortcut = nn.Sequential()
        if first:
            self.shortcut = nn.Sequential(
                # ????????????????1 ????????????????????
                # stride==2 the dim need to be increase
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride[1], bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )
        # if stride[1] != 1 or in_channels != out_channels:
        #     self.shortcut = nn.Sequential(
        #         # kernal = 1 need to des or inc dim
        #         # stride = 2 need to increase dim
        #         nn.Conv2d(in_channels, out_channels*4, kernel_size=1, stride=stride[1], bias=False),
        #         nn.BatchNorm2d(out_channels)
        #     )

    def forward(self, x):
        out = self.bottleneck(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# the newtork with batchnormal is not need to add bias
class ResNet50(nn.Module):
    def __init__(self, Bottleneck, num_classes=1000) -> None:
        super(ResNet50, self).__init__()
        self.in_channels = 64
        # the first is independent, so it is without res
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # conv2
        self.conv2 = self._make_layer(Bottleneck, 64, [[1, 1, 1]] * 3, [[0, 1, 0]] * 3)

        # conv3
        self.conv3 = self._make_layer(Bottleneck, 128, [[1, 2, 1]] + [[1, 1, 1]] * 3, [[0, 1, 0]] * 4)

        # conv4
        self.conv4 = self._make_layer(Bottleneck, 256, [[1, 2, 1]] + [[1, 1, 1]] * 5, [[0, 1, 0]] * 6)

        # conv5
        self.conv5 = self._make_layer(Bottleneck, 512, [[1, 2, 1]] + [[1, 1, 1]] * 2, [[0, 1, 0]] * 3)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def _make_layer(self, block, out_channels, strides, paddings):
        layers = []
        # to judeg if it is the first layer
        flag = True
        for i in range(0, len(strides)):
            layers.append(block(self.in_channels, out_channels, strides[i], paddings[i], first=flag))
            flag = False
            self.in_channels = out_channels * 4

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        x6 = self.avgpool(x5)
        x7 = x6.reshape(x.shape[0], -1)
        x8 = self.fc(x7)
        x8 = torch.unsqueeze(torch.unsqueeze(x8, 2), 3)
        return [x1,x2,x3,x4,x5,x6,x8]



