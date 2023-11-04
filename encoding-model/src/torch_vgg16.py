import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG16(nn.Module):
    def __init__(self):
    # def __init__(self,fmask5, fmask6, fmask7):
        super(VGG16, self).__init__()
        # self.fmask5 = nn.Parameter(torch.as_tensor(fmask5), requires_grad=False)
        # self.fmask6 = nn.Parameter(torch.as_tensor(fmask6), requires_grad=False)
        # self.fmask7 = nn.Parameter(torch.as_tensor(fmask7), requires_grad=False)
        # 3 * 224 * 224
        self.conv1_1 = nn.Conv2d(3, 64, 3)  # 64 * 222 * 222
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=(1, 1))  # 64 * 222* 222
        self.maxpool1 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 64 * 112 * 112

        self.conv2_1 = nn.Conv2d(64, 128, 3)  # 128 * 110 * 110
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=(1, 1))  # 128 * 110 * 110
        self.maxpool2 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 128 * 56 * 56

        self.conv3_1 = nn.Conv2d(128, 256, 3)  # 256 * 54 * 54
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=(1, 1))  # 256 * 54 * 54
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=(1, 1))  # 256 * 54 * 54
        self.maxpool3 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 256 * 28 * 28

        self.conv4_1 = nn.Conv2d(256, 512, 3)  # 512 * 26 * 26
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 26 * 26
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 26 * 26
        self.maxpool4 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 512 * 14 * 14

        self.conv5_1 = nn.Conv2d(512, 512, 3)  # 512 * 12 * 12
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 12 * 12
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 12 * 12
        self.maxpool5 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 512 * 7 * 7

        # view

        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)
        # softmax 1 * 1 * 1000

    def forward(self, x):
        # x.size(0)¼´Îªbatch_size
        in_size = x.size(0)

        c1_1 = self.conv1_1(x)  # 222
        c1_1 = F.relu(c1_1)
        c1_2 = self.conv1_2(c1_1)  # 222
        c1_2 = F.relu(c1_2)
        c1_3 = self.maxpool1(c1_2)  # 112

        c2_1 = self.conv2_1(c1_3)  # 110
        c2_1 = F.relu(c2_1)
        c2_2 = self.conv2_2(c2_1)  # 110
        c2_2 = F.relu(c2_2)
        c2_3 = self.maxpool2(c2_2)  # 56

        c3_1 = self.conv3_1(c2_3)  # 54
        c3_1 = F.relu(c3_1)
        c3_2 = self.conv3_2(c3_1)  # 54
        c3_2 = F.relu(c3_2)
        c3_3 = self.conv3_3(c3_2)  # 54
        c3_3 = F.relu(c3_3)
        c3_4 = self.maxpool3(c3_3)  # 28

        c4_1 = self.conv4_1(c3_4)  # 26
        c4_1 = F.relu(c4_1)
        c4_2 = self.conv4_2(c4_1)  # 26
        c4_2 = F.relu(c4_2)
        c4_3 = self.conv4_3(c4_2)  # 26
        c4_3 = F.relu(c4_3)
        c4_4 = self.maxpool4(c4_3)  # 14

        c5_1 = self.conv5_1(c4_4)  # 12
        c5_1 = F.relu(c5_1)
        c5_2 = self.conv5_2(c5_1)  # 12
        c5_2 = F.relu(c5_2)
        c5_3 = self.conv5_3(c5_2)  # 12
        c5_3 = F.relu(c5_3)
        c5_4 = self.maxpool5(c5_3)  # 7

        # Õ¹Æ½
        out = c5_4.view(in_size, -1)

        f_1 = self.fc1(out)
        f_1 = F.relu(f_1)

        f_2 = self.fc2(f_1)
        f_2 = F.relu(f_2)

        f_3 = self.fc3(f_2)
        f_3 = F.log_softmax(f_3, dim=1)

        f_1 = torch.unsqueeze(torch.unsqueeze(f_1,2),3)
        f_2 = torch.unsqueeze(torch.unsqueeze(f_2, 2), 3)
        f_3 = torch.unsqueeze(torch.unsqueeze(f_3, 2), 3)
        # f_1 = torch.index_select(f_1, dim=1, index=self.fmask5)
        # f_2 = torch.index_select(f_2, dim=1, index=self.fmask6)
        # f_3 = torch.index_select(f_3, dim=1, index=self.fmask7)
        return [c1_3,c2_3,c3_4,c4_4,c5_4,f_1,f_2,f_3]

    # [c1_1, c1_2, c1_3, c2_1, c2_2, c2_3, c3_1, c3_2, c3_3, c3_4, c4_1, c4_2, c4_3, c4_4, c5_1, c5_2, c5_3, c5_4, f_1,
    #  f_2, f_3]
class vgg_fmaps(nn.Module):
    # def __init__(self, mean, std,fmask5, fmask6, fmask7):
    def __init__(self, mean, std):
        super(vgg_fmaps, self).__init__()
        self.mean = nn.Parameter(torch.as_tensor(mean), requires_grad=False)
        self.std = nn.Parameter(torch.as_tensor(std), requires_grad=False)
        # self.extractor = VGG16(fmask5, fmask6, fmask7)
        self.extractor = VGG16()

    def forward(self, _x):
        # _x = (_x - self.mean)/self.std
        # _x = (_x - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        # fmaps = self.extractor(_x)
        return self.extractor((_x - self.mean)/self.std)
        # return fmaps