import torch.nn as nn
import torch.nn.functional as F
import torch


class LeNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.ReLU(inplace=True))

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
