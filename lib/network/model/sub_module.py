import torch
import torch.nn as nn

__all__ = ["ResBlock", "CrdBlock", "DsBlock", "CrdDsBlock"]


class ResBlock(nn.Module):
    def __init__(self, channels=None, drop_prob=0, dilation=1):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=1, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(channels, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels, eps=1e-3)
        )
        self.dropout = nn.Dropout2d(drop_prob)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.feat(x)
        if self.dropout.p != 0:
            y = self.dropout(y)
        return self.relu(y + x)


class CrdBlock(nn.Module):
    def __init__(self, channels=None, drop_prob=0, dilation=1):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 1), stride=1, padding=(dilation, 0), dilation=(dilation, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, (1, 3), stride=1, padding=(0, dilation), dilation=(1, dilation), bias=False),
            nn.BatchNorm2d(channels, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, (3, 1), stride=1, padding=(1, 0), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, (1, 3), stride=1, padding=(0, 1), bias=False),
            nn.BatchNorm2d(channels, eps=1e-3),
        )
        self.dropout = nn.Dropout2d(drop_prob)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.feat(x)
        if self.dropout.p != 0:
            y = self.dropout(y)
        return self.relu(y + x)


class DsBlock(nn.Module):
    def __init__(self, channels=None, drop_prob=0, dilation=1):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=1, padding=dilation, dilation=dilation, groups=channels, bias=False),
            nn.BatchNorm2d(channels, eps=1e-3),
            nn.Conv2d(channels, channels, 1, stride=1, bias=False),
            nn.BatchNorm2d(channels, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, stride=1, padding=dilation, dilation=dilation, groups=channels, bias=False),
            nn.BatchNorm2d(channels, eps=1e-3),
            nn.Conv2d(channels, channels, 1, stride=1, bias=False),
            nn.BatchNorm2d(channels, eps=1e-3),
        )
        self.dropout = nn.Dropout2d(drop_prob)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.feat(x)
        if self.dropout.p != 0:
            y = self.dropout(y)
        return self.relu(y + x)


class CrdDsBlock(nn.Module):
    def __init__(self, channels=None, drop_prob=0, dilation=1):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 1), stride=1,
                      padding=(dilation, 0), dilation=(dilation, 1), groups=channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, (1, 3), stride=1,
                      padding=(0, dilation), dilation=(1, dilation), groups=channels, bias=False),
            nn.BatchNorm2d(channels, eps=1e-3),
            nn.Conv2d(channels, channels, 1, stride=1, bias=False),
            nn.BatchNorm2d(channels, eps=1e-3),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, (3, 1), stride=1, padding=(1, 0), groups=channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, (1, 3), stride=1, padding=(0, 1), groups=channels, bias=False),
            nn.BatchNorm2d(channels, eps=1e-3),
            nn.Conv2d(channels, channels, 1, stride=1, bias=False),
            nn.BatchNorm2d(channels, eps=1e-3),
        )
        self.dropout = nn.Dropout2d(drop_prob)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.feat(x)
        if self.dropout.p != 0:
            y = self.dropout(y)
        return self.relu(y + x)
