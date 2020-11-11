import torch
import torch.nn as nn

from lib.network.model.sub_module import *

__all__ = [
    "llnet", "llnet_prm",
    "llnet_crd", "llnet_ds", "llnet_crd_ds",
    "llnet_crd_prm", "llnet_ds_prm", "llnet_crd_ds_prm",
]


def llnet():
    return LiteLaneNet(block=ResBlock, use_pyramid=False)


def llnet_prm():
    return LiteLaneNet(block=ResBlock, use_pyramid=True)


def llnet_crd():
    return LiteLaneNet(block=CrdBlock, use_pyramid=False)


def llnet_ds():
    return LiteLaneNet(block=DsBlock, use_pyramid=False)


def llnet_crd_ds():
    return LiteLaneNet(block=CrdDsBlock, use_pyramid=False)


def llnet_crd_prm():
    return LiteLaneNet(block=CrdBlock, use_pyramid=True)


def llnet_ds_prm():
    return LiteLaneNet(block=DsBlock, use_pyramid=True)


def llnet_crd_ds_prm():
    return LiteLaneNet(block=CrdDsBlock, use_pyramid=True)


#####################
# LiteLaneNet
#####################
class LiteLaneNet(nn.Module):
    MAX_LANE_NUM = 4

    def __init__(self, block, use_pyramid):
        super().__init__()
        self.encoder = Encoder(alter_block=block, drop_prob=0.1, use_pyramid=use_pyramid)
        self.decoder = Decoder(self.MAX_LANE_NUM + 1)
        self.exist_branch = ExistBranch(self.MAX_LANE_NUM, drop_prob=0.2)

    def forward(self, inp):
        feat1, feat2 = self.encoder(inp)
        return {
            "seg_maps": self.decoder(feat1, feat2),
            "exist_codes": self.exist_branch(feat2)
        }


#####################
# encoder
#####################
class Encoder(nn.Module):
    def __init__(self, alter_block, drop_prob, use_pyramid):
        super().__init__()
        self.init_block = DownSample(3, 16)
        self.feat1 = nn.Sequential(
            DownSample(16, 64),
            alter_block(channels=64, drop_prob=drop_prob),
            alter_block(channels=64, drop_prob=drop_prob),
            alter_block(channels=64, drop_prob=drop_prob),
            alter_block(channels=64, drop_prob=drop_prob),
            alter_block(channels=64, drop_prob=drop_prob),
            alter_block(channels=64, drop_prob=drop_prob),
            alter_block(channels=64, drop_prob=drop_prob),
            alter_block(channels=64, drop_prob=drop_prob),
            alter_block(channels=64, drop_prob=drop_prob),
            alter_block(channels=64, drop_prob=drop_prob),
        )
        if not use_pyramid:
            self.feat2 = nn.Sequential(
                DownSample(64, 128),
                alter_block(channels=128, drop_prob=drop_prob),
                alter_block(channels=128, drop_prob=drop_prob),
                alter_block(channels=128, drop_prob=drop_prob),
                alter_block(channels=128, drop_prob=drop_prob),
                alter_block(channels=128, drop_prob=drop_prob),
                alter_block(channels=128, drop_prob=drop_prob),
                alter_block(channels=128, drop_prob=drop_prob),
                alter_block(channels=128, drop_prob=drop_prob),
            )
        else:
            self.feat2 = nn.Sequential(
                DownSample(64, 128),
                alter_block(channels=128, drop_prob=drop_prob, dilation=16),
                alter_block(channels=128, drop_prob=drop_prob, dilation=8),
                alter_block(channels=128, drop_prob=drop_prob, dilation=4),
                alter_block(channels=128, drop_prob=drop_prob, dilation=2),
                alter_block(channels=128, drop_prob=drop_prob, dilation=16),
                alter_block(channels=128, drop_prob=drop_prob, dilation=8),
                alter_block(channels=128, drop_prob=drop_prob, dilation=4),
                alter_block(channels=128, drop_prob=drop_prob, dilation=2),
            )

    def forward(self, x):
        y = self.init_block(x)
        y = self.feat1(y)
        feat1 = y
        y = self.feat2(y)
        feat2 = y

        return feat1, feat2


class DownSample(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(c_in, c_out - c_in, (3, 3), stride=2, padding=1),
            nn.BatchNorm2d(c_out - c_in, eps=1e-3),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        return torch.cat([self.feat(x), self.pool(x)], 1)


#####################
# decoder
#####################
class Decoder(nn.Module):
    def __init__(self, num_lanes):
        super().__init__()
        self.up1 = UpSample(128, 128)
        self.up2 = UpSample(128 + 64, 64)
        self.up3 = UpSample(64, 16)
        self.seg = nn.Conv2d(16, num_lanes, 1, stride=1)

    def forward(self, feat1, feat2):
        y = torch.cat([feat1, self.up1(feat2)], dim=1)
        y = self.up2(y)
        y = self.up3(y)
        return self.seg(y)


class UpSample(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c_out, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        return self.feat(x)


#####################
# classify branch
#####################
class ExistBranch(nn.Module):
    def __init__(self, c_out, drop_prob):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(128, 32, 3, stride=1, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(32, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(32, 4, 1, stride=1, bias=False),
            nn.BatchNorm2d(4, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout2d(drop_prob),
        )

        self.fc = nn.Sequential(
            nn.Linear(4 * 18 * 50, c_out),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.feat(x)
        y = y.view(-1, 4 * 18 * 50)
        return self.fc(y)


if __name__ == '__main__':
    # test_fps = 0
    test_fps = 0
    net = llnet_crd_ds_prm()

    if test_fps:
        import time
        torch.backends.cudnn.benchmark = True

        net.eval()
        net.cuda()

        x = torch.zeros((1, 3, 288, 800)).cuda() + 1
        for i in range(10):
            y = net(x)

        tt = 0
        with torch.no_grad():
            for i in range(200):
                t = time.time()
                y = net(x)
                tt += time.time() - t
        print('fps: {:.1f}'.format(200 / tt))

    else:
        from torchstat import stat
        stat(net, (3, 288, 800))
