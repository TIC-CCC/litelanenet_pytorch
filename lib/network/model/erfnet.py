import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["erfnet"]


def erfnet():
    return ERFNet()


class ERFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(5)
        self.lane_exist = LaneExist(4)

    def forward(self, x):
        y = self.encoder(x)
        return {
            "seg_maps": self.decoder(y),
            "exist_codes": self.lane_exist(y)
        }


class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.Conv2d(ninput, noutput - ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()
        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)
        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)
        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True,
                                   dilation=(dilated, 1))
        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True,
                                   dilation=(1, dilated))
        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)
        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)
        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if self.dropout.p != 0:
            output = self.dropout(output)

        return F.relu(output + input)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_block = DownsamplerBlock(3, 16)
        self.layers = nn.ModuleList()
        self.layers.append(DownsamplerBlock(16, 64))
        for x in range(0, 5):  # 5 times
            self.layers.append(non_bottleneck_1d(64, 0.1, 1))
        self.layers.append(DownsamplerBlock(64, 128))
        for x in range(0, 2):  # 2 times
            self.layers.append(non_bottleneck_1d(128, 0.1, 2))
            self.layers.append(non_bottleneck_1d(128, 0.1, 4))
            self.layers.append(non_bottleneck_1d(128, 0.1, 8))
            self.layers.append(non_bottleneck_1d(128, 0.1, 16))

    def forward(self, inp):
        output = self.initial_block(inp)
        for layer in self.layers:
            output = layer(output)

        return output


class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3, track_running_stats=True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(UpsamplerBlock(128, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.output_conv = nn.ConvTranspose2d(16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)
        output = self.output_conv(output)

        return output


class LaneExist(nn.Module):
    def __init__(self, out_c):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(128, 32, (3, 3), stride=1, padding=(4, 4), bias=False, dilation=(4, 4)))
        self.layers.append(nn.BatchNorm2d(32, eps=1e-03))
        self.layers_final = nn.ModuleList()
        self.layers_final.append(nn.Dropout2d(0.1))
        self.layers_final.append(nn.Conv2d(32, 5, (1, 1), stride=1, padding=(0, 0), bias=True))
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.linear1 = nn.Linear(5 * 18 * 50, 128)
        self.linear2 = nn.Linear(128, out_c)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = F.relu(output)

        for layer in self.layers_final:
            output = layer(output)

        output = F.softmax(output, dim=1)
        output = self.maxpool(output)
        output = output.view(-1, 5 * 18 * 50)
        output = self.linear1(output)
        output = F.relu(output)
        output = self.linear2(output)
        output = F.sigmoid(output)

        return output


if __name__ == '__main__':
    import time
    from torchstat import stat

    torch.backends.cudnn.benchmark = True

    net = ERFNet()
    stat(net, (3, 288, 800))

    # net = ERFNet(5).cuda()
    # net.eval()
    #
    # x = torch.zeros((1, 3, 288, 800)).cuda() + 1
    # for i in range(10):
    #     y = net(x)
    # t_all = 0
    # with torch.no_grad():
    #     for i in range(200):
    #         t1 = time.time()
    #         y = net(x)
    #         t2 = time.time()
    #         t_all += t2 - t1
    #
    # print('avg_time:', t_all / 200)
    # print('avg_fps:', 200 / t_all)
