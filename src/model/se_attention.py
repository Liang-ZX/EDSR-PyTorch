import torch
import torch.nn as nn
import torch.nn.functional as F


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        # reduction = 2 # TODO for ShuffleNet_SE_beta
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


def mean_channels(x):
    assert(x.dim() == 4)
    spatial_sum = x.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (x.shape[2] * x.shape[3])


def std(x):
    assert(x.dim() == 4)
    x_mean = mean_channels(x)
    x_var = (x - x_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (x.shape[2] * x.shape[3])
    return x_var.pow(0.5)


class StdLayer(nn.Module):
    def __init__(self, num_fea, reduction=16):
        super(StdLayer, self).__init__()
        self.std = std
        self.branch_main = nn.Sequential(
            nn.Conv2d(num_fea, num_fea // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_fea // reduction, num_fea, 1, padding=0, bias=True),
            # nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.std(x)
        y = self.branch_main(y)
        return x * y
