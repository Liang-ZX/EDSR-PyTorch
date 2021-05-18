from torch import nn
import torch


def make_model(args):
    return Bicubic(args)


class Bicubic(nn.Module):
    def __init__(self, args):
        super(Bicubic, self).__init__()
        scale = args.scale[0]
        self.scale = nn.Parameter(torch.FloatTensor([1.0]))
        self.upsample = nn.Upsample(scale_factor=scale, mode="bicubic", align_corners=False)

    def forward(self, x):
        return self.upsample(x)

