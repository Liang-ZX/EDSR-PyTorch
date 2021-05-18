# ACM MobileHCI-2021-SplitSR: An End-to-End Approach to Super-Resolution on Mobile Devices
# The code is written based on RCAN
#
from model import common
from attention.esa import ESA, ESAplus, Scale

import torch.nn as nn
import torch
import torch.nn.functional as F


def make_model(args, parent=False):
    return SplitSR(args)


class SplitSRBlock(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, alpha, bias=True, bn=False, act=nn.ReLU(True)):
        super(SplitSRBlock, self).__init__()
        self.alpha_channel = int(n_feat * alpha)
        self.conv = conv(self.alpha_channel, self.alpha_channel, kernel_size, bias=bias)
        # if bn: modules_body.append(nn.BatchNorm2d(n_feat))
        self.act = act
        # self.body = nn.Sequential(*modules_body)
        self.esa = ESA(n_feat, reduction=16)
        # self.std = StdLayer(self.alpha_channel, reduction=self.alpha_channel)
        # self.se = SELayer(n_feat, reduction=16)  # TODO for SplitSR_SE_beta

    def forward(self, x):
        active, passive = x[:, :self.alpha_channel], x[:, self.alpha_channel:]
        res = self.conv(active)
        res += active
        res = self.act(res)
        # res = self.std(res)
        out = torch.cat([passive, res], dim=1)
        out = self.esa(out)
        # out = self.std(out)
        # out = self.se(out)  # TODO for SplitSR_SE_beta
        return out


class LearnAlphaBlock(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, alpha, bias=True, bn=False, act=nn.ReLU(True)):
        super(LearnAlphaBlock, self).__init__()
        self.alpha = nn.Parameter(torch.FloatTensor([0.25]))
        self.n_feat = n_feat
        self.conv = conv
        self.conv = conv(self.alpha_channel, self.alpha_channel, kernel_size, bias=bias)
        # if bn: modules_body.append(nn.BatchNorm2d(n_feat))
        self.act = act
        # self.body = nn.Sequential(*modules_body)
        self.esa = ESA(n_feat, reduction=16)
        # self.std = StdLayer(self.alpha_channel, reduction=self.alpha_channel)
        # self.se = SELayer(n_feat, reduction=16)  # TODO for SplitSR_SE_beta

    def forward(self, x):
        alpha_channel = int(self.alpha * self.n_feat)
        active, passive = x[:, :alpha_channel], x[:, alpha_channel:]
        # F.conv2d(active, )
        res = self.conv(active)
        res += active
        res = self.act(res)
        # res = self.std(res)
        out = torch.cat([passive, res], dim=1)
        out = self.esa(out)
        # out = self.std(out)
        # out = self.se(out)  # TODO for SplitSR_SE_beta
        return out


class ResidualBlock(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True)):
        super(ResidualBlock, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        # modules_body.append(SELayer(n_feat, reduction=16))
        # modules_body.append(StdLayer(n_feat, reduction=4))
        modules_body.append(ESAplus(n_feat, reduction=16))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = res + x
        return res


# Residual Group (RG)
class StandardGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, act, n_resblocks):
        super(StandardGroup, self).__init__()
        modules_body = []
        modules_body = [
            ResidualBlock(
                conv, n_feat, kernel_size, bias=True, bn=False, act=act) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = res + x
        return res


class SplitGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, alpha, act, n_resblocks):
        super(SplitGroup, self).__init__()
        modules_body = [
            SplitSRBlock(
                conv, n_feat, kernel_size, alpha, bias=True, bn=False, act=act) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = res + x
        return res


class SplitSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(SplitSR, self).__init__()

        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        alpha_ratio = args.alpha_ratio
        # alpha_ratio = nn.Parameter(torch.FloatTensor([0.25]))
        hybrid_index = args.hybrid_index
        scale = args.scale[0]
        act = nn.ReLU(True)
        self.is_student = args.is_student

        # RGB mean for DIV2K
        self.sub_mean = common.MeanShift(args.rgb_range)

        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        self.G0 = SplitGroup(conv, n_feats, kernel_size, alpha_ratio, act=act, n_resblocks=n_resblocks)
        self.G1 = SplitGroup(conv, n_feats, kernel_size, alpha_ratio, act=act, n_resblocks=n_resblocks)
        self.G2 = SplitGroup(conv, n_feats, kernel_size, alpha_ratio, act=act, n_resblocks=n_resblocks)

        self.G3 = SplitGroup(conv, n_feats, kernel_size, alpha_ratio, act=act, n_resblocks=n_resblocks)
        # self.G0 = StandardGroup(conv, n_feats, kernel_size, act=act, n_resblocks=n_resblocks)
        # self.G1 = StandardGroup(conv, n_feats, kernel_size, act=act, n_resblocks=n_resblocks)
        # self.G2 = StandardGroup(conv, n_feats, kernel_size, act=act, n_resblocks=n_resblocks)
        # self.G3 = StandardGroup(conv, n_feats, kernel_size, act=act, n_resblocks=n_resblocks)
        self.G4 = StandardGroup(conv, n_feats, kernel_size, act=act, n_resblocks=n_resblocks)
        self.G5 = StandardGroup(conv, n_feats, kernel_size, act=act, n_resblocks=n_resblocks)
        # self.G6 = StandardGroup(conv, n_feats, kernel_size, act=act, n_resblocks=n_resblocks)

        self.r1 = conv(2 * n_feats, n_feats, kernel_size=1)
        self.r2 = conv(2 * n_feats, n_feats, kernel_size=1)
        self.r3 = conv(2 * n_feats, n_feats, kernel_size=1)
        self.r4 = conv(2 * n_feats, n_feats, kernel_size=1)
        self.r5 = conv(2 * n_feats, n_feats, kernel_size=1)
        # self.r6 = conv(2 * n_feats, n_feats, kernel_size=1)

        self.lrelu = nn.LeakyReLU(True)
        self.LR_conv = conv(n_feats, n_feats, kernel_size=3)

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        self.head = nn.Sequential(*modules_head)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        out_B0 = self.G0(x)
        out_B1 = self.G1(out_B0)
        out_B2 = self.G2(out_B1)
        out_B3 = self.G3(out_B2)
        out_B4 = self.G4(out_B3)
        out_B5 = self.G5(out_B4)
        # out_B6 = self.G6(out_B5)
        out_M1 = self.r1(torch.cat([out_B0, out_B1], dim=1))
        out_M2 = self.r2(torch.cat([out_M1, out_B2], dim=1))
        out_M3 = self.r3(torch.cat([out_M2, out_B3], dim=1))
        out_M4 = self.r4(torch.cat([out_M3, out_B4], dim=1))
        out_B = self.r5(torch.cat([out_M4, out_B5], dim=1))
        # out_B = self.r6(torch.cat([out_M5, out_B6], dim=1))
        # out_B = self.c(torch.cat([out_B0, out_B1, out_B2, out_B3, out_B4, out_B5, out_B6], dim=1))
        res = self.LR_conv(out_B) + x
        # out = res
        x = self.tail(res)
        x = self.add_mean(x)

        if self.training and self.is_student:
            return x, [out_B0, out_B1, out_B2, out_B3, out_B4, out_B5]
        else:
            return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
