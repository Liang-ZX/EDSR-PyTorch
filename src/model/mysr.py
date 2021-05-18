# ACM MobileHCI-2021-SplitSR: An End-to-End Approach to Super-Resolution on Mobile Devices
# The code is written based on RCAN
#
from model import common
from attention.esa import ESA, ESAplus, Scale
from attention.se_attention import SELayer

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
        # out = self.esa(out)
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
        modules_body_split = [
            SplitGroup(conv, n_feats, kernel_size, alpha_ratio, act=act, n_resblocks=n_resblocks)
            for _ in range(hybrid_index)
        ]
        modules_body_standard = [
            StandardGroup(conv, n_feats, kernel_size, act=act, n_resblocks=n_resblocks)
            for _ in range(n_resgroups - hybrid_index)
        ]

        modules_fusion = [conv(2 * n_feats, n_feats, kernel_size=1) for _ in range(n_resgroups-1)]
        # self.fusion = conv(n_resgroups*n_feats, n_feats, kernel_size=1)

        self.lrelu = nn.LeakyReLU(True)
        self.LR_conv = conv(n_feats, n_feats, kernel_size=3)

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        self.head = nn.Sequential(*modules_head)
        self.split_body = nn.Sequential(*modules_body_split)
        self.standard_body = nn.Sequential(*modules_body_standard)
        self.fusion = nn.Sequential(*modules_fusion)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        out_B = []
        last_B = x
        for i in range(len(self.split_body)):
            last_B = (self.split_body[i])(last_B)
            out_B.append(last_B)
        for i in range(len(self.standard_body)):
            last_B = (self.standard_body[i])(last_B)
            out_B.append(last_B)

        last_M = out_B[0]
        for i in range(len(self.fusion)):
            last_M = (self.fusion[i])(torch.cat([last_M, out_B[i+1]], dim=1))
        # last_M = self.fusion(torch.cat(out_B, dim=1))

        # out_B = self.c(torch.cat([out_B0, out_B1, out_B2, out_B3, out_B4, out_B5, out_B6], dim=1))
        res = self.LR_conv(last_M) + x
        # out = res
        x = self.tail(res)
        x = self.add_mean(x)

        if self.training and self.is_student:
            return x, out_B
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
