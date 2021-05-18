# The code is written based on RCAN
#
from model import common
from attention.se_attention import SELayer

import torch.nn as nn
import torch


def make_model(args, parent=False):
    return ShuffleNet(args)


def conv_dw(in_channels, out_channels, kernel_size, bias=True):
    return [nn.Conv2d(in_channels, in_channels, kernel_size,
                      padding=(kernel_size // 2), groups=in_channels, bias=bias),
            nn.Conv2d(in_channels, out_channels, 1, bias=bias)]


class ShuffleV2Block(nn.Module):
    def __init__(self, n_feat, alpha=0.5, att_reduct=4, kernel_size=3, bias=True, bn=False, act=nn.ReLU6(True)):
        super(ShuffleV2Block, self).__init__()
        self.alpha = alpha
        self.alpha_channel = int(n_feat * alpha)

        mid_channels = n_feat // 2
        self.ksize = kernel_size
        self.pad = kernel_size // 2

        branch_main = []
        for i in range(2):
            # pw
            branch_main.append(nn.Conv2d(self.alpha_channel, mid_channels, 1, 1, 0, bias=bias))
            if bn: branch_main.append(nn.BatchNorm2d(mid_channels))
            branch_main.append(act)
            # dw
            branch_main.append(nn.Conv2d(mid_channels, mid_channels, self.ksize, 1, self.pad, groups=mid_channels, bias=bias))
            if bn: branch_main.append(nn.BatchNorm2d(mid_channels))
            branch_main.append(act)
            # pw-linear
            branch_main.append(nn.Conv2d(mid_channels, self.alpha_channel, 1, 1, 0, bias=bias))
            if bn: branch_main.append(nn.BatchNorm2d(self.alpha_channel))
            if i == 0: branch_main.append(act)
        branch_main.append(SELayer(self.alpha_channel, att_reduct))
        self.branch_main = nn.Sequential(*branch_main)

    def forward(self, old_x):
        if abs(self.alpha - 0.5) < 0.001:
            x_proj, x = self.channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        else: # alpha shuffle
            active, passive = old_x[:, :self.alpha_channel], old_x[:, self.alpha_channel:]
            res = self.branch_main(active)
            out = torch.cat([passive, res], dim=1)
            return out

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]


class ShuffleGroup(nn.Module):  # residual in residual
    def __init__(self, conv, n_feat, att_reduct, n_resblocks, kernel_size=3, attention=None, alpha=0.5,
                 act=nn.ReLU6(True)):
        super(ShuffleGroup, self).__init__()
        att_reduct = 4
        modules_body = [ShuffleV2Block(n_feat, att_reduct=att_reduct, alpha=alpha, kernel_size=kernel_size)
                        for _ in range(n_resblocks)]
        # modules_body += conv_dw(n_feat, n_feat, 3)
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = res + x
        return res


class ShuffleNet(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(ShuffleNet, self).__init__()
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        attention = args.att_type
        alpha = args.alpha_ratio
        att_reduct = args.att_reduct
        kernel_size = 3
        scale = args.scale[0]

        # RGB mean for DIV2K
        self.sub_mean = common.MeanShift(args.rgb_range)

        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        act = nn.ReLU6(True)
        # define body module
        modules_body = []
        for i in range(n_resgroups):
            modules_body.append(
                ShuffleGroup(conv, n_feats, att_reduct, n_resblocks=n_resblocks, kernel_size=kernel_size,
                             attention=attention, alpha=alpha, act=act))

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

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
