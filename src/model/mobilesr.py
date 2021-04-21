# The code is written based on RCAN
#
from model import common
from model.se_attention import SELayer
from model.coordatt import CoordAtt, h_swish

import torch.nn as nn


def make_model(args, parent=False):
    return MobileSR(args)


def conv_dw(in_channels, out_channels, kernel_size, bias=True):
    return [nn.Conv2d(in_channels, in_channels, kernel_size,
                      padding=(kernel_size // 2), groups=in_channels, bias=bias),
            nn.Conv2d(in_channels, out_channels, 1, bias=bias)]


class MobileV3Block(nn.Module):  # MobileNetV3
    def __init__(self, inp, oup, expand_ratio=6, att_reduct=4, bias=True, bn=False, attention=None,
                 act=nn.ReLU6(inplace=True)):
        super(MobileV3Block, self).__init__()
        stride = 1
        hidden_dim = int(inp * expand_ratio)
        if bn: bias = False
        se = attention == "SE"
        ca = attention == "CA"
        self.use_res_connect = inp == oup

        modules_body = []
        if expand_ratio > 1:
            # pw
            modules_body.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(hidden_dim))
            modules_body.append(act)
        # dw
        modules_body.append(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=bias))
        if bn: modules_body.append(nn.BatchNorm2d(hidden_dim))
        if se: modules_body.append(SELayer(hidden_dim, att_reduct))
        if ca: modules_body.append(CoordAtt(hidden_dim, hidden_dim, min(att_reduct, expand_ratio)))
        modules_body.append(act)
        # pw-linear
        modules_body.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=bias))
        if bn: modules_body.append(nn.BatchNorm2d(oup))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.body(x)
        else:
            return self.body(x)


# Residual Group
class MobileGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, expand_ratio, n_resblocks, att_reduct=4, attention=None,
                 act=nn.ReLU(True)):
        super(MobileGroup, self).__init__()
        modules_body = [MobileV3Block(n_feat, n_feat, expand_ratio, att_reduct, attention=attention, act=act)
                        for _ in range(n_resblocks)]
        # modules_body += conv_dw(n_feat, n_feat, 3)
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = res + x
        return res


class MobileSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(MobileSR, self).__init__()
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        attention = args.att_type
        expand_ratio = args.expand_ratio
        att_reduct = args.att_reduct
        kernel_size = 3
        scale = args.scale[0]

        # RGB mean for DIV2K
        self.sub_mean = common.MeanShift(args.rgb_range)

        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = []
        for i in range(n_resgroups):
            if i < n_resgroups // 2:
                act = nn.ReLU6(True)
            else:
                act = h_swish(True)
            if i < 3:
                expand_ratio = 3
            modules_body.append(
                MobileGroup(conv, n_feats, kernel_size, expand_ratio, n_resblocks=n_resblocks, attention=attention,
                            att_reduct=att_reduct, act=act))

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
