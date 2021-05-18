import torch
import torch.nn as nn
import torch.nn.functional as F
from model.rfdn import conv_block


def make_model(args, parent=False):
    return MAFFSRN(args)


class Scale(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, x):
        return x * self.scale


class Tail(nn.Module):
    def __init__(self, args, scale, n_feats, kernel_size, wn):
        super(Tail, self).__init__()
        out_feats = scale * scale * args.n_colors
        self.tail_k3 = wn(nn.Conv2d(n_feats, out_feats, 3, padding=3 // 2, dilation=1))
        self.tail_k5 = wn(nn.Conv2d(n_feats, out_feats, 5, padding=5 // 2, dilation=1))
        self.pixelshuffle = nn.PixelShuffle(scale)
        self.scale_k3 = Scale(0.5)
        self.scale_k5 = Scale(0.5)

    def forward(self, x):
        x0 = self.pixelshuffle(self.scale_k3(self.tail_k3(x)))
        x1 = self.pixelshuffle(self.scale_k5(self.tail_k5(x)))

        return x0 + x1


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


class FFG(nn.Module):
    def __init__(self, n_feats, wn, act=nn.ReLU(True)):
        super(FFG, self).__init__()

        self.b0 = MAB(n_feats=n_feats, reduction_factor=4)
        self.b1 = MAB(n_feats=n_feats, reduction_factor=4)
        self.b2 = MAB(n_feats=n_feats, reduction_factor=4)
        self.b3 = MAB(n_feats=n_feats, reduction_factor=4)

        self.reduction1 = wn(nn.Conv2d(n_feats * 2, n_feats, 1))
        self.reduction2 = wn(nn.Conv2d(n_feats * 2, n_feats, 1))
        self.reduction3 = wn(nn.Conv2d(n_feats * 2, n_feats, 1))
        self.res_scale = Scale(1)
        self.x_scale = Scale(1)

    def forward(self, x):
        x0 = self.b0(x)
        x1 = self.b1(x0) + x0
        x2 = self.b2(x1) + x1
        x3 = self.b3(x2)

        res1 = self.reduction1(channel_shuffle(torch.cat([x0, x1], dim=1), 2))
        res2 = self.reduction2(channel_shuffle(torch.cat([res1, x2], dim=1), 2))
        res = self.reduction3(channel_shuffle(torch.cat([res2, x3], dim=1), 2))

        return self.res_scale(res) + self.x_scale(x)


class MAB(nn.Module):
    def __init__(self, n_feats, reduction_factor=4, distillation_rate=0.25):
        super(MAB, self).__init__()
        self.reduce_channels = nn.Conv2d(n_feats, n_feats // reduction_factor, 1)
        self.reduce_spatial_size = nn.Conv2d(n_feats // reduction_factor, n_feats // reduction_factor, 3, stride=2,
                                             padding=1)
        self.pool = nn.MaxPool2d(7, stride=3)
        self.increase_channels = conv_block(n_feats // reduction_factor, n_feats, 1)

        self.conv1 = conv_block(n_feats // reduction_factor, n_feats // reduction_factor, 3, dilation=1,
                                act_type='lrelu')
        self.conv2 = conv_block(n_feats // reduction_factor, n_feats // reduction_factor, 3, dilation=2,
                                act_type='lrelu')

        self.sigmoid = nn.Sigmoid()

        self.conv00 = conv_block(n_feats, n_feats, 3, act_type=None)
        self.conv01 = conv_block(n_feats, n_feats, 3, act_type='lrelu')

        self.bottom11 = conv_block(n_feats, n_feats, 1, act_type=None)
        self.bottom11_dw = conv_block(n_feats, n_feats, 5, groups=n_feats, act_type=None)

    def forward(self, x):
        x = self.conv00(self.conv01(x))
        rc = self.reduce_channels(x)
        rs = self.reduce_spatial_size(rc)
        pool = self.pool(rs)
        conv = self.conv2(pool)
        conv = conv + self.conv1(pool)
        up = torch.nn.functional.interpolate(conv, size=(rc.shape[2], rc.shape[3]), mode='nearest')
        up = up + rc
        out = (self.sigmoid(self.increase_channels(up)) * x) * self.sigmoid(self.bottom11_dw(self.bottom11(x)))
        return out


class MAFFSRN(nn.Module):
    def __init__(self, args):
        super(MAFFSRN, self).__init__()
        # hyper-params
        self.scale = args.scale
        n_FFGs = 4
        n_feats = 32
        kernel_size = 3
        act = nn.LeakyReLU(True)

        wn = lambda x: torch.nn.utils.weight_norm(x)

        # define head module
        head = [wn(nn.Conv2d(3, n_feats, 3, padding=3 // 2))]

        # define body module
        body = []
        for i in range(n_FFGs):
            body.append(
                FFG(n_feats, wn=wn, act=act))

        # define tail module
        tail = Tail(args, self.scale[0], n_feats, kernel_size, wn)

        # make object members
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = tail

    def forward(self, x):
        input_x = x
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x + torch.nn.functional.interpolate(input_x, scale_factor=self.scale[0], mode='bicubic', align_corners=False)

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0 or name.find('skip') >= 0:
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
