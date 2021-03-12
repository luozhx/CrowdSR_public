import math

import torch.nn as nn


class MultiNetwork(nn.Module):
    def __init__(self, config, act=nn.ReLU(True)):
        super(MultiNetwork, self).__init__()

        self.scales = [1, 2, 3, 4]
        self.target_scale = None
        self.networks = nn.ModuleList()

        for scale in self.scales:
            self.networks.append(
                SingleNetwork(num_block=config[scale]['block'], num_feature=config[scale]['feature'], num_channel=3,
                              scale=scale, bias=True, act=act)
            )

    def set_target_scale(self, scale):
        assert scale in self.scales
        self.target_scale = scale

    def forward(self, x):
        assert self.target_scale in self.scales
        x = self.networks[self.target_scale - 1].forward(x)
        return x


class SingleNetwork(nn.Module):
    def __init__(self, num_block, num_feature, num_channel, scale, bias=True, act=nn.ReLU(True)):
        super(SingleNetwork, self).__init__()
        self.num_block = num_block
        self.num_feature = num_feature
        self.num_channel = num_channel
        self.scale = scale

        assert self.scale in [1, 2, 3, 4]

        head = []
        head.append(nn.Conv2d(in_channels=self.num_channel, out_channels=self.num_feature,
                              kernel_size=3, stride=1, padding=1, bias=bias))

        body = []
        for _ in range(self.num_block):
            body.append(ResBlock(self.num_feature, bias=bias, act=act))
        body.append(nn.Conv2d(in_channels=self.num_feature, out_channels=self.num_feature,
                              kernel_size=3, stride=1, padding=1, bias=bias))

        tail = []
        tail.append(nn.Conv2d(in_channels=self.num_feature, out_channels=self.num_channel,
                              kernel_size=3, stride=1, padding=1, bias=bias))

        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)

        if self.scale > 1:
            self.upscale = nn.Sequential(*UpSampler(self.scale, self.num_feature, bias=bias))

    def get_output_nodes(self):
        return self.output_node

    def forward(self, x):
        # feed-forward part
        x = self.head(x)
        res = self.body(x)
        res += x

        if self.scale > 1:
            x = self.upscale(res)
        else:
            x = res

        x = self.tail(x)

        return x


class ConvReLUBlock(nn.Module):
    def __init__(self, num_feature, bias):
        super(ConvReLUBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=num_feature, out_channels=num_feature, kernel_size=3, stride=1, padding=1,
                              bias=bias)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class ResBlock(nn.Module):
    def __init__(self, num_feature, bias=True, bn=False, act=nn.ReLU(True), res_scale=1, kernel_size=3):

        super(ResBlock, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(
                nn.Conv2d(in_channels=num_feature, out_channels=num_feature, kernel_size=kernel_size, stride=1,
                          padding=(kernel_size // 2), bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(num_feature))
            if i == 0:
                modules_body.append(act)

        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        if self.res_scale != 1:
            res = self.body(x).mul(self.res_scale)
        else:
            res = self.body(x)
        res += x

        return res


class UpSampler(nn.Sequential):
    def __init__(self, scale, nFeat, bn=False, act=None, bias=True):
        super(UpSampler, self).__init__()

        modules = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                modules.append(
                    nn.Conv2d(in_channels=nFeat, out_channels=4 * nFeat, kernel_size=3, stride=1, padding=1, bias=bias))
                modules.append(nn.PixelShuffle(2))
                if bn:
                    modules.append(nn.BatchNorm2d(nFeat))
                if act:
                    modules.append(act())
        elif scale == 3:
            modules.append(
                nn.Conv2d(in_channels=nFeat, out_channels=9 * nFeat, kernel_size=3, stride=1, padding=1, bias=bias))
            modules.append(nn.PixelShuffle(3))
            if bn:
                modules.append(nn.BatchNorm2d(nFeat))
            if act:
                modules.append(act())
        else:
            raise NotImplementedError

        self.upsampler = nn.Sequential(*modules)

    def forward(self, x):
        return self.upsampler(x)
