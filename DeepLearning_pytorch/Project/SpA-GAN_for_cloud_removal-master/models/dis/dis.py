import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from collections import OrderedDict

from models.layers import CBR
from models.models_utils import weights_init, print_network


class _Discriminator(nn.Module):
    def __init__(self, in_ch, out_ch,firstChannels=16,secondChannels=12):
        super().__init__()
        self.in_ch = in_ch
        self.firstChannels = firstChannels
        self.secondChannels = secondChannels
        assert firstChannels + secondChannels == in_ch

        self.c0_0 = CBR(firstChannels, 32, bn=False, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c0_1 = CBR(secondChannels, 32, bn=False, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c1 = CBR(64, 128, bn=True, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c2 = CBR(128, 256, bn=True, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c3 = CBR(256, 512, bn=True, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c4 = nn.Conv2d(512, out_ch, 3, 1, 1)

    def forward(self, x):
        x_0 = x[:, :self.firstChannels]
        x_1 = x[:, self.firstChannels:]
        h = torch.cat((self.c0_0(x_0), self.c0_1(x_1)), 1)
        h = self.c1(h)
        h = self.c2(h)
        h = self.c3(h)
        h = self.c4(h)
        return h


class Discriminator(nn.Module):
    def __init__(self, in_ch, out_ch, gpu_ids,firstChannels=16,secondChannels=12):
        super().__init__()
        self.gpu_ids = gpu_ids

        self.dis = nn.Sequential(OrderedDict([('dis', _Discriminator(in_ch, out_ch,firstChannels=firstChannels,secondChannels=secondChannels))]))

        self.dis.apply(weights_init)

    def forward(self, x):
        if self.gpu_ids:
            return nn.parallel.data_parallel(self.dis, x, self.gpu_ids)
        else:
            return self.dis(x)