import torch
import torch.nn as nn
from collections import OrderedDict

from PackageDeepLearn.utils.Architectures.layers import CBR
from PackageDeepLearn.utils import weights_init

# 一张图像生成一个logits：(20,28.128,128)-->(20,1)
class _Discriminator(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.in_ch = in_ch
        self.c0= CBR(self.in_ch, 64, bn=False, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c1 = CBR(64, 128, bn=True, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c2 = CBR(128, 256, bn=True, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c3 = CBR(256, 512, bn=True, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c4 = nn.Conv2d(512, 1, 3, 1, 1)
        self.out = nn.Sequential(
                # +一层MLP
                nn.Linear(64,1),
        )

    def forward(self, x):
        b,_,_,_ = x.shape
        h = self.c0(x)
        h = self.c1(h)
        h = self.c2(h)
        h = self.c3(h)
        h = self.c4(h)
        h = self.out(h.squeeze().reshape(b,-1))
        return h

class Discriminator_logits(nn.Module):
    def __init__(self, in_ch, out_ch, gpu_ids):
        super().__init__()
        self.gpu_ids = gpu_ids

        self.dis = nn.Sequential(OrderedDict([('dis', _Discriminator(in_ch, out_ch))]))

        self.dis.apply(weights_init)

    def forward(self, x):
        if self.gpu_ids:
            return nn.parallel.data_parallel(self.dis, x, self.gpu_ids)
        else:
            return self.dis(x)

# 一张图像生成多个logits：(20,28,128,128)-->(20,1,14,14)
class Discriminator_Patch(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator_Patch, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        x = nn.functional.leaky_relu(self.conv1(x), 0.2, True)
        x = nn.functional.leaky_relu(self.bn2(self.conv2(x)), 0.2, True)
        x = nn.functional.leaky_relu(self.bn3(self.conv3(x)), 0.2, True)
        x = nn.functional.leaky_relu(self.bn4(self.conv4(x)), 0.2, True)
        x = self.conv5(x)
        return x

# def discriminator_loss(real, fake, use_lsgan=True):
#     if use_lsgan:
#         # 使用 LSGAN 损失函数
#         loss = 0.5 * (torch.mean((real - 1) ** 2) + torch.mean(fake ** 2))
#     else:
#         # 使用普通的二元交叉熵损失函数
#         loss = F.binary_cross_entropy_with_logits(real, torch.ones_like(real)) + \
#                F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake))
#     return loss

if __name__ == '__main__':
    D_ = Discriminator(28, 1, [0]).cuda()
    test = torch.rand(20,28,128,128).cuda()
    D_(test)

    D_ = Discriminator_Patch(in_channels=28, use_sigmoid=False).cuda()
    D_(test)  # 20,1,14,14