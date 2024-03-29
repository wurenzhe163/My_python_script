"""
    Trimap generation : T-Net

Author: Zhengwei Li
Date  : 2018/12/24
"""

import torch.nn as nn
#from torchsummary import summary

class InvertedResidual(nn.Module):
    '''
    残差结构，使用了分组卷积，要求输入和输出的通道大小相同，经过了2个3*3一个1*1卷积
    stride==1 & chanel_in == chanel_out的时候使用残差结构
    inp: input chanel
    oup: output chanel
    expand_ration: 中间层的深度调整
    '''
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]   # padding才不会出错

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:      # stride==1 & chanel_in == chanel_out
            return x + self.conv(x)
        else:
            return self.conv(x)

class mobilenet_v2(nn.Module):
    '''Encoder'''
    def __init__(self, nInputChannels=3):
        super(mobilenet_v2, self).__init__()
        # 1/2
        self.head_conv = nn.Sequential(nn.Conv2d(nInputChannels, 32, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU())
        # 1/2
        self.block_1 = InvertedResidual(32, 16, 1, 1)
        # 1/4 
        self.block_2 = nn.Sequential( 
            InvertedResidual(16, 24, 2, 6),
            InvertedResidual(24, 24, 1, 6)
            )
        # 1/8 
        self.block_3 = nn.Sequential( 
            InvertedResidual(24, 32, 2, 6),
            InvertedResidual(32, 32, 1, 6),
            InvertedResidual(32, 32, 1, 6)
            )
        # 1/16 
        self.block_4 = nn.Sequential( 
            InvertedResidual(32, 64, 2, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6)            
            )
        # 1/16
        self.block_5 = nn.Sequential( 
            InvertedResidual(64, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6)          
            )
        # 1/32 
        self.block_6 = nn.Sequential( 
            InvertedResidual(96, 160, 2, 6),
            InvertedResidual(160, 160, 1, 6),
            InvertedResidual(160, 160, 1, 6)          
            )
        # 1/32
        self.block_7 = InvertedResidual(160, 320, 1, 6)

    def forward(self, x):
        x = self.head_conv(x)
        # 1/2
        s1 = self.block_1(x)
        # 1/4 
        s2 = self.block_2(s1)
        # 1/8
        s3 = self.block_3(s2)
        # 1/16
        s4 = self.block_4(s3)
        s4 = self.block_5(s4)
        # 1/32
        s5 = self.block_6(s4)
        s5 = self.block_7(s5)

        return s1, s2, s3, s4, s5

class T_mv2_unet(nn.Module):
    '''
        mmobilenet v2 + unet
    '''

    def __init__(self, nInputChannels=1,classes=3):

        super(T_mv2_unet, self).__init__()
        # -----------------------------------------------------------------
        # encoder  
        # ---------------------
        self.feature = mobilenet_v2(nInputChannels=nInputChannels)

        # -----------------------------------------------------------------
        # decoder 
        # ---------------------

        self.s5_up_conv = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                        nn.Conv2d(320, 96, 3, 1, 1),
                                        nn.BatchNorm2d(96),
                                        nn.ReLU())
        self.s4_fusion = nn.Sequential(nn.Conv2d(96, 96, 3, 1, 1),
                                       nn.BatchNorm2d(96))

        self.s4_up_conv = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                        nn.Conv2d(96, 32, 3, 1, 1),
                                        nn.BatchNorm2d(32),
                                        nn.ReLU())
        self.s3_fusion = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                       nn.BatchNorm2d(32))

        self.s3_up_conv = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                        nn.Conv2d(32, 24, 3, 1, 1),
                                        nn.BatchNorm2d(24),
                                        nn.ReLU())
        self.s2_fusion = nn.Sequential(nn.Conv2d(24, 24, 3, 1, 1),
                                       nn.BatchNorm2d(24))

        self.s2_up_conv = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                        nn.Conv2d(24, 16, 3, 1, 1),
                                        nn.BatchNorm2d(16),
                                        nn.ReLU())
        self.s1_fusion = nn.Sequential(nn.Conv2d(16, 16, 3, 1, 1),
                                       nn.BatchNorm2d(16))

        self.last_conv = nn.Sequential(nn.Conv2d(16, classes, 3, 1, 1),
                                       nn.Softmax(dim=1)
                                       )
        #nn.Softmax(dim=1) , 使用较差熵误差一般不再使用分类函数，交叉熵误差函数会执行logSoftmax
        # self.last_up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, input):

        # -----------------------------------------------
        # encoder 
        # ---------------------
        s1, s2, s3, s4, s5 = self.feature(input)
        # -----------------------------------------------
        # decoder
        # ---------------------
        s4_ = self.s5_up_conv(s5)
        s4_ = s4_ + s4
        s4 = self.s4_fusion(s4_)

        s3_ = self.s4_up_conv(s4)
        s3_ = s3_ + s3
        s3 = self.s3_fusion(s3_)

        s2_ = self.s3_up_conv(s3)
        s2_ = s2_ + s2
        s2 = self.s2_fusion(s2_)

        s1_ = self.s2_up_conv(s2)
        s1_ = s1_ + s1
        s1 = self.s1_fusion(s1_)

        out = self.last_conv(s1)

        return out

#if __name__ == '__main__':
    # T_net = T_mv2_unet(nInputChannels=1,classes=3)
    # s = summary(T_net(),input_size=(3,512,512))

#    from torchstat import stat
 #   a = T_mv2_unet(nInputChannels=3,classes=2)
 #   stat(a, input_size=(1, 512, 512))