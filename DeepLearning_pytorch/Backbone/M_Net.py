"""
    Matting network : M-Net

Author: Zhengwei Li
Date  : 2018/12/24
"""

import torch
import torch.nn as nn


class M_net(nn.Module):
    '''
        encoder + decoder
    '''

    def __init__(self,mInputChannels=4):

        super(M_net, self).__init__()
        # -----------------------------------------------------------------
        # encoder  
        # ---------------------
        # 1/2
        self.en_conv_bn_relu_1 = nn.Sequential(nn.Conv2d(mInputChannels, 16, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(16),
                                       nn.ReLU())
        self.max_pooling_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 

        # 1/4
        self.en_conv_bn_relu_2 = nn.Sequential(nn.Conv2d(16, 32, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU())
        self.max_pooling_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  

        # 1/8
        self.en_conv_bn_relu_3 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU())
        self.max_pooling_3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 

        # 1/16
        self.en_conv_bn_relu_4 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU())
        self.max_pooling_4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  

        # self.en_conv_bn_relu_5 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1, bias=False),
        #                                nn.BatchNorm2d(128),
        #                                nn.ReLU())
        # -----------------------------------------------------------------
        # decoder  
        # ---------------------
        # 1/8
        self.de_conv_bn_relu_1 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU())
        self.deconv_1 = nn.ConvTranspose2d(256, 128, 5, 2, 2, 1, bias=False)

        # 1/4
        self.de_conv_bn_relu_2 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU())
        self.deconv_2 = nn.ConvTranspose2d(128, 64, 5, 2, 2, 1, bias=False)

        # 1/2
        self.de_conv_bn_relu_3 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU())
        self.deconv_3 = nn.ConvTranspose2d(64, 32, 5, 2, 2, 1, bias=False)

        # 1/1
        self.de_conv_bn_relu_4 = nn.Sequential(nn.Conv2d(32, 16, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(16),
                                       nn.ReLU())
        self.deconv_4 = nn.ConvTranspose2d(32, 16, 5, 2, 2, 1, bias=False)


        self.conv1 = nn.Conv2d(20, 1, 5, 1, 2, bias=False)
        self.conv2 = nn.Conv2d(20, 1, 5, 1, 2)


    def forward(self, input):

        # ----------------
        # encoder
        # --------
        x0 = self.en_conv_bn_relu_1(input)
        x0 = self.max_pooling_1(x0)

        x1 = self.en_conv_bn_relu_2(x0)
        x1 = self.max_pooling_2(x1)

        x2 = self.en_conv_bn_relu_3(x1)
        x2 = self.max_pooling_3(x2)

        x3 = self.en_conv_bn_relu_4(x2)
        x3 = self.max_pooling_4(x3)
        # ----------------
        # decoder
        # --------
        y0 = self.de_conv_bn_relu_1(x3)
        y0 = self.deconv_1(torch.cat((y0,x3),1))
        y1 = self.de_conv_bn_relu_2(y0)
        y1 = self.deconv_2(torch.cat((y1,x2),1))

        y2 = self.de_conv_bn_relu_3(y1)
        y2 = self.deconv_3(torch.cat((y2,x1),1))

        y3 = self.de_conv_bn_relu_4(y2)
        y3 = self.deconv_4(torch.cat((y3,x0),1))
        # y0 = self.de_conv_bn_relu_1(x3)
        # y0 = self.deconv_1(y0)
        # y1 = self.de_conv_bn_relu_2(y0)
        # y1 = self.deconv_2(y1)
        #
        # y2 = self.de_conv_bn_relu_3(y1)
        # y2 = self.deconv_3(y2)
        #
        # y3 = self.de_conv_bn_relu_4(y2)
        # y3 = self.deconv_4(y3)

        # raw alpha pred
        out1 = self.conv1(torch.cat((y3,input),1))
        
        
        y0_0 = self.de_conv_bn_relu_1(x3)
        y0_0 = self.deconv_1(torch.cat((y0_0,x3),1))
        y1_1 = self.de_conv_bn_relu_2(y0_0)
        y1_1 = self.deconv_2(torch.cat((y1_1,x2),1))

        y2_2 = self.de_conv_bn_relu_3(y1_1)
        y2_2 = self.deconv_3(torch.cat((y2_2,x1),1))

        y3_3 = self.de_conv_bn_relu_4(y2_2)
        y3_3 = self.deconv_4(torch.cat((y3_3,x0),1))
        out2 = self.conv2(torch.cat((y3_3,input),1))


        return out1,out2 

if __name__ == '__main__':
    from torchstat import stat
    a = M_net(mInputChannels=4)
    stat(a, input_size=(4, 512, 512))




