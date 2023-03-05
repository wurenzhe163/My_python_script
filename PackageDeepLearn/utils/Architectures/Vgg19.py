import torch
from torchvision import models
from torch import nn

class Vgg19(nn.Module):
    def __init__(self,requires_grad=False):
        super(Vgg19, self).__init__()
        self.vgg = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()

        for x in range(7):
            self.slice1.add_module(str(x),self.vgg[x])
        for x in range(7,21):
            self.slice2.add_module(str(x),self.vgg[x])
        for x in range(21,30):
            self.slice3.add_module(str(x),self.vgg[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self,x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)


        return [h_relu1,h_relu2,h_relu3]

if __name__ == '__main__':
    model = Vgg19()
    x = torch.randn(2,12,128,128)
    # x = nn.Conv2d(12, 3, 1, 1, 0)(x)
    b, c, h, w = x.shape
    x = x.reshape(b, c // 3, 3, h, w)
    x = x.sum(dim=1)
    y1,y2,y3 = model(x)
    print(y1.shape)
    print(y2.shape)
    print(y3.shape)