"""
Detail:
Ref:
Project:my_python_script
Time:2022/3/8 14:10
Author:WRZ
"""
from .Visualize import visualize
'''
这里不提前import函数，根据需求调用
仅放一些网络初始化以及辅助算法
'''
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)