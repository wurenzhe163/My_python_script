import torch
from torch import nn


class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, stride=1, bias=None, modulation=False):  #padding=1 被删除
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        目前只能作用于3*3卷积
        需要修改成包含空洞卷积的方案
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        # self.padding = padding
        self.padding = (self.kernel_size - 1) //2
        padding = self.padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)

        # 最终输出的卷积层，设置输入通道数和输出通道数
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        # 偏置层，学习之前公式(2)中说的偏移量
        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=padding, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        # register_backward_hook是为了方便查看这几层学出来的结果，对网络结构无影响。
        # 当训练一个网络，想要提取中间层的参数、或者特征图的时候，使用hook就能派上用场了
        # 这里获得的self._set_lr梯度输入和输出结果，为固定格式
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            # 权重学习层，这个是后来提出的第二个版本的卷积
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=padding, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            # 学习权重
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2  # N = ks*ks , 卷积中的点个数

        # 对于3*3卷积padding为1
        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

#-----------------------------------------------------------------------------------------------------------------------#
        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1
        # 我们学习出的量是float类型的，而像素坐标都是整数类型的，所以我们还要用双线性插值的方法去推算相应的值
        # 左上角坐标q_lt,左下角(q_lb)，右上角(q_rt)，右下角(q_rb)
        # 同时为了防止偏移量过大移出feature map的范围通过torch.clamp对卷积核坐标做了限制 (V2方案)
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)，计算每个点四邻域的权重（距离），用于双线性插值
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)，每个点四邻域的数值
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)，权重*坐标值求和
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        # 如果设置了modulation，那么就对每个位置乘上其对应的权重
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        '''
        生成了相对于原点(0,0)的相对坐标(x,y),3*3卷积生成9个坐标
        '''
        #
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)    # 2N ，每个点有x,y

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        '''
        根据输出的特征图，计算在输入特征图中卷积核中心点的坐标（坐标以xxx标识）
        h，w是输出特征图的高和宽
        '''
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(self.kernel_size//2, h*self.stride+1, self.stride),
            torch.arange(self.kernel_size//2, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        '''

        '''
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        '''
        根据坐标获取值
        '''
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset
