import torch
import torch.nn as nn
from cloud_matting.T_Net import T_mv2_unet



T_net = T_mv2_unet


class net(nn.Module):
    '''
		end to end net
    '''

    def __init__(self,mnetModel='cascade',nInputChannels=1, classes=3, mInputChannels=4):

        super(net, self).__init__()

        self.t_net = T_net(nInputChannels=nInputChannels,classes=classes)
        if mnetModel=='cascade':
            from cloud_matting.M_net_cascade import M_net
        if mnetModel=='parallel':
            from cloud_matting.M_Net import M_net
        self.m_net = M_net(mInputChannels=mInputChannels)

    def forward(self, input):

        # trimap,并使用softmax函数激活,替换成了argmax激活，tnet已经固定
        trimap = self.t_net(input)
        #         trimap_softmax = F.softmax(trimap, dim=1)
        trimap_softmax = torch.argmax(trimap, axis=1)

        #  超参数 应急
        one_hot_codes = torch.eye(3)
        trimap_softmax = one_hot_codes[trimap_softmax].permute(0, 3, 1, 2).to(trimap.device)

        # 分离背景、前景、不确定区域: bs, fs, us
        bg, unsure, fg = torch.split(trimap_softmax, split_size_or_sections=1,
                                     dim=1)  # -----------------这里原来是bg,fg,unsure

        # concat input and trimap
        m_net_input = torch.cat((input, trimap_softmax), 1)

        # matting
        alpha_r, img_pre = self.m_net(m_net_input)
        # fusion module
        # paper : alpha_p = fs + us * alpha_r
        alpha_p = fg + unsure * alpha_r
        # alpha_p = alpha_r
        # alpha_p 布尔精华
        #         alpha_p = (alpha_p * 100).round() / 100

        # cloud removal 精炼
        #         img_pre=torch.div((1-alpha_p),1,rounding_mode='floor')*input + img_pre
        img_pre = (1 - alpha_p) * input + img_pre
        #         img_pre = (((1-alpha_p) // 1) * input) + img_pre
        return trimap, alpha_p, img_pre

if __name__ == '__main__':
    pass




