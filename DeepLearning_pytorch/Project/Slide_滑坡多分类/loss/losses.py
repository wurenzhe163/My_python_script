import torch
import torch.nn

class TverskyLoss(torch.nn.Module):
    '''
    二分类，适用于Onehot数据
    '''
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.2, beta=0.8):

        #comment out if your model contains a sigmoid or equivalent activation layer
        ###-----------------------------------------------------###
        ###  如果有了激活函数这里如果加入Sigmoid会导致loss函数指向错误   ###
        ###-----------------------------------------------------###

        #inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputsN = inputs[:,0,:,:].reshape(-1)
        targetsN = targets[:,0,:,:].reshape(-1)
        inputsP = inputs[:, 1,:,:].reshape(-1)
        targetsP = targets[:,1,:,:].reshape(-1)

        #True Positives, False Positives & False Negatives
#         TP = (inputs * targets).sum()
#         FP = ((1-targets) * inputs).sum()
#         FN = (targets * (1-inputs)).sum()
        TP = (inputsP * targetsP ).sum()
        FP = (inputsP * targetsN ).sum()
        FN = (targetsP * inputsN ).sum()

        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)

        return 1 - Tversky
    
def TverskyLoss_i(gt,pre,alpha=0.2,layerAttention = 2,eps=1e-6):
    """
    目前只适合三分类
    Args:
        gt:  Ground - truth
        pre: Prediction
        alpha: prediction of Alpha-image
        layerAttention:  select one channel to attention
        eps:  in case the denominator is zero

    Returns:
        TverskyLoss of channel
    """

    shape = gt.shape
    gt = torch.split(gt, 1, 1)
    pre = torch.nn.Softmax(dim=1)(pre)        # 源网络中没有使用这个误差函数
    pre = torch.split(pre, 1, 1)

    tp = (gt[layerAttention] * pre[layerAttention]).sum()
    fp = ((gt[0] + gt[1]) * pre[layerAttention]).sum()
    fn = (gt[layerAttention] * (pre[0] + pre[1])).sum()
    # print('{},{},{}'.format(tp,fp,fn))
    if tp>0 :
        return (1 - (tp+eps)/(tp + alpha* fp+ (1-alpha)*fn + eps))/shape[0]
    if tp == 0:
#           print('fp={},shape={}'.format(fp, shape))
        return fp /(shape[0] * shape[2] * shape[3])
def CrossEntropyLoss():
    return torch.nn.CrossEntropyLoss()
def MSE(gt,pre,eps=0.0001):

    return torch.sqrt(torch.pow(pre - gt, 2) + eps).mean()
    #均方误差

# L_alpha = torch.sqrt(torch.pow(alpha_pre - alpha_gt, 2.).mean() + eps)


# L_t = CrossEntropyLoss()(trimap_pre, torch.argmax(trimap_gt, axis=1).long())
# TverskyLoss = TverskyLoss_i(trimap_gt, trimap_pre, alpha=0.2, layerAttention=2, flat=1e-6)