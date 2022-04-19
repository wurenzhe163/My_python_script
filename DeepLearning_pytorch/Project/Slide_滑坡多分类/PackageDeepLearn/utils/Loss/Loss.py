import torch.nn as nn
import torch.nn.functional as F
import torch


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

def dice_loss(prediction, target):
    """Calculating the dice loss
    Args:
        prediction = predicted image
        target = Targeted image
    Output:
        dice_loss"""

    smooth = 1.0

    i_flat = prediction.view(-1)
    t_flat = target.view(-1)

    intersection = (i_flat * t_flat).sum()

    return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))


def calc_loss(prediction, target, bce_weight=0.5):
    """Calculating the loss and metrics
    Args:
        prediction = predicted image
        target = Targeted image
        metrics = Metrics printed
        bce_weight = 0.5 (default)
    Output:
        loss : dice loss of the epoch """
    bce = F.binary_cross_entropy_with_logits(prediction, target)
    prediction = F.sigmoid(prediction)
    dice = dice_loss(prediction, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    return loss


class TverskyLoss(nn.Module):
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

        inputs = torch.sigmoid(inputs)

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



# 超分损失函数https://www.cnblogs.com/jiangnanyanyuchen/p/11884912.html
class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss