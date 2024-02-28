import torch.nn as nn
import torch.nn.functional as F
import torch

from PackageDeepLearn.utils.Architectures.Vgg19 import Vgg19

def MSE(gt,pre):

    return torch.sqrt(torch.pow(pre - gt, 2)).mean()
    #均方误差

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

    def forward(self, inputs, targets, smooth=1, alpha=0.2, beta=0.8,softmax=False):

        #comment out if your model contains a sigmoid or equivalent activation layer
        ###-----------------------------------------------------###
        ###  如果有了激活函数这里如果加入Softmax会导致loss函数指向错误   ###
        ###-----------------------------------------------------###
        if softmax:
            inputs = torch.softmax(inputs,dim=1)

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

def TverskyLoss_i(gt, pre, alpha=0.2, layerAttention=[2, 3], eps=1e-6, softmax=False, argmax=False):
    """

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
    # gt = torch.split(gt, 1, 1)
    # pre = torch.split(pre, 1, 1)
    if softmax:
        pre = torch.nn.Softmax(dim=1)(pre)
    if argmax:
        pre = torch.argmax(pre, )

    layerNotAttention = lambda shape, layer: [i for i in range(shape[1]) if i != layer]
    fp, fn = 0, 0;
    for i in layerAttention:
        fp += (gt[:, layerNotAttention(shape, 2), :, :].sum(axis=1) * pre[:, i, :, :]).sum()
        fn += (gt[:, i, :, :] * pre[:, layerNotAttention(shape, 2), :, :].sum(axis=1)).sum()
    fp = fp / len(layerAttention)
    fn = fn / len(layerAttention)

    tp = torch.Tensor([(gt[:, i, :, :] * pre[:, i, :, :]).sum() for i in layerAttention]).sum()

    # tp = (gt[layerAttention] * pre[layerAttention]).sum()
    # fp = ((gt[0] + gt[1]) * pre[layerAttention]).sum()
    # fn = (gt[layerAttention] * (pre[0] + pre[1])).sum()

    if tp > 0:
        return (1 - (tp + eps) / (tp + alpha * fp + (1 - alpha) * fn + eps)) / shape[0]
    if tp == 0:
        return (fp + fn) / (shape[0] * shape[2] * shape[3])

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

# RMAE(与label有关的比值损失)
class Loss_MRAE(nn.Module):
    def __init__(self):
        super(Loss_MRAE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = torch.abs(outputs - label) / torch.abs(label +1e-7)
        mrae = torch.mean(error.reshape(-1))
        return mrae

# SmoothL1
class SmoothL1Loss(nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean', beta=1.0):
        super(SmoothL1Loss, self).__init__()

        self.criterion_SmoothL1 = torch.nn.SmoothL1Loss(size_average=size_average,
                                                        reduce=reduce, reduction=reduction, beta=beta)

    def __call__(self, outputs, label):
        return self.criterion_SmoothL1(outputs, label)

# Gan_loss
class Adversarial_loss(nn.Module):
    def __init__(self,
                 gen:nn.Module,
                 dis:nn.Module,
                 total_iteration:int,
                 loss_tag='origin',
                 lr=0.0001,
                 betas=(0.5,0.999),
                 weight_decay=0.00001
                ):
        """
        loss_tag: origin  --> 采用二元交叉熵误差
                  WGAN-GP --> 推土机距离，以及逞罚全中
                  LSGAN   --> 最小二乘距离
        所用鉴别器结果不得激活
        """
        super().__init__()
        # 载入模型
        self.dis = dis.cuda()
        self.gen = gen
        self.loss_tag = loss_tag

        # 为discriminator额外定义optimizer
        self.opt_dis = torch.optim.Adam(self.dis.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        # 为discriminator额外定义scheduler
        self.scheduler_dis = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_dis, total_iteration, eta_min=1e-6)
        # SpA-GAN loss,采用Softplus进行激活，该函数类似Relu
        self.criterionSoftplus = nn.Softplus()

    # WGAN-GP 惩罚系数
    def gradient_penalty(self, real_b, fake_b):
        """
        计算梯度惩罚项
        """
        batch_size = real_b.size(0)
        # 生成随机权重
        alpha = torch.rand(batch_size, 1, 1, 1).to(real_b.device)
        # 计算随机线性插值
        interpolates = (alpha * real_b + (1 - alpha) * fake_b).requires_grad_(True)
        # 计算梯度
        d_interpolates = self.dis(interpolates)
        fake = torch.ones_like(d_interpolates).to(fake_b.device)
        gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                          grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True)[0]
        # 计算梯度范数
        gradients_norm = gradients.view(batch_size, -1).norm(2, dim=1)
        # 计算梯度惩罚项
        penalty = ((gradients_norm - 1) ** 2).mean()

        return penalty

    def D_loss(self,real_b,fake_b):
        '''
        real_b : target
        fake_b : output
        '''
        self.opt_dis.zero_grad()
        pred_fake = self.dis.forward(fake_b.detach())# 对生成数据detach，禁止生成器更新。（因为生成数据是生成器获得的）
        pred_real = self.dis.forward(real_b)

        if self.loss_tag == 'origin':
            # 常规GAN loss采用二元交叉熵误差
            self.criterion = nn.BCEWithLogitsLoss()
            loss_d_fake = self.criterion(pred_fake, torch.zeros_like(pred_fake))
            loss_d_real = self.criterion(pred_real, torch.ones_like(pred_real))
            loss_d = (loss_d_fake + loss_d_real) / 2
        elif self.loss_tag == 'WGAN-GP':
            gradient_penalty = self.gradient_penalty(real_b, fake_b)
            loss_d_fake = pred_fake.mean()
            loss_d_real = pred_real.mean()
            loss_d =  loss_d_fake - loss_d_real + 10 * gradient_penalty
        elif self.loss_tag == 'LSGAN':
            loss_d_fake = torch.mean(pred_fake ** 2)
            loss_d_real = torch.mean(pred_real - 1) ** 2
            loss_d = 0.5 * loss_d_real + loss_d_fake
        else :
            print('Please input right tag')
            # Discriminator直接反向传播优化，并更新optimizer和scheduler
        loss_d.backward(retain_graph=True)    # retain_graph=True ， 如果不detach生成结果，那么这里保留计算图内存
        self.opt_dis.step()
        self.scheduler_dis.step()

        return loss_d_fake,loss_d_real,loss_d

    def G_loss(self,fake_b):

        # 这里计算的G——loss需要与其余loss组合后反向传播
        pred_fake = self.dis.forward(fake_b)
        if self.loss_tag == 'origin':
            self.criterion = nn.BCEWithLogitsLoss()
            loss_g_gan = self.criterion(pred_fake, torch.ones_like(pred_fake))
        elif self.loss_tag == 'WGAN-GP':
            loss_g_gan =-pred_fake.mean()
        elif self.loss_tag == 'LSGAN':
            loss_g_gan = torch.abs(-torch.mean(pred_fake))

        return loss_g_gan

    def update_gen(self,out_gen):
        self.gen = out_gen


# Meaning Loss (Perceived distance)
# 注意：感知距离需要预训练网络结构
# Perceptual Loss
# 注意：感知损失需要预训练网络结构
class Perceptual_loss(nn.Module):

    def __init__(self, lossFunction):
        super().__init__()
        self.lossFunction = lossFunction
        self.Vgg19 = Vgg19().cuda()

    def forward(self, output, target):
        # Conv2d = nn.Conv2d(inputChannels, 3, 1, 1, 0,bias=False)
        # Conv2d.load_state_dict(test.state_dict()) # 载入固定权重，可以人为指定
        # o1,o2,o3 = Vgg19(Conv2d(output))
        # t1,t2,t3 = Vgg19(Conv2d(target))

        b, c, h, w = output.shape
        output = output.reshape(b, c // 3, 3, h, w)
        output = output.sum(dim=1)
        target = target.reshape(b, c // 3, 3, h, w)
        target = target.sum(dim=1)
        o1, o2, o3 = self.Vgg19(output)
        t1, t2, t3 = self.Vgg19(target)

        return self.lossFunction(o1, t1) + self.lossFunction(o2, t2) + self.lossFunction(o3, t3)


