'''
参考代码：https://github.com/isl-org/MultiObjectiveOptimization/blob/master/multi_task/train_multi_task.py
参考解析：https://blog.csdn.net/Treasureashes/article/details/119003456
参考解析：blog.csdn.net/Marvin_Huoshan/article/details/120096374?spm=1001.2014.3001.5502
原论文：Multi-Task Learning as Multi-Objective Optimization。  https://arxiv.org/pdf/1810.04650.pdf
任务初衷：由于任务之间不是完全竞争或者不竞争的关系，而是一种相互博弈的关系，这时候单纯的线性解就没那么有用了，
所以需要去找到一个帕累托最优解来优化多个任务的表现，也就是，把多任务学习变成多目标优化问题。
优点：能够平衡多个损失函数，寻找多个损失函数优化的帕累托解。(这里将多任务，更换为单个任务，多个损失函数)
'''
import torch


grads = {}
# 这个函数是为了获取中间变量的梯度，我方案中的Z不是一个叶子结点，所以其梯度在反向传播之后不会被保存
def save_grad(name):
    def hook(grad):
        grads[name] = grad

    return hook


def MTL(loss, Floss):
    '''
    使用多任务学习的多个梯度来决定最终梯度
    :param loss: 损失1
    :param Floss: 损失2
    :return:
    '''
    Z.register_hook(save_grad('Z'))
    # 对loss进行反向传播，获取各层的梯度
    loss.backward(retain_graph=True)
    theta1 = grads['Z'].view(-1)

    # 将计算图中的梯度清零，准备对第二种loss进行反向传播
    optimizer.zero_grad(retain_graph=True)
    Floss.backward()
    theta2 = grads['Z'].view(-1)

    # alpha解的分子部分
    part1 = torch.mm((theta2 - theta1), theta2.T)
    # alpha解的分母部分
    part2 = torch.norm(theta1 - theta2, p=2)
    # 二范数的平方
    part2.pow(2)
    alpha = torch.div(part1, part2)
    min = torch.ones_like(alpha)
    alpha = torch.where(alpha > 1, min, alpha)
    min = torch.zeros_like(alpha)
    alpha = torch.where(alpha < 0, min, alpha)
    # alpha theta1 & (1 - alpha) theta2
    # 将alpha等维度拓展
    alpha1 = alpha
    alpha2 = (1 - alpha)
    # 将各层梯度清零
    optimizer.zero_grad()
    # 根据比率alpha1 & alpha2分配Loss1和Loss2的比率
    MTLoss = alpha1 * loss + alpha2 * Floss
    MTLoss.backward()
