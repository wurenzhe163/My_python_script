# 《Multi-Gradient Descent for Multi-Objective Reinforcement Learning》
#总的来说，MGDA_UB 算法是一种有效的多目标优化算法，它可以处理多个损失函数之间的权衡关系，并将它们结合起来进行优化。
# 替代pytorch原生的loss.backward(),使用时候还需要调整
import torch



# 这个实现似乎不太行，近考虑了一层
# def mgda_ub_update(losses, model, opt, lr, wd, device):
#     # 使用PyTorch自动微分功能计算每个任务损失相对于模型参数的梯度。
#     loss_grads = [torch.autograd.grad(loss, model.parameters(), retain_graph=True) for loss in losses]
#
#     # 计算每个任务梯度的L2范数
#     grad_norms = [g[0].norm(p=2) for g in loss_grads]
#     # 通过将其除以其L2范数来计算每个任务梯度的方向
#     grad_dirs = [g[0] / gn for g, gn in zip(loss_grads, grad_norms)]
#
#     # 初始化大小为num_losses x num_losses的矩阵P为0
#     num_losses = len(losses)
#     P = torch.zeros((num_losses, num_losses), device=device)
#     # 计算投影矩阵P。将P的对角元素设置为1，通过计算每对梯度方向的点积来得到其非对角线元素。
#     for i in range(num_losses):
#         for j in range(num_losses):
#             if i == j:
#                 P[i, j] = 1.0
#             else:
#                 P[i, j] = torch.dot(grad_dirs[i].view(-1), grad_dirs[j].view(-1))
#     # 计算每个任务梯度的缩放因子。通过取投影矩阵P的逆并将其乘以一个全1的向量来完成
#     lambdas = torch.linalg.inv(P) @ torch.ones((num_losses,), device=device)
#
#     # 将所有任务梯度连接成一个单独的向量
#     # grad_vec = torch.cat([g.flatten() for grads in loss_grads for g in grads])
#
#     grad_vec = torch.cat([torch.unsqueeze(g[0].flatten(),1) for g in loss_grads],dim=1)
#     # 计算每个任务梯度的权重。通过将每个缩放因子lambda乘以对应任务梯度的平方L2范数来完成
#     weights = [l * gn ** 2 for l, gn in zip(lambdas, grad_norms)]
#
#     # 将权重应用于连接的任务梯度向量
#     # grad_vec = torch.mul(grad_vec, torch.unsqueeze(torch.tensor(weights, device=device), 1))
#     grad_vec = torch.mm(grad_vec, torch.unsqueeze(torch.tensor(weights, device=device), 1))
#     # 将连接的任务梯度向量的形状调整为与模型参数相同的形状
#     grad = grad_vec.view_as(list(model.parameters())[0])
#     # 将权重衰减添加到梯度中
#     for param in model.parameters():
#         param.grad.add_(wd, param.data)
#     # grad.add_(wd, model.parameters())
#     # 使用梯度和学习率更新模型参数
#     opt.step(grad, lr)
#     # 清除模型参数的梯度以进行下一次迭代
#     opt.zero_grad()


def mgda_ub(losses, model, lr=1e-3):
    # 计算所有损失函数的梯度
    gradients = [torch.autograd.grad(loss, model.parameters(), retain_graph=True) for loss in losses]
    # 计算所有损失函数的梯度的范数
    gradient_norms = [torch.norm(torch.cat([grad.reshape(-1) for grad in grads]), p=2) for grads in gradients]
    # 计算所有损失函数梯度的范数的平均值
    gradient_norms_mean = sum(gradient_norms) / len(gradient_norms)
    # 计算所有损失函数梯度的范数的标准差
    gradient_norms_std = torch.std(torch.tensor(gradient_norms))
    # 计算所有损失函数梯度的范数的偏离程度
    gradient_norms_deviation = [(gradient_norms[i] - gradient_norms_mean) / gradient_norms_std for i in range(len(losses))]
    # 根据偏离程度计算权重
    weights = [torch.exp(-gradient_norms_deviation[i]) for i in range(len(losses))]
    # 归一化权重
    weights = [w / sum(weights) for w in weights]
    # 计算加权梯度
    weighted_gradients = []
    for i in range(len(losses)):
        weighted_gradients.append([w * grad for w, grad in zip(weights, gradients[i])])
    # 计算加权平均梯度
    averaged_gradients = [sum(grads) / len(grads) for grads in zip(*weighted_gradients)]
    # 更新模型参数
    for param, grad in zip(model.parameters(), averaged_gradients):
        param.data.sub_(lr * grad)
    return model


def mgda_ub_2loss(loss1,loss2,model,alpha=0.5):
    '''
	alpha定义了两个损失函数的重要性
    '''
	# 计算两个损失值对应的梯度
    grad1 = torch.autograd.grad(loss1, model.parameters(), create_graph=True)
    grad2 = torch.autograd.grad(loss2, model.parameters(), create_graph=True)

    # 使用 MGDA_UB 算法更新模型参数
    w = list(model.parameters())
    for j in range(len(w)):
        grad = alpha * grad1[j] + (1 - alpha) * grad2[j]
        norm = torch.sqrt(torch.sum(grad ** 2) + epsilon)
        w[j] = w[j] - 0.01 * grad / norm
    model.load_state_dict({'weight': w[0], 'bias': w[1]})
    return model
# optimizer.zero_grad()
#
# optimizer.step()