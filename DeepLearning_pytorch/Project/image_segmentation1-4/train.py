"""
Detail:
Ref:
Project:my_python_script
Time:2022/4/8 10:50
Author:WRZ
"""
import sys
#sys.path.append(r'C:\Users\11733\Desktop')
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

import u2net,Log_weights, slide_dataset,T_Net
from loss import losses, LearningRate, Ms_ssim
from PackageDeepLearn.utils import OtherTools, DataIOTrans,Visualize


# %%
image_name = lambda path, endwith: sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(endwith)])
path1 = image_name(r'D:\Wrz\batchlors_code\dataset_segmentation1-4\labels', '.tif')

def weigths_calculate(path = path1):

    image0 = 0
    image1 = 0
    image2 = 0
    image3 = 0
    for i in path:
        label = DataIOTrans.DataIO.read_IMG(i)[:, :, 0]  # 512*512
        image0 += np.sum([label == 0]) # 514026281
        image1 += np.sum([label == 1]) # 31193170
        image2 += np.sum([label == 2]) # 70396515
        image3 += np.sum([label == 3]) # 6976034
    ALL = image0 + image1 + image2 + image3
    # tensor([ 1.2112, 19.9592,  8.8441, 89.2473], dtype=torch.float64)
    return torch.tensor([1/(image0/ALL),1/(image1/ALL),1/(image2/ALL),1/(image3/ALL)])



def loss_function(label_gt,pre,
                  weights = torch.tensor([1.2112, 19.9592,  8.8441, 89.2473], dtype=torch.float32).to( OtherTools.DEVICE_SLECT())):

    train_loss = 0;#L2_all = 0;
    if type(pre) == tuple:
        for each in pre:
            train_loss += nn.CrossEntropyLoss(weight = weights, size_average=True)(each,label_gt.long())
        loss = train_loss/7 #+ L2_all/7
    else :
        loss = nn.CrossEntropyLoss(weight = weights, size_average=True)(pre, label_gt.long())
    return loss


# %%
class trainModel(object):
    def __init__(self, train_phase, lr,
                 train_dir, saveDir, batch,
                 nThreads, lrdecayType, start_epoch, train_epoch,
                 save_epoch, finetuning=False):
        print("============> Environment init")
        self.train_phase = train_phase
        self.lr = lr
        self.dir = train_dir
        self.saveDir = saveDir
        self.batch = batch
        self.nThreads = nThreads
        self.start_epoch = start_epoch
        self.train_epoch = train_epoch
        self.save_epoch = save_epoch
        self.lrdecayType = lrdecayType

        self.device = OtherTools.DEVICE_SLECT()# 选择cuda

        print("============> Building model ...")
        # train_phase

        #选择模型
        if train_phase == 'U2NET':
            self.model = u2net.U2NET(in_ch=3,out_ch=4)
        if train_phase == 'TNET':
            self.model = T_Net.T_mv2_unet(nInputChannels=3,classes=4)

        self.model.to(self.device)  #将模型加载到相应的设备中


        self.train_data = getattr(slide_dataset, 'SlideDatasets')(dir=self.dir,
                                                                    Numclass=4)
        # 上式等价于slide_dataset.SlideDatasets(dir=self.dir, Numclass=4)，赋予新的属性

        # self.train_data = ConcatDataset([self.train_data1,self.train_data2,self.train_data3,self.train_data4])

        self.trainloader = DataLoader(self.train_data,# 传入的数据
                                      batch_size=self.batch,
                                      drop_last=True,
                                      shuffle=True,
                                      num_workers=self.nThreads,
                                      pin_memory=True)


        # 记录
        self.trainlog = Log_weights.Train_Log(self.saveDir)  #创建保存训练进程的实例，

        # 微调模式？
        if finetuning:
            self.start_epoch, self.model, self.lr = self.trainlog.load_model(self.model,
                                                                    lastest_out_path=finetuning)
            self.start_epoch = self.start_epoch + 1
            self.train_epoch = self.train_epoch + self.start_epoch + 1

        self.optimizer = optim.Adam(self.model.parameters(),lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        #self.optimizer = optim.SGD(self.model.parameters(),lr=self.lr)

    def train(self, epoch, tqdm_leaveLen):
        # 切换模型为训练模式
        self.model.train() #
        if self.lrdecayType != 'keep':
            self.lr = LearningRate.set_lr(self.lrdecayType, self.lr, self.lrDecay, epoch, 220,
                                          self.optimizer)  #这里将self.train_epoch 替换为350,便于poly计算,一定不要用你self.lr，否则会迭代学习率消失


        pbar = tqdm(self.trainloader, leave=bool(epoch == tqdm_leaveLen))

        loss_ = 0
        for i, sample_batched in enumerate(self.trainloader):
            #                 print('第{}次循环'.format(i))
            self.train_img,self.train_label = \
            sample_batched['train_img'].to(self.device),sample_batched['train_label'].to(self.device)


            
            # ----------------------------后期加入影像预测-----------------------------
            self.pre = self.model(self.train_img)#输入原始影像进u2net
            
            Visualize.visualize(
                      original_img=self.train_img[0, :, :, :].cpu().permute(1, 2, 0),
                      train_label=self.train_label[0,:,:].cpu(),
                      pre = torch.squeeze(torch.argmax(self.pre[0][0, :, :, :],axis=0)).detach().cpu()
                      )
            # ------------------------------------------------------------------------
            # loss = loss_function(self.train_label,self.pre) #求损失函数
            loss = loss_function(self.train_label,self.pre)

            pbar.set_description(
                'Epoch : %d/%d, Iter : %d/%d,lr : %.6f  Loss: %.4f ' %
                (epoch, self.train_epoch, i + 1, len(self.train_data) // self.batch,
                 self.lr, loss.data.item()
                 )              )

            # 画图展示
            if (i + 1) % 100 == 0:
                if type(self.pre) == tuple:
                    pre =torch.squeeze(torch.argmax(self.pre[0][0, :, :, :],axis=0)).detach().cpu()
                else:
                    pre = torch.squeeze(torch.argmax(self.pre[0, :, :, :],axis=0)).detach().cpu() #降维

                Visualize.visualize(savepath=self.saveDir + f'/train/epoch{epoch:03d}' + f'_iter{i:04d}',
                          original_img=self.train_img[0, :, :, :].cpu().permute(1, 2, 0),
                          train_label=self.train_label[0,:,:].cpu(),
                          pre = pre
                          )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 将损失拆成数字表达
            loss_ += loss.item()


        loss_ = loss_ / (i + 1)  #平均损失函数


        self.log = pd.DataFrame({"epoch": epoch,
                            "train_epoch": self.train_epoch,
                            'tain_loss': loss_,
                            "lr": self.lr,
                            }, index=[epoch])
        self.trainlog.save_log(self.log)

        return loss_


    def execute(self, lrDecay):

        inint_train_loss = 10
        
        self.lrDecay = lrDecay
        self.tqdm_leaveLen = self.train_epoch
        for epoch in tqdm(range(self.start_epoch, self.train_epoch + 1)):

            train_loss = self.train(epoch, self.tqdm_leaveLen)
            if train_loss < inint_train_loss:
                inint_train_loss = train_loss
                self.trainlog.save_model(self.model, epoch, self.lr, train_loss=True, val_loss=False)
            # 固定
            if epoch % self.save_epoch == 0:
                self.trainlog.save_model(self.model, epoch, self.lr, train_loss=False, val_loss=False)

# %%
batch = 1
nThreads = 0
train_dir = r'D:\Wrz\batchlors_code\dataset_segmentation1-4'
lr = 0.001
saveDir = r'D:\Wrz\batchlors_code\dataset_segmentation1-4\save_dir'
train_epoch = 200
lrdecayType = 'poly'
lrDecay = 2
save_epoch = 20
train_phase = 'U2NET'  # 'U2NET'  'TNET'
start_epoch = 1
ckpt = r'D:\Wrz\batchlors_code\dataset_segmentation1-4\save_dir\model\0058model_obj.pth'

# %%
#
# Model = trainModel(train_phase, lr, train_dir, test_dir, saveDir,
#                    batch, nThreads, lrdecayType, start_epoch, train_epoch, save_epoch, finetuning=False)

Model = trainModel(train_phase, lr, train_dir, saveDir,
                   batch, nThreads, lrdecayType, start_epoch, train_epoch, save_epoch,
                   finetuning=ckpt)


print('train_datalen={}'.format(Model.train_data.__len__()))
Model.execute(lrDecay=lrDecay)
