"""
Detail:
Ref:
Project:my_python_script
Time:2022/4/8 10:50
Author:WRZ
"""
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from slide import u2net,Log_weights, slide_dataset,T_Net
import pandas as pd

from slide.loss import losses, LearningRate, Ms_ssim
from slide.PackageDeepLearn.utils import OtherTools, DataIOTrans,Visualize
from tqdm import tqdm


# %%
def loss_function(label_gt,pre):


    L1_all = 0;#L2_all = 0;
    if type(pre) == tuple:
        for each in pre:
            loss1 = losses.TverskyLoss()(label_gt, each, alpha=0.5, beta=0.5)
            loss2 = torch.nn.L1Loss()(label_gt, each)
    #        L1 = torch.nn.CrossEntropyLoss()
    #        loss1 = L1(label_gt, torch.argmax(each,axis=1).long())
    #        loss2 = losses.MSE(label_gt, each)
            # L2 = 1 - M(label_gt, each)
            L1_all+=(loss1+loss2)
            # L2_all+=L2
    
        loss = L1_all/7 #+ L2_all/7
    else:
        loss = losses.TverskyLoss()(label_gt, pre)
#    loss.requires_grad = True
    # L1 = losses.MSE(label_gt, pre)
    # L2 = 1 - M(label_gt, pre)

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

        self.device = OtherTools.DEVICE_SLECT()

        print("============> Building model ...")
        # train_phase

        if train_phase == 'U2NET':
            self.model = u2net.U2NET()
        if train_phase == 'TNET':
            self.model = T_Net.T_mv2_unet(nInputChannels=3,classes=2)

        self.model.to(self.device)

        self.train_data = getattr(slide_dataset, 'SlideDatasets')(dir=self.dir,
                                                                    Numclass=2)

        # self.train_data = ConcatDataset([self.train_data1,self.train_data2,self.train_data3,self.train_data4])

        self.trainloader = DataLoader(self.train_data,
                                      batch_size=self.batch,
                                      drop_last=True,
                                      shuffle=True,
                                      num_workers=self.nThreads,
                                      pin_memory=True)


        # 记录
        self.trainlog = Log_weights.Train_Log(self.saveDir)

        # 微调模式？
        if finetuning:
            self.start_epoch, self.model = self.trainlog.load_model(self.model,
                                                                    lastest_out_path=finetuning)
            self.start_epoch = self.start_epoch + 1
            self.train_epoch = self.train_epoch + self.start_epoch + 1

        self.optimizer = optim.Adam(self.model.parameters(),lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        #self.optimizer = optim.SGD(self.model.parameters(),lr=self.lr)
    def train(self, epoch, tqdm_leaveLen):
        # 切换模型为训练模式
        self.model.train()
        if self.lrdecayType != 'keep':
            self.lr = LearningRate.set_lr(self.lrdecayType, self.lr, self.lrDecay, epoch, 220,
                                          self.optimizer)  # 这里将self.train_epoch 替换为350,便于poly计算,一定不要用你self.lr，否则会迭代学习率消失


        pbar = tqdm(self.trainloader, leave=bool(epoch == tqdm_leaveLen))
        loss_ = 0
        for i, sample_batched in enumerate(pbar):
            #                 print('第{}次循环'.format(i))
            self.train_img,self.train_label,self.train_dem = \
            sample_batched['traim_img'].to(self.device),sample_batched['train_label'].to(self.device),\
                sample_batched['train_dem'].to(self.device)


            
            # ----------------------------后期加入影像预测-----------------------------
            self.pre = self.model(torch.concat((self.train_img,self.train_dem),dim=1))

            # ------------------------------------------------------------------------
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
                    pre = torch.squeeze(torch.argmax(self.pre[0, :, :, :],axis=0)).detach().cpu()

                Visualize.visualize(savepath=self.saveDir + f'/train/epoch{epoch:03d}' + f'_iter{i:04d}',
                          original_img=self.train_img[0, :, :, :].cpu().permute(1, 2, 0),
                          train_label=DataIOTrans.DataTrans.OneHotDecode(
                              self.train_label[0, :, :, :].permute(1, 2, 0).cpu()),
                          pre = pre
                          )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 将损失拆成数字表达
            loss_ += loss.item()


        loss_ = loss_ / (i + 1)


        self.log = pd.DataFrame({"epoch": epoch,
                            "train_epoch": self.train_epoch,
                            'tain_loss': loss_,
                            "lr": self.lr,
                            }, index=[epoch])
        self.trainlog.save_log(self.log)

        return loss_



    def execute(self, lrDecay):

        inint_train_loss =10
        
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
batch = 2
nThreads = 0
train_dir =  r'D:/Wrz/batchlors_code/landslide_dataset'
lr = 0.001
saveDir = r'D:\Wrz\batchlors_code\slide\save_1dem'
train_epoch = 200
lrdecayType = 'poly'
lrDecay = 2
save_epoch = 20
train_phase = 'U2NET'  # 'U2NET'  'TNET'
start_epoch = 1

# %%
#
# Model = trainModel(train_phase, lr, train_dir, test_dir, saveDir,
#                    batch, nThreads, lrdecayType, start_epoch, train_epoch, save_epoch, finetuning=False)

Model = trainModel(train_phase, lr, train_dir, saveDir,
                   batch, nThreads, lrdecayType, start_epoch, train_epoch, save_epoch,
                   finetuning=False)

print('train_datalen={}'.format(Model.train_data.__len__()))
Model.execute(lrDecay=lrDecay)
