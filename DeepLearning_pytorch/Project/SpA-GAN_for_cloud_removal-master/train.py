import os
import random
import shutil
import yaml
from attrdict import AttrMap
import time

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from data_manager import S1S2_pinmemory,val_CloudRemovalDataset
from models.gen.SPANet import Generator
from models.dis.dis import Discriminator
import utils
from utils import gpu_manage, save_image, checkpoint
from log_report import LogReport
from log_report import TestReport
import signal,sys



def validate(val_loader, model):
    model.eval()
    losses_smoothl1 = nn.SmoothL1Loss(beta=0.2)
    loss_all = []
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda().float()
        target = target.cuda().float()
        with torch.no_grad():
            # compute output
            output = model(input)
            loss = losses_smoothl1(output[1], target)
            loss_all.append(loss)
    return torch.mean(torch.tensor(loss_all))

def train(config):
    gpu_manage(config)

    ### DATASET LOAD ###
    print('===> Loading datasets')

    # dataset = TrainDataset(config)
    train_dataset = S1S2_pinmemory(Parendir=r'E:\Datasets\2020-06-01_2020-09-30_processed',
                            ImageSize = (512, 512),
                            CropSize = (64, 256),
                            Resize = (128, 128),
                            DataTag = 'All',
                            SpecChannels = 12,
                            SarChannels = 1,
                            Batch_Sample = config.batchsize,
                            Batch_Cycle = 100)

    validation_dataset = val_CloudRemovalDataset(
                             Parendir=r'E:\Datasets\2020-06-01_2020-09-30_processed',
                             ImageSize = (512,512),
                             CropSize = (128,128),
                             Resize = (128,128),
                             DataTag = 'All',
                             SpecChannels=12,
                             SarChannels=1)

    print('train dataset:', len(train_dataset))
    print('validation dataset:', len(validation_dataset))
    training_data_loader = DataLoader(dataset=train_dataset, num_workers=config.threads, batch_size=config.batchsize, shuffle=False)
    validation_data_loader = DataLoader(dataset=validation_dataset, num_workers=config.threads, batch_size=config.validation_batchsize, shuffle=False)

    ### MODELS LOAD ###
    print('===> Loading models')

    gen = Generator(gpu_ids=config.gpu_ids, in_channels=14,out_channels=12)

    if config.gen_init is not None:
        param = torch.load(config.gen_init)
        gen.load_state_dict(param)
        print('load {} as pretrained model'.format(config.gen_init))

    dis = Discriminator(in_ch=26, out_ch=1, gpu_ids=config.gpu_ids,firstChannels=14,secondChannels=12)

    if config.dis_init is not None:
        param = torch.load(config.dis_init)
        dis.load_state_dict(param)
        print('load {} as pretrained model'.format(config.dis_init))

    # setup optimizer
    opt_gen = optim.Adam(gen.parameters(), lr=config.lr, betas=(config.beta1, 0.999), weight_decay=0.00001)
    opt_dis = optim.Adam(dis.parameters(), lr=config.lr, betas=(config.beta1, 0.999), weight_decay=0.00001)

    real_a = torch.FloatTensor(config.batchsize, config.in_ch, config.width, config.height)
    real_b = torch.FloatTensor(config.batchsize, config.out_ch, config.width, config.height)
    M = torch.FloatTensor(config.batchsize, config.width, config.height)

    criterionL1 = nn.L1Loss()
    criterionSmoothL1 = nn.SmoothL1Loss(beta=0.2)
    criterionMSE = nn.MSELoss()
    criterionSoftplus = nn.Softplus()

    if config.cuda:
        gen = gen.cuda()
        dis = dis.cuda()
        criterionL1 = criterionL1.cuda()
        criterionSmoothL1 = criterionSmoothL1.cuda()
        criterionMSE = criterionMSE.cuda()
        criterionSoftplus = criterionSoftplus.cuda()
        real_a = real_a.cuda()
        real_b = real_b.cuda()
        M = M.cuda()

    real_a = Variable(real_a)
    real_b = Variable(real_b)

    logreport = LogReport(log_dir=config.out_dir)
    validationreport = TestReport(log_dir=config.out_dir)

    print('===> begin')
    start_time=time.time()
    # main
    for epoch in range(1, config.epoch + 1):
        epoch_start_time = time.time()
        for iteration, batch in enumerate(training_data_loader, 1):
            gen.train()
            real_a_cpu, real_b_cpu, M_cpu = batch[0], batch[1], batch[2]
            real_a.data.resize_(real_a_cpu.size()).copy_(real_a_cpu)
            real_b.data.resize_(real_b_cpu.size()).copy_(real_b_cpu)
            M.data.resize_(M_cpu.size()).copy_(M_cpu)
            att, fake_b = gen.forward(real_a)

            ################
            ### Update D ###
            ################
            
            opt_dis.zero_grad()

            # train with fake
            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = dis.forward(fake_ab.detach())
            batchsize, _, w, h = pred_fake.size()

            loss_d_fake = torch.sum(criterionSoftplus(pred_fake)) / batchsize / w / h

            # train with real
            real_ab = torch.cat((real_a, real_b), 1)
            pred_real = dis.forward(real_ab)
            loss_d_real = torch.sum(criterionSoftplus(-pred_real)) / batchsize / w / h

            # Combined loss
            loss_d = loss_d_fake + loss_d_real

            loss_d.backward()

            if epoch % config.minimax == 0:
                opt_dis.step()

            ################
            ### Update G ###
            ################
            
            opt_gen.zero_grad()

            # First, G(A) should fake the discriminator
            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = dis.forward(fake_ab)
            loss_g_gan = torch.sum(criterionSoftplus(-pred_fake)) / batchsize / w / h

            # Second, G(A) = B
            loss_g_l1 = criterionL1(fake_b, real_b) * config.lamb
            loss_g_att = criterionMSE(att[:,0,:,:], M)
            loss_g = loss_g_gan + loss_g_l1 + loss_g_att

            loss_g.backward()

            opt_gen.step()

            # log
            if iteration % 10 == 0:
                print("===> Epoch[{}]({}/{}): loss_d_fake: {:.4f} loss_d_real: {:.4f} loss_g_gan: {:.4f} loss_g_l1: {:.4f}".format(
                epoch, iteration, len(training_data_loader), loss_d_fake.item(), loss_d_real.item(), loss_g_gan.item(), loss_g_l1.item()))
                
                log = {}
                log['epoch'] = epoch
                log['iteration'] = len(training_data_loader) * (epoch-1) + iteration
                log['gen/loss'] = loss_g.item()
                log['dis/loss'] = loss_d.item()

                logreport(log)



            # with torch.no_grad():
            #     log_validation = test(config, validation_data_loader, gen, criterionMSE, epoch)
            #     validationreport(log_validation)
            # print('validation finished')
            if iteration % 10000 == 0:
                checkpoint(config, epoch,iteration, gen, dis)
                logreport.save_lossgraph()
                validationreport.save_lossgraph()

                l_eval = validate(validation_data_loader, gen)
                print("===> Epoch[{}]({}/{}): loss_d_fake: {:.4f} loss_d_real: {:.4f} loss_g_gan: {:.4f} loss_g_l1: {:.4f} l_eval: {:.4f}".format(
                        epoch, iteration, len(training_data_loader), loss_d_fake.item(), loss_d_real.item(), loss_g_gan.item(),
                        loss_g_l1.item(),l_eval.item()))

                # ----------捕获程序中断信号,并保存模型
                def signal_handler(sig, frame):
                    print("程序接收到中断信号，正在保存模型参数...")
                    checkpoint(config, epoch, gen, dis)
                    sys.exit(0)
                signal.signal(signal.SIGINT, signal_handler)
        print('epoch', epoch, 'finished, use time', time.time() - epoch_start_time)
    print('training time:', time.time() - start_time)


if __name__ == '__main__':
    with open('config.yml', 'r', encoding='UTF-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = AttrMap(config)

    utils.make_manager()
    n_job = utils.job_increment()
    config.out_dir = os.path.join(config.out_dir, '{:06}'.format(n_job))
    os.makedirs(config.out_dir)
    print('Job number: {:04d}'.format(n_job))

    # 保存本次训练时的配置
    shutil.copyfile('config.yml', os.path.join(config.out_dir, 'config.yml'))

    train(config)



