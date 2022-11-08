import sys
import torch
import os
import numpy as np
import torch.nn.functional as F
from PackageDeepLearn.utils import OtherTools,DataIOTrans,Visualize
from PackageDeepLearn import ImageAfterTreatment
import u2net
from tqdm import tqdm



class PreModel(object):
    def __init__(self,lastest_out_path,saveDir,pre_phase='Just_alpha'):

        self.saveDir = saveDir
        self.pre_phase = pre_phase
        self.device = OtherTools.DEVICE_SLECT()
        DataIOTrans.make_dir(self.saveDir)

        print("============> Building model ...")
        # build model
        if self.pre_phase == 'img':
            self.model = u2net.U2NET(in_ch=3,out_ch=4).to(self.device)#,dtype=torch.float64

        # lode_model
        if self.device.type == 'cpu':
            ckpt = torch.load(lastest_out_path,map_location=lambda storage, loc: storage)
        else:
            ckpt = torch.load(lastest_out_path)

        self.epoch = ckpt['epoch']
        self.lr = ckpt['lr']
        self.model.load_state_dict(ckpt['state_dict'])

        print("=> loaded checkpoint '{}' (epoch {})".format(lastest_out_path, ckpt['epoch']))

    def __call__(self,pre_img_dir=False,pre_img=False,kernel = [],stride = []):
        """

        Args:
            pre_img_dir:
            pre_img: 是否读入整景影像进行处理
            kernel:
            stride:

        Returns:

        """
        self.kernel = kernel
        self.stride = stride
        self.model.eval()
        search_files = lambda path : sorted([os.path.join(path,f) for f in os.listdir(path) if f.endswith(".tif")])
        if pre_img_dir:
            imgs = search_files(pre_img_dir)
            for i,eachimg in enumerate(imgs):
                # 转换图像结构
                I = DataIOTrans.DataIO.read_IMG(eachimg)[:,:,0:3]/255
                img_ = torch.from_numpy(I[np.newaxis,...]).to(self.device)
                img_ = img_.permute(0,3,1,2).float()


                # #-----------------------------------------------this
                if self.pre_phase=='img':
                    self.alpha_pre = self.model(img_)

                # 解码显示
                self.alpha_pre_decode =DataIOTrans.DataTrans.OneHotDecode\
                (self.alpha_pre[0][0, :, :, :].permute(1, 2, 0).detach().cpu().numpy()).astype(np.uint8)
                
                Visualize.visualize(img = I,out=self.alpha_pre_decode)
                
                Visualize.save_img(path=self.saveDir,
                index=i,norm=False,endwith='.tif',alpha=self.alpha_pre_decode[...,np.newaxis])
                
        if pre_img_dir==False and pre_img!=False:
            Img_Post = ImageAfterTreatment.Img_Post()
            data = Img_Post.read_IMG(pre_img).astype(np.float32)
            Shape = data.shape
            data = Img_Post.expand_image(data, self.stride, self.kernel)
            data_list, H, W = Img_Post.cut_image(data, self.kernel, self.stride) 

            outList = []
            for i,img in enumerate(tqdm(data_list, ncols=80)):
                img[img==255] = 0
                img_ = img[None,:,:,0:3]/255
                
                img_ = torch.from_numpy(img_).to(self.device)
                img_ = img_.permute(0,3,1,2)
                if self.pre_phase == 'img':
                    self.output = self.model(img_)
                # alpha_pre_decode = torch.squeeze(self.alpha_pre[0][0]).detach().cpu().numpy()
                self.output_decode = DataIOTrans.DataTrans.OneHotDecode \
                    (self.output[0][0, :, :, :].permute(1, 2, 0).detach().cpu().numpy()).astype(np.uint8)

                outList.append(self.output_decode[:,:,None])
                if i%100==0:
                    Visualize.visualize(savepath=self.saveDir+'\\'+f'{i:04d}'+'.jpg',Input=img[:,:,0:3].astype(np.uint8),Output=self.output_decode)

            outPut =Img_Post.join_image2(img = outList, kernel=self.kernel, stride=self.stride, H=H, W=W, S=1)
            Visualize.save_img(path=self.saveDir,index=i,norm=False,endwith='.tif',Output=outPut[0:Shape[0],0:Shape[1],:])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='1-4目标预测输出')
    parser.add_argument('--ckpt', type=str, metavar='', default=r'E:\Raiway_Demo\dataset_segmentation1-4\save_dir\model\0058model_obj.pth', help='预训练权重')
    parser.add_argument('--pre_img_dir', type=str, metavar='',default=False, help='图像切片文件夹')
    parser.add_argument('--pre_img', type=str, metavar='', default=r'E:\Raiway_Demo\dataset_segmentation1-4\image\0001.tif', help='整张图像路径(任意大小)')
    parser.add_argument('--output', type=str, metavar='',default=r'E:\Raiway_Demo\dataset_segmentation1-4\save_dir\pre', help='图像输出路径')
    parser.add_argument('--kernel', type=int, default=[512,512],nargs='+', metavar='', help='裁剪图像大小')
    parser.add_argument('--stride', type=int,default=256, metavar='', help='裁剪步长')
    parser.add_argument("--port", default=52162)
    parser.add_argument("--mode", default='client')
    args = parser.parse_args()

    # 相对路径
    pre_img_dir = args.pre_img_dir
    pre_img = args.pre_img
    ckpt = args.ckpt
    kernel = args.kernel
    stride = args.stride
    output = args.output

    Model = PreModel(lastest_out_path=ckpt,
                     saveDir=output,pre_phase='img')(pre_img_dir=pre_img_dir,pre_img=pre_img,kernel=kernel,stride=stride)


