import torch
import os,cv2
import numpy as np
import torch.nn.functional as F
from slide.PackageDeepLearn.utils import OtherTools,DataIOTrans,Visualize
from slide.PackageDeepLearn import ImageAfterTreatment
import slide.u2net as u2net
from tqdm import tqdm





class PreModel(object):
    def __init__(self,lastest_out_path,saveDir,pre_phase='Just_alpha'):

        self.saveDir = saveDir
        self.pre_phase = pre_phase
        self.device = OtherTools.DEVICE_SLECT()

        print("============> Building model ...")
        # build model
        if self.pre_phase == 'img':
            self.model = u2net.U2NET(in_ch=3,out_ch=2).to(self.device)
        if self.pre_phase == 'img_dem':
            self.model = u2net.U2NET(in_ch=4,out_ch=2).to(self.device)


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
            imgs = search_files(os.path.join(pre_img_dir, "image_tif"))
            dems = search_files(os.path.join(pre_img_dir, "dem_tif"))
            for i,(eachimg,eachdem) in enumerate(zip(imgs,dems)):
                I = DataIOTrans.DataIO.read_IMG(eachimg).astype(np.float32)

                img = F.interpolate(torch.from_numpy(I[None,:]).permute(0,3,1,2), size=(512,512), mode='nearest')/255
                img = img.to(self.device)

                
                D = DataIOTrans.DataIO.read_IMG(eachdem).astype(np.float32)[:,:,0:1]
 #               dem =  torch.from_numpy(D[np.newaxis,...]).to(self.device)
                dem = F.interpolate(torch.from_numpy(D[None,:]).permute(0,3,1,2), size=(512,512), mode='nearest')/255
                dem = dem.to(self.device)
                # #-----------------------------------------------this
                if self.pre_phase=='img':
                    self.alpha_pre = self.model(img)
                if self.pre_phase=='img_dem':
                    self.alpha_pre = self.model(torch.concat((img,dem),axis=1))
                    
#                alpha_pre_decode = torch.squeeze(self.alpha_pre[0]).detach().cpu().numpy()
                self.alpha_pre_decode =DataIOTrans.DataTrans.OneHotDecode\
                (self.alpha_pre[0][0, :, :, :].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8))

                
                Visualize.save_img(path=self.saveDir,
                index=i,norm=False,endwith='.tif',alpha=self.alpha_pre_decode[...,np.newaxis])


if __name__ == '__main__':
 


    # 相对路径
    pre_img_dir = r'D:\Wrz\batchlors_code\landslide_dataset'
    ckpt = r'D:\Wrz\batchlors_code\slide\save_1dem\model\ckpt_train.pth'
    output = r'D:\Wrz\batchlors_code\slide\save_1dem\pre'

    Model = PreModel(lastest_out_path=ckpt,
                     saveDir=output,pre_phase='img_dem')(pre_img_dir=pre_img_dir)


