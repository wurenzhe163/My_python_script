"""
Detail:
Ref:
Project:my_python_script
Time:2022/4/8 10:24
Author:WRZ
"""
import torch
import os,cv2
import numpy as np
import torch.nn.functional as F
from PackageDeepLearn.utils import Visualize, DataIOTrans

search_files = lambda path,endwith: sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(endwith)])

class SlideDatasets(torch.utils.data.Dataset):
    '''
    该数据集建立方法服务于图像语义分割模型
    input
        :images_dir    输入图像路径
        :masks_dir     输入标签路径
        :Numclass=2    图像分类标签数
        :augmentation=None   图像扩充方式DataIOTrans.DataTrans.data_augmentation
    output
        :return         PrefetchDataset(Image, Mask)
    '''

    def __init__(
            self,
            dir,
            Numclass=None,  # 分类数
    ):
        self.train_dir = search_files(os.path.join(dir, "images"),'.tif')
        #self.dem_dir = search_files(os.path.join(dir, "dem_tif"),'.tif')
        self.labels_dir = search_files(os.path.join(dir,"labels"),".tif")
        self.Numclass = Numclass

    def __getitem__(self, i):
        # read images and masks
        self.train_img = DataIOTrans.DataIO.read_IMG(self.train_dir[i])[:,:,0:3]/255  #512*512*4
        # def revalue(img):
        #     img[img == 5] = 1
        #     img[img == 6] = 2
        #     img[img == 8] = 3
        #     img[img == 10] = 4
        #     return img

        #self.dem_img = DataIOTrans.DataIO.read_IMG(self.dem_dir[i])[:,:,0]
        self.train_label = DataIOTrans.DataIO.read_IMG(self.labels_dir[i])[:,:,0]#512*512
        self.train_label[self.train_label==255] = 0
        # self.train_label[self.train_label==255] = 1 #转化为二值图像，0为黑色，1为白色
        self.onehottrain_label = DataIOTrans.DataTrans.OneHotEncode(self.train_label , self.Numclass)#512*512*4


        # self.train_img = F.interpolate(self.train_img[None,:], size=(512,512), mode='nearest') #1*512*512*4
        self.train_img = torch.from_numpy(self.train_img[None, :])  # 1*512*512*4
        #self.dem_img = F.interpolate(self.dem_img[None,:], size=(512,512), mode='nearest')
        #self.onehottrain_label = F.interpolate(self.onehottrain_label[None,:], size=(512,512), mode='nearest')#1*512*512*4
        self.onehottrain_label = torch.from_numpy(self.onehottrain_label[None,:])  # 1*512*512*4

        # self.onehottrain_label[self.onehottrain_label>0.5]=1
        # self.onehottrain_label[self.onehottrain_label<=0.5]=0


        self.train_img2 = self.train_img[0,:,:,:]#torch.Size([512, 512, 4])
        #self.dem_img2 = self.dem_img[0,:,:,:]
        self.onehottrain_label2 = torch.argmax(self.onehottrain_label[0,:,:,:],axis=2) #torch.Size([512, 512])

        
        # 注意,经过augmentation,数据dtype=float64,需要转换数据类型才能够正常显示
        sample = {"train_img": self.train_img2.permute(2,0,1),

                  'train_label': self.onehottrain_label2,
                  #'train_dem':self.dem_img2
                  }
        for i, j in sample.items():
            sample[i] = j.type(torch.FloatTensor)
            
        return sample

    def visu(self):
        Visualize.visualize(
            original_image=self.train_img2,
            #dem_image = self.dem_img2.permute(1,2,0),
            onehotTrimap=self.onehottrain_label2
            # DataIOTrans.DataTrans.OneHotDecode(self.onehottrain_label2),
        )

    def __len__(self):
        # return length of

        return len(self.train_dir)


if __name__ == '__main__':
    dir = r'D:\Wrz\batchlors_code\dataset_segmentation1-4'
    Numclass = 4
    test = SlideDatasets(dir, Numclass=Numclass)
    print(test.__getitem__(80))
    test.visu()
    print(test.__len__())

