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
from slide.PackageDeepLearn.utils import Visualize, DataIOTrans

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
        self.train_dir = search_files(os.path.join(dir, "image_tif"),'.tif')
        self.dem_dir = search_files(os.path.join(dir, "dem_tif"),'.tif')
        self.labels_dir = search_files(os.path.join(dir,"mask_tif"),".tif")
        self.Numclass = Numclass

    def __getitem__(self, i):
        # read images and masks
        self.train_img = DataIOTrans.DataIO.read_IMG(self.train_dir[i])/255
        self.dem_img = DataIOTrans.DataIO.read_IMG(self.dem_dir[i])[:,:,0]
        self.train_label = DataIOTrans.DataIO.read_IMG(self.labels_dir[i])[:,:,0]
        self.train_label[self.train_label==255] = 1
        self.onehottrain_label = DataIOTrans.DataTrans.OneHotEncode(self.train_label , self.Numclass)


        # 使用合成图像最大值对原始图像以及合成图像归一化
#            ImageMask = np.concatenate([self.train_img, self.onehottrain_label], axis=2)  # 图像与Lable一同变换
#           # ImageMask = cv2.resize(ImageMask,(512,512),interpolation=cv2.INTER_AREA)
        p = np.random.choice([0,1])
        self.augmentation0 = DataIOTrans.DataTrans.data_augmentation(ToTensor=True,Contrast=p)
        self.augmentation1 = DataIOTrans.DataTrans.data_augmentation(ToTensor=True)
        self.train_img = self.augmentation0(self.train_img)
        self.dem_img = self.augmentation1(self.dem_img)
        self.onehottrain_label = torch.from_numpy(self.onehottrain_label).permute(2,0,1)
#        self.onehottrain_label = self.augmentation1(self.onehottrain_label)

        self.train_img = F.interpolate(self.train_img[None,:], size=(512,512), mode='nearest')
        self.dem_img = F.interpolate(self.dem_img[None,:], size=(512,512), mode='nearest')
        
        self.onehottrain_label = F.interpolate(self.onehottrain_label[None,:], size=(512,512), mode='nearest')
        self.onehottrain_label[self.onehottrain_label>0.5]=1
        self.onehottrain_label[self.onehottrain_label<=0.5]=0
        
        self.tain_img2 = self.train_img[0,:,:,:]
        self.dem_img2 = self.dem_img[0,:,:,:]
        self.onehottrain_label2 = self.onehottrain_label[0,:,:,:]
        
        # 注意,经过augmentation,数据dtype=float64,需要转换数据类型才能够正常显示
        sample = {"traim_img": self.tain_img2,
                  'train_label': self.onehottrain_label2,
                  'train_dem':self.dem_img2
                  }
        for i, j in sample.items():
            sample[i] = j.type(torch.FloatTensor)
            
        return sample

    def visu(self):
        Visualize.visualize(
            original_image=self.tain_img2.permute(1,2,0),
            dem_image = self.dem_img2.permute(1,2,0),
            onehotTrimap=DataIOTrans.DataTrans.OneHotDecode(self.onehottrain_label2.permute(1,2,0)),
        )

    def __len__(self):
        # return length of

        return len(self.train_dir)


if __name__ == '__main__':
    dir = r'D:/Wrz/batchlors_code/landslide_dataset'
    Numclass = 2
    test = SlideDatasets(dir, Numclass=Numclass)
    print(test.__getitem__(0))
    test.visu()
    print(test.__len__())
