import torch
import torchvision
import os,random
import pandas as pd
import numpy as np
from osgeo import gdal
import shutil
import cv2

from torch.utils.data import Dataset
from PackageDeepLearn.utils import DataIOTrans,Visualize
from models.gen.SPANet import Generator
def Agument(npArray:list,
            seed :int,):
    ''' 定义Preagument '''
    Agument = DataIOTrans.DataTrans.data_augmentation(ToTensor=True)
    NewArray = []
    for each in npArray:
        # 随机种子必须在循环内
        torch.manual_seed(seed)
        each = Agument(each)
        NewArray.append(each)

    return NewArray

def standardization(Array,Channels,Statistic,Imagetype='S2'):
    MinMax_Standard_array = DataIOTrans.DataTrans.MinMax_Standard(Array[..., 0:Channels],
                                          max=Statistic[Imagetype + '_valuesBack'].to_list(),
                                          min=Statistic[Imagetype + '_valuesFront'].to_list(),
                                          mean=Statistic[Imagetype + '_mean'].to_list(),
                                          std=Statistic[Imagetype + '_std'].to_list())
    return MinMax_Standard_array

def save_img(Visu_img,Path,name,norm=True,histogram=False):
    Spec_Visu = Visu_img  # RGB
    h,w,c = Spec_Visu.shape
    Spec_max = Spec_Visu.reshape(h*w,c).max(axis = 0)
    Spec_min = Spec_Visu.reshape(h*w,c).min(axis = 0)
    Spec_01 = DataIOTrans.DataTrans.MinMaxScaler(Spec_Visu,Spec_max,Spec_min)
    Visualize.save_img(path=Path,name=name,image = Spec_01,norm=norm,histogram=histogram)

class S1S2_CloudRemovalDataset(Dataset):
    '''
    生成训练用数据集
    csv_path：xx_concat.csv
    DataTag:'All'/'Real'/'Synth'
    '''
    def __init__(self,
                 Parendir,
                 Savepath,
                 ImageSize = (512,512),
                 DataTag = 'All',
                 SpecChannels=12,
                 SarChannels=2):

        csv_path = os.path.join(Parendir, '2020-06-01_2020-09-30_concat.csv')
        S2_Statistic = os.path.join(Parendir, 'S2\\S2_Statistic.csv')
        S1_Statistic = os.path.join(Parendir, 'S1\\S1_Statistic.csv')
        # 初始化参数
        self.Parendir = Parendir
        self.Savepath = Savepath
        self.ImageSize= ImageSize
        self.SpecChannels = SpecChannels
        self.SarChannels = SarChannels
        # 载入CSV文件
        self.Csv = pd.read_csv(csv_path)
        self.S2_Statistic = pd.read_csv(S2_Statistic)
        self.S1_Statistic = pd.read_csv(S1_Statistic)
        # 分为Real/Synth
        self.Real_Data = self.Csv [self.Csv ['part'] == 'Real']
        self.Synth_Data = self.Csv [self.Csv ['part'] == 'Synth']
        # 定义数据集取值域
        if DataTag == 'All':
            self.dataset = self.Csv
        elif DataTag == 'Real':
            self.dataset = self.Real_Data
        elif DataTag == 'Synth':
            self.dataset = self.Synth_Data
        elif DataTag == 'Random':
            self.dataset = self.Csv.sample(n=10)

    def __getitem__(self, i):
        self.Item = self.dataset.iloc[i]
        # 生成绝对路径(可以用相对路径替代)
        Dir = os.path.join(self.Parendir,self.Item.part,f'{int(self.Item.num):05d}')

        # 载入SAR图像升轨
        ASAR_path = os.path.join(Dir, self.Item.tif_ASCENDING)
        self.ASAR = DataIOTrans.DataIO.read_IMG(ASAR_path)
        self.ASAR = cv2.resize(self.ASAR,self.ImageSize)
        self.ASAR = standardization(self.ASAR,self.SarChannels,
                                    self.S1_Statistic,Imagetype='S1')
        # 载入SAR图像降轨
        DSAR_path = os.path.join(Dir, self.Item.tif_DESCENDING)
        self.DSAR = DataIOTrans.DataIO.read_IMG(DSAR_path)
        self.DSAR = cv2.resize(self.DSAR,self.ImageSize)
        self.DSAR = standardization(self.DSAR,self.SarChannels,
                                    self.S1_Statistic,Imagetype='S1')

        # 载入Spec与Cloud图像
        Spec_path = os.path.join(Dir,self.Item.tif)
        self.Spec = DataIOTrans.DataIO.read_IMG(Spec_path)
        self.Spec0 = cv2.resize(self.Spec,self.ImageSize)
        Cloud_path = os.path.join(Dir, self.Item.tif_Negtive)
        self.Cloud = DataIOTrans.DataIO.read_IMG(Cloud_path)
        self.Cloud = cv2.resize(self.Cloud,self.ImageSize)

        # 重新获取Cloud修复地表反射率
        self.Cloud0 = self.Cloud[..., :self.SpecChannels] * self.Cloud[..., 15, None] + \
                     (1 - self.Cloud[..., 15, None]) * self.Spec0[..., :self.SpecChannels]

        self.Spec = standardization(self.Spec0,self.SpecChannels,
                                    self.S2_Statistic,Imagetype='S2')
        self.Cloud = standardization(self.Cloud0,self.SpecChannels,
                                    self.S2_Statistic,Imagetype='S2')

        self.Sample = Agument([self.Spec,self.ASAR,self.DSAR,self.Cloud],
                              seed=torch.random.seed())
        return torch.concat([self.Sample[3],self.Sample[1],self.Sample[2]],dim=0),self.Sample[0]

    def Mv_tif(self):
        src_file_path = [self.Spec_path,self.ASAR_path,self.DSAR_path,self.Cloud_path]
        dst_file_path = [os.path.join(self.Savepath ,self.Item.part,
                            f'{self.Item.num:05d}', os.path.basename(each)) for each in src_file_path]
        for i,j in zip(src_file_path,dst_file_path):
            shutil.copy(i, j)
        pass

    def Save_jpg(self):
        Path = os.path.join(self.Savepath,self.Item.part, f'{self.Item.num:05d}')
        name = 'clear.jpg'
        save_img(self.Spec0[..., 3:0:-1], Path, name, norm=True, histogram=False)

        Path = os.path.join(self.Savepath,self.Item.part, f'{self.Item.num:05d}')
        name='Cloud.jpg'
        save_img(self.Cloud0[..., 3:0:-1], Path, name, norm=True, histogram=False)
        name = 'Cloud_hist.jpg'
        save_img(self.Cloud0[..., 3:0:-1], Path, name, norm=True, histogram=True)


    def __len__(self):
        return len(self.dataset)



# 模型预测，输出进行反标准化，RGB-JPG导出，原始tif导出


if __name__ == '__main__':
    # 导入CSV文件,对图像进行标准化，定义一个Class包含数据存取
    # 随机抽取样本，并进行可视化展示(保存图像至单独文件夹，原始文件，以及渲染后的RGB)
    Path = r'E:\Datasets\2020-06-01_2020-09-30_processed'
    PrePath = r'E:\Datasets\Pre'
    pretrained_model_path = r'E:\BaiduSyncdisk\09_Code\python-script\img_byme\DeepLearning_pytorch\Project\SpA-GAN_for_cloud_removal-master\results\000034\models\gen_model_epoch_1_iter_20000.pth'

    gen = Generator(gpu_ids=False, in_channels=14, out_channels=12).cpu()
    param = torch.load(pretrained_model_path)
    gen.load_state_dict(param)
    gen.eval()

    # model = ('mst_plus_plus',pretrained_model_path=pretrained_model_path).cpu()
    # model.eval()

    Test = S1S2_CloudRemovalDataset(
                 Parendir=Path,
                 Savepath=PrePath,
                 ImageSize = (512,512),
                 DataTag = 'Random',
                 SpecChannels=12,
                 SarChannels=1
                         )

    for i in range(len(Test)):
        A = Test[i][0]
        Test.Save_jpg()
        # Test.Mv_tif()

        with torch.no_grad():
            # compute output
            _,output = gen(A[None,...].float())

        output = torch.squeeze(output).permute(1,2,0).numpy()
        Path = os.path.join(Test.Savepath,Test.Item.part, f'{Test.Item.num:05d}')
        name = 'Pre.jpg'
        save_img(output[..., 3:0:-1], Path, name, norm=True, histogram=False)