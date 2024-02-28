import torch
import random
import numpy as np
import torchvision
import os
from osgeo import gdal
import pandas as pd
import cv2

from torch.utils import data
from data import DataIOTrans,Visualize

def Agument(npArray:list,
            CropSize:tuple,
            Resize:tuple,
            seed :int,
            args=2):

    CropSizes = random.randint(*CropSize)
    Agument0 = DataIOTrans.DataTrans.data_augmentation(ToTensor=True,
                                                       HFlip=0.5,VFlip=0.5,)

    Agument1 = DataIOTrans.DataTrans.data_augmentation(Crop=CropSizes,Resize=Resize)
    if args == 0 :
        Agument = Agument0
    elif args == 1:
        Agument = Agument1
    elif args == 2:
        Agument =  torchvision.transforms.Compose(Agument0.transforms+Agument1.transforms)
    else:
        print('Agument is wrong, please check parameter <args>')

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

class S1S2_pinmemory(data.Dataset):
    '''
    生成训练用数据集
    csv_path：xx_concat.csv
    DataTag:'All'/'Real'/'Synth'
    Batch_Sample：载入到内存一个Batch的样本数量
    Batch_Cycle: 每个Batch样本循环次数
    '''
    def __init__(self,
                 Parendir,
                 ImageSize = (512,512),
                 CropSize = (64,256),
                 Resize = (128,128),
                 DataTag = 'All',
                 SpecChannels=12,
                 SarChannels=1,
                 Batch_Sample=20,
                 Batch_Cycle=10):

        super().__init__()
        csv_path = os.path.join(Parendir, '2020-06-01_2020-09-30_concat.csv')
        S2_Statistic = os.path.join(Parendir, 'S2\\S2_Statistic.csv')
        S1_Statistic = os.path.join(Parendir, 'S1\\S1_Statistic.csv')
        # 初始化参数
        self.Parendir = Parendir
        self.ImageSize= ImageSize
        self.CropSize = CropSize
        self.Resize = Resize
        self.SpecChannels = SpecChannels
        self.SarChannels = SarChannels
        self.Batch_Sample = Batch_Sample
        self.Batch_Cycle = Batch_Cycle
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


    def NextBatch(self,num):
        '''
        预加载一个Batch
        '''
        self.Dataset_OneBatch = []
        for i in range(self.Batch_Sample):
            Item = self.dataset.iloc[i+num]
            # 生成绝对路径(可以用相对路径替代)
            Dir = os.path.join(self.Parendir, Item.part, f'{int(Item.num):05d}')

            # 载入SAR图像升轨
            ASAR_path = os.path.join(Dir, Item.tif_ASCENDING)
            self.ASAR = DataIOTrans.DataIO.read_IMG(ASAR_path)
            self.ASAR = cv2.resize(self.ASAR, self.ImageSize)
            self.ASAR = standardization(self.ASAR, self.SarChannels,
                                        self.S1_Statistic, Imagetype='S1')
            # 载入SAR图像降轨
            DSAR_path = os.path.join(Dir, Item.tif_DESCENDING)
            self.DSAR = DataIOTrans.DataIO.read_IMG(DSAR_path)
            self.DSAR = cv2.resize(self.DSAR, self.ImageSize)
            self.DSAR = standardization(self.DSAR, self.SarChannels,
                                        self.S1_Statistic, Imagetype='S1')


            # 载入Spec图像# 载入Cloud
            Spec_path = os.path.join(Dir, Item.tif)
            self.Spec_path = Spec_path
            self.Spec = DataIOTrans.DataIO.read_IMG(Spec_path)
            self.Spec = cv2.resize(self.Spec, self.ImageSize)

            Cloud_path = os.path.join(Dir, Item.tif_Negtive)
            self.Cloud = DataIOTrans.DataIO.read_IMG(Cloud_path)
            self.Cloud = cv2.resize(self.Cloud, self.ImageSize)

            # 重新获取Cloud修复地表反射率
            self.Cloud = self.Cloud[...,:self.SpecChannels] * self.Cloud[..., 15, None] + \
                         (1 - self.Cloud[..., 15, None]) * self.Spec[...,:self.SpecChannels]

            self.Spec = standardization(self.Spec, self.SpecChannels,
                                        self.S2_Statistic, Imagetype='S2')
            self.Cloud = standardization(self.Cloud, self.SpecChannels,
                                         self.S2_Statistic, Imagetype='S2')

            self.Sample = Agument([self.Spec,self.ASAR,self.DSAR,self.Cloud],
                                  CropSize=self.CropSize,
                                  Resize= self.Resize,
                                  seed=torch.random.seed(),
                                  args=0)

            self.Dataset_OneBatch.append(self.Sample)


    def __getitem__(self, i):

        Batch_i = i % (self.Batch_Sample * self.Batch_Cycle)
        # 重新载入新的Batch
        if i==0 or Batch_i == 0:
            # 将计数值传递到函数
            self.NextBatch(num=i//self.Batch_Cycle)

        self.Data = Agument(
                  self.Dataset_OneBatch[Batch_i % self.Batch_Sample],
                  CropSize=self.CropSize,
                  Resize= self.Resize,
                  seed=torch.random.seed(),
                  args=1)
        # print('数据集取索引i={}'.format(i))

        M = torch.clip((self.Data[3] - self.Data[0]).sum(axis=0), 0, 1)
        return torch.concat([self.Data[3], self.Data[1], self.Data[2]], dim=0), self.Data[0],M

    def visu(self):
        '''
        展示图像
        '''
        Visualize.visualize(
            Clear=self.Data[0][2, ...],
            ASAR=self.Data[1][1, ...],
            DSAR=self.Data[2][1, ...],
            Cloud=self.Data[3][2, ...],
        )
        pass

    def __len__(self):
        return len(self.dataset) * self.Batch_Cycle

class val_CloudRemovalDataset(data.Dataset):
    '''
    生成训练用数据集
    csv_path：xx_concat.csv
    DataTag:'All'/'Real'/'Synth'
    '''
    def __init__(self,
                 Parendir,
                 ImageSize = (512,512),
                 CropSize = (128,128),
                 Resize = (128,128),
                 DataTag = 'All',
                 SpecChannels=12,
                 SarChannels=2):

        csv_path = os.path.join(Parendir, '2020-06-01_2020-09-30_concat_val.csv')
        S2_Statistic = os.path.join(Parendir, 'S2\\S2_Statistic.csv')
        S1_Statistic = os.path.join(Parendir, 'S1\\S1_Statistic.csv')
        # 初始化参数
        self.Parendir = Parendir
        self.ImageSize= ImageSize
        self.CropSize = CropSize
        self.Resize = Resize
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

    def check_out(self):
        pass

    def __getitem__(self, i):
        Item = self.dataset.iloc[i]
        # 生成绝对路径(可以用相对路径替代)
        Dir = os.path.join(self.Parendir,Item.part,f'{int(Item.num):05d}')

        # 载入SAR图像升轨
        ASAR_path = os.path.join(Dir, Item.tif_ASCENDING)
        self.ASAR = DataIOTrans.DataIO.read_IMG(ASAR_path)
        self.ASAR = cv2.resize(self.ASAR,self.ImageSize)
        self.ASAR = standardization(self.ASAR,self.SarChannels,
                                    self.S1_Statistic,Imagetype='S1')
        # 载入SAR图像降轨
        DSAR_path = os.path.join(Dir, Item.tif_DESCENDING)
        self.DSAR = DataIOTrans.DataIO.read_IMG(DSAR_path)
        self.DSAR = cv2.resize(self.DSAR,self.ImageSize)
        self.DSAR = standardization(self.DSAR,self.SarChannels,
                                    self.S1_Statistic,Imagetype='S1')

        # 载入Spec与Cloud图像
        Spec_path = os.path.join(Dir,Item.tif)
        self.Spec = DataIOTrans.DataIO.read_IMG(Spec_path)
        self.Spec = cv2.resize(self.Spec,self.ImageSize)
        Cloud_path = os.path.join(Dir, Item.tif_Negtive)
        self.Cloud = DataIOTrans.DataIO.read_IMG(Cloud_path)
        self.Cloud = cv2.resize(self.Cloud,self.ImageSize)

        # 重新获取Cloud修复地表反射率
        self.Cloud = self.Cloud[..., :self.SpecChannels] * self.Cloud[..., 15, None] + \
                     (1 - self.Cloud[..., 15, None]) * self.Spec[..., :self.SpecChannels]

        self.Spec = standardization(self.Spec,self.SpecChannels,
                                    self.S2_Statistic,Imagetype='S2')
        self.Cloud = standardization(self.Cloud,self.SpecChannels,
                                    self.S2_Statistic,Imagetype='S2')

        self.Sample = Agument([self.Spec,self.ASAR,self.DSAR,self.Cloud],
                              CropSize=self.CropSize,
                              Resize= self.Resize,
                              seed=torch.random.seed())
        return torch.concat([self.Sample[3],self.Sample[1],self.Sample[2]],dim=0),self.Sample[0]

    def visu(self):
        '''
        展示图像
        '''
        Visualize.visualize(
            Clear=self.Sample[0][2, ...],
            ASAR=self.Sample[1][1, ...],
            DSAR=self.Sample[2][1, ...],
            Cloud=self.Sample[3][2, ...],
        )
        pass

    def __len__(self):
        return len(self.dataset)


def read_tif(path):

    dataset = gdal.Open(path)
    if dataset == None:
        raise Exception("Unable to read the data file")

    nXSize = dataset.RasterXSize  # 列数
    nYSize = dataset.RasterYSize  # 行数
    bands = dataset.RasterCount  # 波段

    Raster1 = dataset.GetRasterBand(1)


    if Raster1.DataType == 1:
        datatype = np.uint8
    elif Raster1.DataType == 2:
        datatype = np.uint16
    else:
        datatype = float

    data = np.zeros([nYSize, nXSize, bands], dtype=datatype)
    for i in range(bands):
        band = dataset.GetRasterBand(i + 1)
        data[:, :, i] = band.ReadAsArray(0, 0, nXSize, nYSize)  # .astype(np.complex)
    return data


class TrainDataset(data.Dataset):

    def __init__(self, config):
        super().__init__()
        self.config = config

        train_list_file = os.path.join(config.datasets_dir, config.train_list)
        # 如果数据集尚未分割，则进行训练集和测试集的分割
        if not os.path.exists(train_list_file) or os.path.getsize(train_list_file) == 0:
            files = os.listdir(os.path.join(config.datasets_dir, 'ground_truth'))
            random.shuffle(files)
            n_train = int(config.train_size * len(files))
            train_list = files[:n_train]
            test_list = files[n_train:]
            np.savetxt(os.path.join(config.datasets_dir, config.train_list), np.array(train_list), fmt='%s')
            np.savetxt(os.path.join(config.datasets_dir, config.test_list), np.array(test_list), fmt='%s')

        self.imlist = np.loadtxt(train_list_file, str)

    def __getitem__(self, index):
        t = read_tif(os.path.join(self.config.datasets_dir, 'ground_truth', str(self.imlist[index]))).astype(np.float32)
        x = read_tif(os.path.join(self.config.datasets_dir, 'cloudy_image', str(self.imlist[index]))).astype(np.float32)
#        t = cv2.imread(os.path.join(self.config.datasets_dir, 'ground_truth', str(self.imlist[index])), 1).astype(np.float32)
#        x = cv2.imread(os.path.join(self.config.datasets_dir, 'cloudy_image', str(self.imlist[index])), 1).astype(np.float32)

        M = np.clip((t-x).sum(axis=2), 0, 1).astype(np.float32)
        #x = x / 255
        #t = t / 255
        x = x / 10000
        t = t / 10000
        x = x.transpose(2, 0, 1)
        t = t.transpose(2, 0, 1)

        return x, t, M

    def __len__(self):
        return len(self.imlist)





class TestDataset(data.Dataset):
    def __init__(self, test_dir, in_ch, out_ch):
        super().__init__()
        self.test_dir = test_dir
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.test_files = os.listdir(os.path.join(test_dir, 'cloudy_image'))

    def __getitem__(self, index):
        filename = os.path.basename(self.test_files[index])
        
        # x = cv2.imread(os.path.join(self.test_dir, 'cloudy_image', filename), 1).astype(np.float32)

        x = read_tif(os.path.join(self.test_dir, 'cloudy_image', filename)).astype(np.float32)
        # x = x / 255
        x = x / 10000

        x = x.transpose(2, 0, 1)

        return x, filename

    def __len__(self):

        return len(self.test_files)
