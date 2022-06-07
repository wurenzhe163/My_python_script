import glob
import cv2
import random
import numpy as np
import pickle
import os
from osgeo import gdal
import numpy as np

from torch.utils import data
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
 #       t = read_tif(os.path.join(self.config.datasets_dir, 'ground_truth', str(self.imlist[index]))).astype(np.float32)
 #       x = read_tif(os.path.join(self.config.datasets_dir, 'cloudy_image', str(self.imlist[index]))).astype(np.float32)
#        t = cv2.imread(os.path.join(self.config.datasets_dir, 'ground_truth', str(self.imlist[index])), 1).astype(np.float32)
#        x = cv2.imread(os.path.join(self.config.datasets_dir, 'cloudy_image', str(self.imlist[index])), 1).astype(np.float32)
        t = np.load(self.imlist[index],allow_pickle=True).item()["OriginImage"].astype(np.float32)
        x = np.load(self.imlist[index],allow_pickle=True).item()["Gimage"].astype(np.float32)
        #t = t[...,np.newaxis]
        #t = np.concatenate((t,t,t),axis=2)

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

class TrainDataset2(data.Dataset):

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
