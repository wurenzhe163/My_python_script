import numpy as np
import os
from osgeo import gdal
from tqdm import tqdm

search_files = lambda path : sorted([os.path.join(path,f) for f in os.listdir(path)])

def make_dir(path):
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        os.makedirs(path)
        print(path + ' 创建成功')
        return path
    return path

class Denormalize(object):
    '''
    return : 反标准化
    '''
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return transforms.functional.normalize(tensor, self._mean, self._std)

class DataTrans(object):
    @staticmethod
    def OneHotEncode(LabelImage,NumClass):
        '''
        Onehot Encoder  2021/03/23 by Mr.w
        -------------------------------------------------------------
        LabelImage ： Ndarry   |   NumClass ： Int
        -------------------------------------------------------------
        return： Ndarry
        '''
        one_hot_codes = np.eye(NumClass).astype(np.uint8)
        try:
            one_hot_label = one_hot_codes[LabelImage]
        except IndexError:
            # pre treat cast 不连续值 到 连续值
            Unique_code = np.unique(LabelImage)
            Objectives_code = np.arange(len(Unique_code))
            for i, j in zip(Unique_code, Objectives_code):
                LabelImage[LabelImage == i] = j
                # print('影像编码从{},转换为{}'.format(i, j))

            one_hot_label = one_hot_codes[LabelImage]
        # except IndexError:
        #     # print('标签存在不连续值，最大值为{}--->已默认将该值进行连续化,仅适用于二分类'.format(np.max(LabelImage)))
        #     LabelImage[LabelImage == np.max(LabelImage)] = 1
        #     one_hot_label = one_hot_codes[LabelImage]
        return one_hot_label

    @staticmethod
    def OneHotDecode(OneHotImage):
        '''
        OneHotDecode 2021/03/23 by Mr.w
        -------------------------------------------------------------
        OneHotImage : ndarray -->(512,512,x)
        -------------------------------------------------------------
        return : image --> (512,512)
        '''
        return np.argmax(OneHotImage,axis=-1)

    @staticmethod
    def calculate_mean_std(array):
        """
        计算多波段数组每个波段的均值和标准差。
        Args:
            array: 三维NumPy数组，形状为 (高, 宽, 波段数)
        Returns:
            mean: 每个波段的均值组成的列表
            std: 每个波段的标准差组成的列表
        """
        # 检查输入数组是否为三维数组
        if len(array.shape) != 3:
            raise ValueError("输入数组必须是三维的。")
        
        # 获取数组的高、宽和波段数
        h, w, c = array.shape
        
        # 初始化均值和标准差列表
        mean = []
        std = []
        
        # 遍历每个波段，计算均值和标准差
        for i in range(c):
            band = array[:, :, i]
            band_mean = np.mean(band)
            band_std = np.std(band)
            mean.append(band_mean)
            std.append(band_std)
        return mean, std
    
    @staticmethod
    def StandardScaler(array, mean=None, std=None):
        """
        Z-Norm
        Args:
            array: 矩阵 ndarray
            mean: 均值 list
            std: 方差 list
        Returns:标准化后的矩阵
        """
        if mean == None and std == None:
            mean,std = DataTrans.calculate_mean_std(array)

        if len(array.shape) == 2:
            return (array - mean) / std

        elif len(array.shape) == 3:
            array_ = np.zeros_like(array).astype(np.float64)
            h, w, c = array.shape
            for i in range(c):
                array_[:, :, i] = (array[:, :, i] - mean[i]) / std[i]
        return array_
    
    @staticmethod
    def MinMaxArray(array):
        '''计算最大最小值'''
        if len(array.shape) == 2:
            array = array[...,None]

        if len(array.shape) == 3:
            h, w, c = array.shape
            max = [];min=[]
            for i in range(c):
                max.append(np.max(array[:,:,i]))
                min.append(np.min(array[:,:,i]))
        else:
            print('array.shape is wrong')

        return max,min

    @staticmethod
    def MinMaxScaler(array, max=None, min=None,scale = 1):
        if max == None and min == None:
            max,min = DataTrans.MinMaxArray(array)
        '''归一化'''
        if len(array.shape) == 2:
            return (array - min) / (max-min)

        elif len(array.shape) == 3:
            array_ = np.zeros_like(array).astype(np.float64)
            h, w, c = array.shape
            for i in range(c):
                array_[:, :, i] = (array[:, :, i] - min[i]) / (max[i]-min[i]) * scale
        return array_

    @staticmethod
    def MinMax_Standard(array, max:list=None, min:list=None,mean:list=None,std:list=None):
        '''
        先归一化，再标准化.
        注意该标准化所用的mean和std均是归一化之后计算值
        '''
        if max == None and min == None and mean == None and std == None:
            max,min = DataTrans.MinMaxArray(array)
            mean,std = DataTrans.calculate_mean_std(array)
            
        if len(array.shape) == 2:
            _ = (array - min) / (max - min)
            return (_ - mean) / std

        elif len(array.shape) == 3:
            array_1 = np.zeros_like(array).astype(np.float64)
            array_2 = np.zeros_like(array).astype(np.float64)
            h, w, c = array.shape
            for i in range(c):
                array_1[:, :, i] = (array[:, :, i] - min[i]) / (max[i]-min[i])
                array_2[:, :, i] = (array_1[:, :, i] - mean[i]) / std[i]
        return array_2

    @staticmethod
    def copy_geoCoordSys(img_pos_path, img_none_path):
        '''
        获取img_pos坐标，并赋值给img_none
        :param img_pos_path: 带有坐标的图像
        :param img_none_path: 不带坐标的图像
        '''

        def def_geoCoordSys(read_path, img_transf, img_proj):
            array_dataset = gdal.Open(read_path)
            img_array = array_dataset.ReadAsArray(0, 0, array_dataset.RasterXSize, array_dataset.RasterYSize)
            if 'int8' in img_array.dtype.name:
                datatype = gdal.GDT_Byte
            elif 'int16' in img_array.dtype.name:
                datatype = gdal.GDT_UInt16
            else:
                datatype = gdal.GDT_Float32

            if len(img_array.shape) == 3:
                img_bands, im_height, im_width = img_array.shape
            else:
                img_bands, (im_height, im_width) = 1, img_array.shape

            filename = read_path[:-4] + '_proj' + read_path[-4:]
            driver = gdal.GetDriverByName("GTiff")  # 创建文件驱动
            dataset = driver.Create(filename, im_width, im_height, img_bands, datatype)
            dataset.SetGeoTransform(img_transf)  # 写入仿射变换参数
            dataset.SetProjection(img_proj)  # 写入投影

            # 写入影像数据
            if img_bands == 1:
                dataset.GetRasterBand(1).WriteArray(img_array)
            else:
                for i in range(img_bands):
                    dataset.GetRasterBand(i + 1).WriteArray(img_array[i])
            print(read_path, 'geoCoordSys get!')

        dataset = gdal.Open(img_pos_path)  # 打开文件
        img_pos_transf = dataset.GetGeoTransform()  # 仿射矩阵
        img_pos_proj = dataset.GetProjection()  # 地图投影信息
        def_geoCoordSys(img_none_path, img_pos_transf, img_pos_proj)

    @staticmethod
    def rename_band(img_path,new_names:list,rewrite=False):
        ds = gdal.Open(img_path)
        band_count = ds.RasterCount
        assert band_count == len(new_names) , 'BnadNames length not match'
        for i in range(band_count):
            ds.GetRasterBand(i+1).SetDescription(new_names[i])
        driver = gdal.GetDriverByName('GTiff')
        if rewrite:
            dst_ds = driver.CreateCopy(img_path, ds)
        else:
            DirName = os.path.dirname(img_path)
            BaseName = os.path.basename(img_path).split('.')[0]+'_Copy.'+os.path.basename(img_path).split('.')[1]
            dst_ds = driver.CreateCopy(os.path.join(DirName,BaseName), ds)
        dst_ds = None
        ds = None




try:
    import torch
    from torchvision import transforms

    @staticmethod
    def data_augmentation(ToTensor=False,ColorJitter:tuple=None,Resize=None,Contrast=None,Equalize=None,HFlip=None,Invert=None,VFlip=None,
                          Rotation=None,Grayscale=None,Perspective=None,Erasing=None,Crop=None,
                          ): # dHFlip=None
        """

        DataAgumentation 2021/03/23 by Mr.w
        -------------------------------------------------------------
        ToTensor : False/True , 注意转为Tensor，通道会放在第一维
        ColorJitter: tulpe  brightness-->0-1 ,contrast-->0-1,  saturation-->0-1, hue(色调抖动)-->0-0.5
        Resize : tuple-->(500,500)
        Contrast : 0-1 -->图像被自动对比度的可能,支持维度1-3
        Equalize : 0-1 -->图像均衡可能性 , 仅支持uint8
        HFlip : 0-1 --> 图像水平翻转
        Invert : 0-1--> 随机反转给定图像的颜色
        VFlip : 0-1 --> 图像垂直翻转
        Rotation : 0-360 --> 随机旋转度数范围, as : 90 , [-90,90]，支持维度1-3
        Grayscale : 0-1 --> 随机转换为灰度图像
        Perspective : 0-1 --> 随机扭曲图像
        Erasing : 0-1 --> 随机擦除
        Crop : tuple --> (500,500)
        -------------------------------------------------------------
        return : transforms.Compose(train_transform) --> 方法汇总
        """
        #列表导入Compose
        train_transform = []
        if ToTensor == True:
            trans_totensor = transforms.ToTensor()
            train_transform.append(trans_totensor)
        if ColorJitter != None:
            train_transform.append(transforms.ColorJitter(
                ColorJitter[0], ColorJitter[1], ColorJitter[2], ColorJitter[3]))
        if Contrast != None:
            trans_Rcontrast = transforms.RandomAutocontrast(p=Contrast)
            train_transform.append(trans_Rcontrast)
        if Equalize != None:
            trans_REqualize = transforms.RandomEqualize(p=Equalize)
            train_transform.append(trans_REqualize)
        if HFlip != None:
            train_transform.append(transforms.RandomHorizontalFlip(p=HFlip))
        if Invert != None:
            train_transform.append(transforms.RandomInvert(p=Invert))
        if VFlip != None:
            train_transform.append(transforms.RandomVerticalFlip(p=VFlip))
        if Rotation != None:
            train_transform.append(transforms.RandomRotation(Rotation,expand=False,center=None,fill=0,resample=None))
        if Grayscale != None:
            train_transform.append(transforms.RandomGrayscale(p=Grayscale))
        if Perspective != None:
            train_transform.append(transforms.RandomPerspective(distortion_scale=0.5,p=Perspective,fill=0))
        if Erasing != None:
            train_transform.append(transforms.RandomErasing(p=Erasing,scale=(0.02, 0.33),ratio=(0.3, 3.3),value=0,inplace=False))
        if Crop != None:
            train_transform.append(transforms.RandomCrop(Crop,padding=None,pad_if_needed=False,fill=0,padding_mode='constant'))
        if Resize != None:
            trans_Rsize = transforms.Resize(Resize)  # Resize=(500,500)
            train_transform.append(trans_Rsize)

        return transforms.Compose(train_transform)
    setattr(DataTrans, "data_augmentation", data_augmentation)
except:
    print('torch or torchvision do not import')


class DataIO(object):
    @staticmethod
    def _get_dir(*path,DATA_DIR=r''):
        """
        as : a = ['train', 'val', 'test'] ; _get_dir(*a,DATA_DIR = 'D:\\deep_road\\tiff')
        :return list path
        """
        return [os.path.join(DATA_DIR, each) for each in path]

    @staticmethod
    def read_IMG(path,flag=0,datatype=None):
        """
        读为一个numpy数组,读取所有波段
        path : img_path as:c:/xx/xx.tif
        flag:0不反回坐标，1返回坐标
        """
        dataset = gdal.Open(path)
        if dataset == None:
            raise Exception("Unable to read the data file")

        nXSize = dataset.RasterXSize  # 列数
        nYSize = dataset.RasterYSize  # 行数
        bands = dataset.RasterCount  # 波段
        Raster1 = dataset.GetRasterBand(1)
        if flag==1:
            img_transf = dataset.GetGeoTransform()  # 仿射矩阵
            img_proj = dataset.GetProjection()  # 地图投影信息
        if datatype is None:
            if Raster1.DataType == 1 :
                datatype = np.uint8
            elif Raster1.DataType == 2:
                datatype = np.uint16
            else:
                datatype = float
        data = np.zeros([nYSize, nXSize, bands], dtype=datatype)
        for i in range(bands):
            band = dataset.GetRasterBand(i + 1)
            data[:, :, i] = band.ReadAsArray(0, 0, nXSize, nYSize)  # .astype(np.complex)
        if flag==0:
            return data
        elif flag==1:
            return data,img_transf,img_proj
        else:
            print('None Output, please check')

    @staticmethod
    def save_Gdal(img_array, SavePath, datatype=None, img_transf=None, img_proj=None):
        """

        Args:
            img_array:  [H,W,C] , RGB色彩，不限通道深度
            SavePath:
            transf: 是否进行投影
            img_transf: 仿射变换
            img_proj: 投影信息

        Returns: 0

        """
        dirname = os.path.dirname(SavePath)
        if os.path.isabs(dirname):
            make_dir(dirname)

        # 判断数据类型
        if datatype == None:
            if 'int8' in img_array.dtype.name:
                datatype = gdal.GDT_Byte
            elif 'int16' in img_array.dtype.name:
                datatype = gdal.GDT_UInt16
            else:
                datatype = gdal.GDT_Float32

        # 判断数据维度，仅接受shape=3或shape=2
        if len(img_array.shape) == 3:
            im_height, im_width, img_bands = img_array.shape
        else:
            img_bands, (im_height, im_width) = 1, img_array.shape

        driver = gdal.GetDriverByName("GTiff")  # 创建文件驱动
        dataset = driver.Create(SavePath, im_width, im_height, img_bands, datatype)

        if img_transf and img_proj:
            dataset.SetGeoTransform(img_transf)  # 写入仿射变换参数
            dataset.SetProjection(img_proj)  # 写入投影

        # 写入影像数据
        if len(img_array.shape) == 2:
            dataset.GetRasterBand(1).WriteArray(img_array)
        else:
            for i in range(img_bands):
                dataset.GetRasterBand(i + 1).WriteArray(img_array[:, :, i])

        dataset = None

    @staticmethod
    def TransImage_Values(path,transFunc=DataTrans.MinMaxScaler,bandSave:list=None,scale=1,datatype=gdal.GDT_Byte):
        '''
        path ： 输入图像路径
        transFunc : 任意关于array的变换函数(DataTrans.StandardScaler , DataTrans.MinMaxScaler, DataTrans.MinMax_Standard)
        datatype : gdal.GDT_Byte, gdal.GDT_UInt16, gdal.GDT_Float32. default = None以当前数据格式自动保存
        bandSave : 需要保留的波段 如:[0,1,2]
        '''
        DirName = os.path.dirname(path)
        BaseName = os.path.splitext(os.path.basename(path))
        SavePath = os.path.join(DirName,BaseName[0] +'_Trans' +BaseName[1])
        data,img_transf,img_proj = DataIO.read_IMG(path,flag=1)
        if transFunc:
            data = transFunc(data) * scale
        if bandSave:
            data = data[:,:,bandSave]
        DataIO.save_Gdal(data,SavePath,datatype=datatype,img_transf = img_transf,img_proj = img_proj)
        return SavePath
