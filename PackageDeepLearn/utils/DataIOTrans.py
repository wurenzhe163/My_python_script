import numpy as np
import os
from osgeo import gdal
from tqdm import tqdm
from osgeo import gdal, ogr, osr
from PackageDeepLearn.utils import Statistical_Methods

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
        '''归一化'''
        if max == None and min == None:
            max,min = DataTrans.MinMaxArray(array)

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
    def MinMaxBoundary(array, bins=256, y=50):
        """
        array: 2d或3d矩阵
        bins : 直方图的bin数
        y    : 百分比截断数,前后截取
        """
        if len(array.shape) == 2:
            array = array.flatten()
            counts, bin_edges = np.histogram(array, bins=bins)
            boundary = Statistical_Methods.Cal_HistBoundary(counts, bin_edges, y=y)
            min_value = boundary['min_value']
            max_value = boundary['max_value']
            return max_value, min_value
        elif len(array.shape) == 3:
            min_values = []
            max_values = []
            for channel in range(array.shape[2]):
                channel_counts, channel_bin_edges = np.histogram(array[:, :, channel], bins=bins)
                channel_boundary = Statistical_Methods.Cal_HistBoundary(channel_counts, channel_bin_edges, y=y)
                min_values.append(channel_boundary['min_value'])
                max_values.append(channel_boundary['max_value'])
            return max_values, min_values
        else:
            raise ValueError("Array must be either 2D or 3D.")
        
    @staticmethod
    def MinMaxBoundaryScaler(array, bins=256, y=100, scale=1):
        '''
        scale : 归一化之后的放缩比例
        '''
        max,min = DataTrans.MinMaxBoundary(array, bins=bins, y=y)
        return DataTrans.MinMaxScaler(array, max=max, min=min, scale=scale)

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
    def rename_band(img_path, new_names: list, rewrite=False):
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"File {img_path} not found.")
        
        if not os.access(img_path, os.W_OK):
            raise PermissionError(f"No write permission for file {img_path}.")

        ds = gdal.Open(img_path, gdal.GA_Update)
        if ds is None:
            raise FileNotFoundError(f"Unable to open file {img_path}.")

        band_count = ds.RasterCount
        if band_count != len(new_names):
            raise ValueError('BandNames length does not match the number of bands.')

        try:
            for i in range(band_count):
                ds.GetRasterBand(i + 1).SetDescription(new_names[i])
            ds.FlushCache()  # Ensure all changes are written

            driver = gdal.GetDriverByName('GTiff')
            if rewrite:
                temp_file = img_path + '.tmp'
                dst_ds = driver.CreateCopy(temp_file, ds)
                dst_ds.FlushCache()
                dst_ds = None
                ds = None

                # Replace the original file with the temporary file
                os.replace(temp_file, img_path)
            else:
                DirName = os.path.dirname(img_path)
                BaseName = os.path.basename(img_path).split('.')[0] + '_Copy.' + os.path.basename(img_path).split('.')[1]
                dst_ds = driver.CreateCopy(os.path.join(DirName, BaseName), ds)
                dst_ds.FlushCache()
                dst_ds = None
        except Exception as e:
            raise RuntimeError(f"An error occurred while renaming bands: {e}")
        finally:
            ds = None  # Ensure dataset is closed

    @staticmethod
    def raster_to_vector(raster_paths, vector_path, retain_values=None, attributes=None, merge_features=False, append=False):
        """
        将栅格数据转换为矢量数据
        raster_paths: 栅格文件路径列表
        vector_path: 输出矢量文件路径
        retain_values: 要保留的DN值列表，默认None表示所有值
        merge_features: 是否将同一个DN值的所有要素合并为一个整体
        append: 是否追加到现有矢量文件
        attributes: 可选的属性字典，键为属性字段名，值为属性值列表
        """
        drv = ogr.GetDriverByName("ESRI Shapefile")
        
        if append and os.path.exists(vector_path):
            # 以追加模式打开现有矢量文件
            dst_ds = drv.Open(vector_path, 1)
            if dst_ds is None:
                raise Exception(f"Failed to open existing file {vector_path} in append mode.")
            dst_layer = dst_ds.GetLayer()
        else:
            # 创建新的矢量文件
            dst_ds = drv.CreateDataSource(vector_path)
            if dst_ds is None:
                raise Exception(f"Failed to create vector file {vector_path}.")
            
            # 设置空间参考系统为WGS84
            srs_wgs84 = osr.SpatialReference()
            srs_wgs84.ImportFromEPSG(4326)
            
            dst_layer = dst_ds.CreateLayer("polygonized", srs=srs_wgs84)
            if dst_layer is None:
                raise Exception(f"Failed to create layer in vector file {vector_path}.")
            
            # 添加字段
            fd = ogr.FieldDefn("DN", ogr.OFTInteger)
            dst_layer.CreateField(fd)

            # 添加属性字段
            if attributes:
                for attr_name, attr_value in attributes.items():
                    # 根据属性值的类型定义字段类型
                    if isinstance(attr_value, int):
                        fd = ogr.FieldDefn(attr_name, ogr.OFTInteger)
                    elif isinstance(attr_value, float):
                        fd = ogr.FieldDefn(attr_name, ogr.OFTReal)
                    else:
                        # 假设其他类型都为字符串类型
                        fd = ogr.FieldDefn(attr_name, ogr.OFTString)
                    
                    # 创建字段
                    dst_layer.CreateField(fd)

        for idx, raster_path in tqdm(enumerate(raster_paths)):
            # 打开栅格文件
            src_ds = gdal.Open(raster_path)
            if src_ds is None:
                print(f"Failed to open raster file {raster_path}. Skipping.")
                continue

            srcband = src_ds.GetRasterBand(1)

            # 获取栅格的空间参考系统
            src_srs = osr.SpatialReference()
            src_srs.ImportFromWkt(src_ds.GetProjection())

            # 如果栅格不是WGS84坐标系，则进行坐标转换
            if not src_srs.IsSame(srs_wgs84):
                dst_ds_warped = gdal.AutoCreateWarpedVRT(src_ds, src_ds.GetProjection(), srs_wgs84.ExportToWkt())
                srcband = dst_ds_warped.GetRasterBand(1)

            # 创建临时矢量文件
            temp_ds = drv.CreateDataSource('/vsimem/temp.shp')
            if temp_ds is None:
                print("Failed to create temporary vector file. Skipping.")
                src_ds = None
                continue

            temp_layer = temp_ds.CreateLayer("polygonized", srs=srs_wgs84)
            if temp_layer is None:
                print("Failed to create layer in temporary vector file. Skipping.")
                temp_ds = None
                src_ds = None
                continue

            # 添加字段
            fd = ogr.FieldDefn("DN", ogr.OFTInteger)
            temp_layer.CreateField(fd)

            # 栅格转矢量
            gdal.Polygonize(srcband, None, temp_layer, 0, [], callback=None)

            if merge_features:
                # 将同一个DN值的所有要素合并为一个整体
                unique_values = set()
                for feature in temp_layer:
                    unique_values.add(feature.GetField("DN"))
                
                for value in unique_values:
                    if retain_values is None or value in retain_values:
                        temp_layer.SetAttributeFilter(f"DN = {value}")
                        union_geom = None
                        for feature in temp_layer:
                            if union_geom is None:
                                union_geom = feature.GetGeometryRef().Clone()
                            else:
                                union_geom = union_geom.Union(feature.GetGeometryRef())
                        if union_geom is not None:
                            new_feature = ogr.Feature(dst_layer.GetLayerDefn())
                            new_feature.SetGeometry(union_geom)
                            new_feature.SetField("DN", value)
                            if attributes:
                                for attr_name, attr_values in attributes.items():
                                    new_feature.SetField(attr_name, attr_values[idx])
                            dst_layer.CreateFeature(new_feature)
                        temp_layer.SetAttributeFilter(None)
            else:
                # 将临时矢量文件的要素添加到目标矢量文件中
                for feature in temp_layer:
                    dn_value = feature.GetField("DN")
                    if retain_values is None or dn_value in retain_values:
                        new_feature = ogr.Feature(dst_layer.GetLayerDefn())
                        new_feature.SetGeometry(feature.GetGeometryRef())
                        new_feature.SetField("DN", dn_value)
                        if attributes:
                            for attr_name, attr_values in attributes.items():
                                new_feature.SetField(attr_name, attr_values[idx])
                        dst_layer.CreateFeature(new_feature)

            # 清理临时矢量文件
            temp_ds = None
            src_ds = None

        # 清理
        dst_ds = None


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
    def get_nodata(path):
        # 打开TIFF影像文件
        in_dataset = gdal.Open(path)

        # 检查是否成功打开
        if in_dataset is None:
            raise Exception('无法打开文件')

        # 获取Nodata值
        nodata_value = in_dataset.GetRasterBand(1).GetNoDataValue()
        in_dataset = None
        return nodata_value


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
    def TransImage_Values(path,transFunc=DataTrans.MinMaxScaler,bandSave:list=None,scale=1,
                          Nodata:list=None,datatype=gdal.GDT_Byte):
        '''
        path ： 输入图像路径
        transFunc : 任意关于array的变换函数(DataTrans.StandardScaler , DataTrans.MinMaxScaler, DataTrans.MinMax_Standard)
        datatype : gdal.GDT_Byte, gdal.GDT_UInt16, gdal.GDT_Float32. default = None以当前数据格式自动保存
        bandSave : 需要保留的波段 如:[0,1,2]
        Nodata   : Nodata转换，[0,255]图像中数值为0的像素转为255
        '''
        DirName = os.path.dirname(path)
        BaseName = os.path.splitext(os.path.basename(path))
        SavePath = os.path.join(DirName,BaseName[0] +'_Trans' +BaseName[1])
        data,img_transf,img_proj = DataIO.read_IMG(path,flag=1)
        if Nodata:
            data[data == Nodata[0]] = Nodata[1]
        if transFunc:
            data = transFunc(data) * scale
        if bandSave:
            data = data[:,:,bandSave]
        DataIO.save_Gdal(data,SavePath,datatype=datatype,img_transf = img_transf,img_proj = img_proj)
        return SavePath


class DataCrop(object):
    @staticmethod
    def get_extent(dataset):
        """获取栅格数据集的地理范围"""
        gt = dataset.GetGeoTransform()
        cols = dataset.RasterXSize
        rows = dataset.RasterYSize
        minx = gt[0]
        maxx = gt[0] + (gt[1] * cols)
        miny = gt[3] + (gt[5] * rows)
        maxy = gt[3]
        return minx, maxx, miny, maxy

    @staticmethod
    def calculate_intersection(extents:list):
        """
        计算一系列范围的交集
        extents : [get_extent(ds) for ds in datasets]
        """
        minx = max(extent[0] for extent in extents)
        maxx = min(extent[1] for extent in extents)
        miny = max(extent[2] for extent in extents)
        maxy = min(extent[3] for extent in extents)
        return minx, maxx, miny, maxy

    @staticmethod
    def check_projection_consistency(datasets:list):
        """
        检查所有数据集的投影是否一致
        datasets : [gdal.Open(tif, gdal.GA_ReadOnly) for tif in tif_files]
        """
        projections = [ds.GetProjection() for ds in datasets]
        if not all(proj == projections[0] for proj in projections):
            return False
        return True

    @staticmethod
    def reproject_datasets(datasets, target_projection, x_res=None, y_res=None):
        """
        重投影数据集集合，可以指定输出分辨率
        datasets : [gdal.Open(tif, gdal.GA_ReadOnly) for tif in tif_files]
        target_projection: gdalProjection 如：gdal.Open('tif_path', gdal.GA_ReadOnly).GetProjection()
        x_res : 行分辨率，不指定则保持
        y_res : 列分辨率，不指定则保持
        """
        reprojected_datasets = []
        for ds in datasets:
            warp_options = {
                'dstSRS': target_projection,
                'format': 'MEM'
            }
            if x_res and y_res:
                warp_options.update({
                    'xRes': x_res,
                    'yRes': y_res
                })
            reprojected_ds = gdal.Warp('', ds, **warp_options)
            reprojected_datasets.append(reprojected_ds)
        return reprojected_datasets

    @staticmethod
    def crop_datasets(datasets, file_names, minx, miny, maxx, maxy):
        """裁剪数据集集合，使用原始文件名列表来命名输出文件"""
        for ds, name in zip(datasets, file_names):
            output_file = f'cropped_{name}'
            gdal.Warp(output_file, ds, outputBounds=[minx, miny, maxx, maxy], cropToCutline=True)
            ds = None  # 关闭文件
    
    # os.chdir(r'D:\BaiduSyncdisk\02_论文相关\在写\几何畸变\数据\小范围对比\SAM提取结果\test')
    # # 获取所有TIF文件的路径
    # tif_files = [file for file in os.listdir('.') if file.endswith('.tif')]

    # # 打开所有栅格数据集
    # datasets = [gdal.Open(tif, gdal.GA_ReadOnly) for tif in tif_files]

    # # 检查投影一致性
    # if check_projection_consistency(datasets):
    #     # 如果投影一致，直接获取交集范围并裁剪
    #     extents = [get_extent(ds) for ds in datasets]
    #     minx, maxx, miny, maxy = calculate_intersection(extents)
    #     crop_datasets(datasets, tif_files, minx, miny, maxx, maxy)
    # else:
    #     # 如果投影不一致，重投影到目标投影，然后裁剪
    #     target_projection = gdal.Open(tif_files[0], gdal.GA_ReadOnly).GetProjection()  # 假设以第一个文件为目标投影
    #     x_res, y_res = 10, 10  # 示例分辨率，可以根据需要调整
    #     reprojected_datasets = reproject_datasets(datasets, target_projection, x_res, y_res)
    #     extents = [get_extent(ds) for ds in reprojected_datasets]
    #     minx, maxx, miny, maxy = calculate_intersection(extents)
    #     crop_datasets(reprojected_datasets, tif_files, minx, miny, maxx, maxy)

    # print(f"Processed all images. Intersection extent: {minx}, {maxx}, {miny}, {maxy}")