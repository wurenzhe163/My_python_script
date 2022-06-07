import sys,PIL,os
import numpy as np
from functools import partial
from osgeo import gdal
# imports for building the network
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow import reduce_sum
from tensorflow.keras.backend import pow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, Concatenate, Add, Flatten, Activation
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
#定位个人编辑的库
sys.path.append(os.getcwd())
import tf_img_seg as tfis
from PIL import Image
# 需要修改
dir_name = r'D:\deep_road\tiff\haha'
seg_mark = 1   # 1 分割电板（多分类）    2 分割风机
model_name = 'mobile_unet 400.h5'  # 模型名称

erod_dilate_size = 0                                    #腐蚀膨胀算子大小，erod_dilate_size=0 则不进行该项运算,通常采用 5或3

#可以不修改
img_path = dir_name + '\\' + 'image'                    #原始影像
img_path_save = dir_name + '\\' + 'image_splice'        # 切分后影像保存位置
kernel = [512, 512]                                     #切块大小
stride = 512                                            #切块步长
img_label = dir_name + '\\'+'label'                     #标签图像，没有标签时无法进行评估
label_path_save = dir_name + '\\' + 'label_splice'      #切分后标签
img_pre = dir_name + '\\' + 'pre'                       # 预测切块
model_h5 = dir_name +'\\' + 'Weight' + '\\' + model_name  #模型名称
img_pre_union = dir_name + '\\' + 'pre_union'           # 合并切块位置
logs = dir_name + '\\' + 'logs'
batch__ = 6                                             #评估时的分组数


# os.chdir(dir_name)
# os.chdir(dir_name)
def main(mark,img_path=img_path,img_path_save=img_path_save,kernel=kernel,stride=stride,img_label=img_label,
         label_path_save=label_path_save,img_pre=img_pre,model_h5=model_h5,img_pre_union=img_pre_union,seg_mark=seg_mark):
    '''
    #输入必要的参数
     img_path: 底图所在路径
     img_path_save:底图切片保存路径
     kernel - --->卷积核大小, 矩阵int
     stride - --->步长, int
     img_label: label所在路径
     label_path_save:label切片保存路径
     img_pre:深度学习预测结果保存路径
     ckpt:选择生成的权重中其中一个
     img_pre_union:预测结果合并，并赋予底图对应的坐标系
    '''
    if type(mark) == int:
        if mark == 0:
            endwith = '.png'
            dtype = np.uint8
            save_name = 'image'
            Img_pre(img_path, kernel, stride, endwith, dtype, img_path_save, save_name).main()
            try :
                judg= Img_pre(path=img_label, endwith='.tif').search_files()
                if len(judg) != 0:
                    save_name2 = 'label'
                    Img_pre(img_label, kernel, stride, endwith, dtype, label_path_save, save_name2).main()
                else :
                    print('label文件夹:  {}  为空'.format(img_label))
            except FileNotFoundError:
                print('没有label文件夹: {}'.format(img_label))



        #预测
        if mark == 1:
            #准备数据
            img_path_ = Img_pre(path=img_path_save, endwith='.png').search_files()
            pre = tf.data.Dataset.from_tensor_slices(img_path_)
            pre_dataset = pre.map(partial(tfis.load_pro_2, kernel=kernel))

            #调用模型
            if seg_mark == 1:
                model = tf.keras.models.load_model(model_h5)
            elif seg_mark == 2:
                model = tfis.UNet(kernel[0], kernel[1])
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
                model.compile(optimizer=optimizer, loss=[tfis.focal_tversky_loss], metrics=[tfis.tversky, 'accuracy', tfis.iou])
                model.load_weights(model_h5)
                # tfis.model_loadweight(model, model_h5, load_pretrained_model=True)

            model.summary()
            tfis.prediction(model, seg_mark=seg_mark,dataset=pre_dataset, pathname=img_pre, name='pre')


        #影像拼接、闭运算 并赋予原始图像坐标系
        if mark == 2:
            path_pre = Img_pre(path=img_pre, endwith='.png').search_files()

            if erod_dilate_size != 0:
                print('执行影像腐蚀膨胀算法')
                # 腐蚀膨胀
                files_num = len(path_pre)
                num = 0
                for each_image in path_pre:
                    image = PIL.Image.open(each_image)
                    image = np.asarray(image, dtype=np.uint8)
                    num += 1
                    print('执行到第{}个文件，共{}个文件'.format(num,files_num))
                    if np.max(image) != 0:
                        #######连接
                        #膨胀
                        image = Img_Post.dilate_image(image,erod_dilate_size)
                        #腐蚀
                        image = Img_Post.erode_image(image, erod_dilate_size)

                        #######削减碎斑
                        #腐蚀
                        image = Img_Post.erode_image(image,erod_dilate_size)
                        #膨胀
                        image = Img_Post.dilate_image(image,erod_dilate_size)
                        image = np.around(image).astype(np.uint8)
                        A = PIL.Image.fromarray(image)
                        A.save(each_image)#, quality=95)   #覆盖原文件
                    else:
                        continue
            else:
                print('跳过影像腐蚀膨胀运算')
            #加载分割前的图片
            img_path_ = Img_pre(path=img_path, endwith='.tiff').search_files()
            for i, tif_path in enumerate(img_path_):
                # 拼接时所需参数
                rsw = Img_Post.read_shape(tif_path, dtype=np.uint8, kernel=kernel, stride=stride)
                img_path_2 = path_pre[0:rsw[3] * rsw[4]]
                del path_pre[0:rsw[3] * rsw[4]]
                img_size = kernel.copy()
                img_size.append(1)
                mask = Img_Post.mosaic_img(img_path_2, rsw, endwith='.png', stride=stride,img_size=img_size)
                name0 = 'pre_all' + f'{(i+1):04d}'
                Img_Post.save_image(np.squeeze(mask), img_pre_union,name=name0)

            non_ =  Img_pre(path=img_pre_union, endwith='.png').search_files()


            print('********执行重投影*********')
            pos =  img_path_
            for each_non, each_pos in zip(non_, pos):
                Img_Post.copy_geoCoordSys(each_pos, each_non)


        # 评价,需要有lable
        if mark == 3:

            try :
                judg= Img_pre(path=label_path_save, endwith='.png').search_files()
                if len(judg) != 0:
                    img_path = Img_pre(path=img_path_save, endwith='.png').search_files()
                    img_path = sorted(img_path, reverse=False)
                    label_path = Img_pre(path=label_path_save, endwith='.png').search_files()
                    label_path = sorted(label_path, reverse=False)
                    txt_all2 = [img_path[i] + ' ' + label_path[i] for i in range(len(img_path))]
                    test = tf.data.Dataset.from_tensor_slices(txt_all2)
                    #dataset = test.map(tfis.load_pro_2)
                    dataset = test.map(partial(tfis.load_pro_, kernel=kernel)).batch(batch__)
                    if seg_mark == 1:
                        model = tf.keras.models.load_model(model_h5)
                        tfis.evaluate_1(model,dataset=dataset)
                    if seg_mark == 2:
                        model = tfis.UNet(kernel[0], kernel[1])
                        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
                        model.compile(optimizer=optimizer, loss=[tfis.focal_tversky_loss], metrics=[tfis.tversky, 'accuracy', tfis.iou])
                        model.load_weights(model_h5)
                        tfis.evaluate_2(model,dataset=dataset)
                else :
                    print('label文件夹:  {}  为空'.format(label_path_save))
            except FileNotFoundError:
                print('没有label文件夹: {}'.format(label_path_save))

        # #tensorboard 可视化,定位到logs所在路径
        # if mark == 4 :
        #     os.system("tensorboard --logdir=logs")
    else:
        print('mark is an int number from 0-4.')


def wash_image(img_path_save,label_path_save):
    img_path = Img_pre(path=img_path_save, endwith='.png').search_files()
    img_path = sorted(img_path, reverse=False)
    label_path = Img_pre(path=label_path_save, endwith='.png').search_files()
    label_path = sorted(label_path, reverse=False)
    files_num = len(label_path)
    num = 0
    for each_label , each_img in zip(label_path,img_path):
        Label_img = PIL.Image.open(each_label)
        Label_img = np.array(Label_img,dtype=np.uint8)
        num += 1
        if num % 300 == 0:
            print('执行到第{}个文件，共{}个文件'.format(num,files_num))
        if np.sum(Label_img) == 0:
            os.remove(each_label)
            os.remove(each_img)


class Img_pre(object):
    def __init__(self,path=[],kernel=[],stride=[],endwith=[],dtype=[],save_path=[],save_name=[]):
        """
        path: 文件所在文件夹路径
        kernel - --->卷积核大小, 矩阵int
        stride - --->步长, int
        endwith: 文件名后缀
        dtype: 文件类型
        save_path: 文件保存文件夹路径
        save_name: 文件保存名称
        """
        self.path = path
        self.kernel = kernel
        self.stride = stride
        self.endwith = endwith
        self.dtype = dtype
        self.save_path = save_path
        self.save_name = save_name

        #中间变量
        self.var1 = '  '
        self.var2 = '  '

    def search_files(self):
        path = self.path
        endwith = self.endwith
        """
        返回当前文件夹下文件
        path : 路径
        endwith : The default is '.tif'.
        Returns: s ,   列表
        """
        s = []
        for f in os.listdir(path):
            if f.endswith(endwith):
                s.append(os.path.join(path, f))
        self.var1 = s
        return s
    def read_IMG(self,path):
        dtype = self.dtype
        '''
        PIL读图
        '''
        image = Image.open(path)
        image = np.asarray(image,dtype)
        return image
    def readtif_GDAL(self,path):     #需要输入影像path
        dtype = self.dtype
        """
        读为一个numpy数组,读取所有波段
        path : img_path as:c:/xx/xx.tif
        """
        dataset = gdal.Open(path)
        nXSize = dataset.RasterXSize  # 列数
        nYSize = dataset.RasterYSize  # 行数
        bands = dataset.RasterCount  # 波段

        data = np.zeros([nYSize, nXSize, bands], dtype=dtype)
        for i in range(bands):
            band = dataset.GetRasterBand(i + 1)
            data[:, :, i] = band.ReadAsArray(0, 0, nXSize, nYSize)  # .astype(np.complex)
        return data

    def expand_image(self,img):       #需要GDAL读入img
        stride = self.stride
        kernel = self.kernel
        """
        填充，在右方，下方填充0
        img: 输入图像,二维或三维矩阵int or float, as : img.shape=(512,512,3)
        kernel: 卷积核大小,矩阵int
        stride: 步长,int
        """
        expand_H = stride - (img.shape[0] - kernel[0]) % stride
        expand_W = stride - (img.shape[1] - kernel[1]) % stride
        if len(img.shape) == 3:
            img2 = img.shape[2]
            H_x = np.zeros((expand_H, img.shape[1], img2), dtype=img.dtype)
            W_x = np.zeros((img.shape[0] + expand_H, expand_W, img2), dtype=img.dtype)
        else:
            H_x = np.zeros((expand_H, img.shape[1]), dtype=img.dtype)
            W_x = np.zeros((img.shape[0] + expand_H, expand_W), dtype=img.dtype)
        img = np.r_[img, H_x]  # 行
        # img = np.c_[img,W_x]#列
        img = np.concatenate([img, W_x], axis=1)
        return img

    def cut_image(self,img):      #需要输入expand_image
        stride = self.stride
        kernel = self.kernel
        """"
        切片，将影像分成固定大小的块
        img     ---->输入图像,二维或三维矩阵int or float
        """
        H, W = [0, 0]
        a_append = []

        total_number_H = int((img.shape[0] - kernel[0]) / stride + 1)
        total_number_W = int((img.shape[1] - kernel[1]) / stride + 1)
        if len(img.shape) == 3:
            for H in range(total_number_H):  # H为高度方向切片数
                Hmin = H * stride
                Hmax = H * stride + kernel[0]

                for W in range(total_number_W):  # W为宽度方向切片数
                    Wmin = W * stride
                    Wmax = W * stride + kernel[1]
                    imgd = img[Hmin:Hmax, Wmin:Wmax, :]
                    a_append.append(imgd)
        else:
            for H in range(total_number_H):
                Hmin = H * stride
                Hmax = H * stride + kernel[0]

                for W in range(total_number_W):
                    Wmin = W * stride
                    Wmax = W * stride + kernel[1]
                    imgd = img[Hmin:Hmax, Wmin:Wmax]
                    a_append.append(imgd)
        if total_number_H * total_number_W == len(a_append):
            print('right')
        else:
            print('wrong')
        return a_append

    def make_dir(self,path):
        isExists = os.path.exists(path)
        # 判断结果
        if not isExists:
            os.makedirs(path)
            print(path + ' 创建成功')
            return True

    def save_img(self,img,name):      #输入cut_image的列表
        pathname = self.save_path
        """
        img: cut_image的列表
        """
        self.make_dir(pathname)

        # for i in range(len(a_append)):
        A = PIL.Image.fromarray(img)
        s = pathname + '\\' + name + '.png'
        A.save(s)#, quality=95)

    def main(self):
        self.search_files()
        count = 0
        k = 0
        for each_line in self.var1:
            self.var2 = self.read_IMG(each_line)
            self.var2 = self.cut_image(self.expand_image(self.var2))
            print('{}'.format(each_line))
            if k != 0:
                count = count + k + 1
            for k in range(len(self.var2)):
                num = count + k
                name0 = self.save_name + f'{num:04d}'
                self.save_img(np.squeeze(self.var2[k]),name=name0)
                print(name0)


class Img_Post:
    """
    预测影像合并并赋予坐标系
    """
    # 原始文件的长宽高，以及扩充后的长宽高
    @staticmethod
    def read_shape(path, dtype, kernel=[256, 256], stride=256):
        img = Img_pre(dtype=dtype).readtif_GDAL(path)
        r0, s0, w0 = img.shape
        print('img_shape = {} {} {}and dtype={}'.format(img.shape[0], img.shape[1], img.shape[2], img.dtype))
        img = Img_pre(stride=stride,kernel=kernel).expand_image(img)
        r1, s1, w1 = img.shape
        r1 = int((img.shape[0] - kernel[0]) / stride + 1)
        s1 = int((img.shape[1] - kernel[1]) / stride + 1)
        print('expand_img_shape = {} {} {}and dtype={}'.format(img.shape[0], img.shape[1], img.shape[2], img.dtype))

        return [r0, s0, w0, r1, s1, w1]

    @staticmethod
    def join_image(img, kernel=[512, 512], stride=512, H=17, W=10, S=0):
        """
        Parameters
        ----------
        img : 矩阵[img1,img2,img3]
        kernel : TYPE, optional
            DESCRIPTION. The default is [512,512].
        stride : TYPE, optional
            DESCRIPTION. The default is 512.
        H : 行方向数量
        W : 列方向数量
        S : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        zeros_np : 合成后的单个矩阵
            DESCRIPTION.

        """
        zeros_np = np.zeros(
            (H * kernel[0] - (H - 1) * (kernel[0] - stride), W * kernel[1] - (W - 1) * (kernel[1] - stride), S),
            dtype=img[0].dtype)
        num_index = 0
        if len(img[0].shape) == 3:
            for h in range(H):  # H为高度方向切片数
                Hmin = h * stride
                Hmax = h * stride + kernel[0]
                for w in range(W):  # W为宽度方向切片数
                    Wmin = w * stride
                    Wmax = w * stride + kernel[1]
                    zeros_np[Hmin:Hmax, Wmin:Wmax, :] = img[num_index]
                    print('Hmin:{},Hmax:{},Wmin:{},Wmax:{},合并目标：{}'.format(Hmin, Hmax, Wmin, Wmax, zeros_np.shape))
                    num_index += 1
        else:
            for h in range(H):  # H为高度方向切片数
                Hmin = h * stride
                Hmax = h * stride + kernel[0]
                for w in range(W):  # W为宽度方向切片数
                    Wmin = w * stride
                    Wmax = w * stride + kernel[1]
                    zeros_np[Hmin:Hmax, Wmin:Wmax] = img[num_index]
                    num_index += 1

        return zeros_np

    @staticmethod
    def save_image(img, pathname, name='road'):
        """保存,PIL支持uint8，但是有的图像压缩成uint8会出错"""
        def make_dir(path):
            isExists = os.path.exists(path)
            # 判断结果
            if not isExists:
                os.makedirs(path)
                print(path + ' 创建成功')
                return True
        make_dir(pathname)
        # for i in range(len(a_append)):
        A = PIL.Image.fromarray(img)
        s = pathname + '\\' + name + '.png'
        A.save(s, quality=95)

        # 投影坐标
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

    @classmethod
    def mosaic_img(cls,img_path, rsw, endwith='', img_size=[256, 256, 1], stride=256):
        img = []
        if endwith != '':
            if endwith == '.npy':
                for each_img in img_path:
                    img.append(np.load(each_img))
            if endwith == '.png' or endwith == '.jpg':
                for each_img in img_path:
                    image = PIL.Image.open(each_img)
                    image = np.asarray(image, dtype=np.uint8)
                    img.append(image.reshape(img_size))
            joint_img = cls.join_image(img, kernel=img_size[0:2], stride=stride, H=rsw[3], W=rsw[4],
                                             S=img_size[-1])  # rsw[5]
            joint_img = joint_img[0:rsw[0], 0:rsw[1], :]  # z左上角切片，扩充零在右下角
            return joint_img
        else:
            print('False/n')
            return 0

    @staticmethod
    def erode_image(bin_image,kernel_size):
        """
        erode bin image
        Args:
            bin_image: image with 0,1 pixel value
        Returns:
            erode image
        """
        # kernel = np.ones(shape=(kernel_size, kernel_size))

        if ((kernel_size % 2) == 0) or (kernel_size < 1):
            raise ValueError("kernel size must be odd and bigger than 1")
        # if (bin_image.max() != 1) or (bin_image.min() != 0):
        #     raise ValueError("input image's pixel value must be 0 or 1")
        d_image = np.zeros(shape=bin_image.shape)
        center_move = int((kernel_size - 1) / 2)
        for i in range(center_move, bin_image.shape[0] - kernel_size + 1):
            for j in range(center_move, bin_image.shape[1] - kernel_size + 1):
                d_image[i, j] = np.min(bin_image[i - center_move:i + center_move,
                                       j - center_move:j + center_move])
        return d_image

    @staticmethod
    def dilate_image(bin_image,kernel_size):
        """
        dilate bin image
        Args:
            bin_image: image as label
        Returns:
            dilate image
        """
        if (kernel_size % 2 == 0) or kernel_size < 1:
            raise ValueError("kernel size must be odd and bigger than 1")
        # if (bin_image.max() != 1) or (bin_image.min() != 0):
        #     raise ValueError("input image's pixel value must be 0 or 1")
        d_image = np.zeros(shape=bin_image.shape)
        center_move = int((kernel_size - 1) / 2)
        for i in range(center_move, bin_image.shape[0] - kernel_size + 1):
            for j in range(center_move, bin_image.shape[1] - kernel_size + 1):
                d_image[i, j] = np.max(bin_image[i - center_move:i + center_move, j - center_move:j + center_move])
        return d_image


    pass


