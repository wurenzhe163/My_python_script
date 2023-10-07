import os.path

import matplotlib.pyplot as plt
import numpy as np
from .DataIOTrans import DataIO,make_dir
import cv2,torch
from torch.utils.tensorboard import SummaryWriter
import tensorboard
'''
可视化输出,包含打印字段
'''
def visualize(savepath=None,min_max=False,**images):
    """
    plt展示图像
    {Name: array，
    …………}
    """
    n_images = len(images)
    plt.figure(figsize=(20,8))
    for idx, (name, image) in enumerate(images.items()):
        if min_max:
            h, w, c = image.shape
            max_ = np.max(image.reshape(h * w, c), axis=0)
            min_ = np.min(image.reshape(h * w, c), axis=0)
            image = (image - min_) / (max_ - min_)
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([])
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_',' ').title(), fontsize=20)
        plt.imshow(image)

    if savepath:
        plt.savefig(savepath)
    else:
        plt.show()

def save_img(path,name,image,norm=False,histogram=False,
              img_transf=False, coordnates=None, img_proj=None):

    make_dir('{}'.format(path))
    SavePath = os.path.join(path,name)

    if norm:
        image = image * 255
        image = image.astype(np.uint8)
        if histogram:
            image = histogram_equalization_rgb(image)
    elif 'float' in str(image.dtype):
        image = image.astype(np.float32)

    if name.endswith('.tif'):
        DataIO.save_Gdal(image, SavePath)
        if img_transf:
            DataIO.save_Gdal(image, SavePath, img_transf = img_transf, coordnates = coordnates, img_proj = img_proj)
    else:
        cv2.imwrite(SavePath, image)  # cv2能够保存整型多波段，或者float多波段


def histogram_equalization_rgb(img):
    """对输入的RGB图像进行直方图均衡化处理"""
    # 将图像转换为HSV颜色空间
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # 对亮度通道进行直方图均衡化
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])

    # 将图像转换回BGR颜色空间
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return bgr
def plot_network(model,logdir='./',comment='My_Network',shape=(8,3,512,512)):
    """
    针对于单输入的模型可视化
    Args:
        model: 模型
        logdir: 保存文件夹
        comment: 绘图标题
        shape:  模型输入shape
    """
    x=torch.rand(shape)
    with SummaryWriter(log_dir=logdir,comment=comment) as w:
           w.add_graph(model, x)

    #tensorboard --logdir=./logs



