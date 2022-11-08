import matplotlib.pyplot as plt
import numpy as np
from .DataIOTrans import DataIO,make_dir
# import cv2

'''
可视化输出,包含打印字段
'''
def visualize(savepath=None,**images):
    """
    plt展示图像
    {Name: array，
    …………}
    """
    n_images = len(images)
    plt.figure(figsize=(20,8))
    for idx, (name, image) in enumerate(images.items()):
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

def save_img(path,index=0,norm=False,endwith='.tif',
             img_transf=False, coordnates=None, img_proj=None,**images):
    """
    Args:
        path: 保存路径
        index: 编号
        norm: 输入的是归一化图像放大至0-255，uint8
        endwith: 图像格式
        img_transf: 是否进行坐标变换
        coordnates: 仿射变换参数
        img_proj: 投影信息
        **images: name:array
    """

    for idx, (name, image) in enumerate(images.items()):
        make_dir('{}/{}'.format(path,name))
        SavePath = '{}/{}/{}{}'.format(path,name,f'{index:05d}',endwith)
        if norm: 
            image = image * 255
            image = image.astye(np.uint8)
        if 'float' in str(image.dtype):
            image = image.astype(np.float32)
        if endwith == '.tif':
            DataIO.save_Gdal(image, SavePath, img_transf=img_transf, coordnates=coordnates, img_proj=img_proj)
        else:
            cv2.imwrite(SavePath,image)
            # cv2能够保存整型多波段，或者float多波段



