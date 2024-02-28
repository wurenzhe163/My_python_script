import numpy as np
import argparse
from tqdm import tqdm
import yaml
from attrdict import AttrMap

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data_manager import TestDataset
from utils import gpu_manage, save_image, heatmap
from models.gen.SPANet import Generator
import matplotlib.pyplot as plt
from osgeo import gdal

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



def save_Gdal(img_array, SavePath, img_transf=False, coordnates=None, img_proj=None):
    """

    Args:
        img_array:  [H,W,C] , RGB色彩，不限通道深度
        SavePath:
        img_transf: 是否进行投影
        coordnates: 仿射变换
        img_proj: 投影信息

    Returns: 0

    """

    # 判断数据类型
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

    if img_transf:
        dataset.SetGeoTransform(coordnates)  # 写入仿射变换参数
        dataset.SetProjection(img_proj)  # 写入投影

    # 写入影像数据
    if len(img_array.shape) == 2:
        dataset.GetRasterBand(1).WriteArray(img_array)
    else:
        for i in range(img_bands):
            dataset.GetRasterBand(i + 1).WriteArray(img_array[:, :, i])

    dataset = None
    
    
def predict(config, args):
    gpu_manage(args)
    dataset = TestDataset(args.test_dir, config.in_ch, config.out_ch)
    data_loader = DataLoader(dataset=dataset, num_workers=config.threads, batch_size=1, shuffle=False)

    ### MODELS LOAD ###
    print('===> Loading models')

    gen = Generator(gpu_ids=config.gpu_ids)

    param = torch.load(args.pretrained)
    gen.load_state_dict(param)

    if args.cuda:
        gen = gen.cuda(0)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
            x = Variable(batch[0])
            filename = batch[1][0]
            if args.cuda:
                x = x.cuda()
            
            att, out = gen(x)
            decode_x   = torch.squeeze(x.permute(2,3,1,0)).cpu().numpy()
            decode_out = torch.squeeze(out.permute(2,3,1,0)).cpu().numpy()
            decode_att = torch.squeeze(att).cpu().numpy()

            save_Gdal(decode_out,'{}/{}/{}'.format(args.out_dir,'out',f'{i:04d}.tif'))
            save_Gdal(decode_att,'{}/{}/{}'.format(args.out_dir,'att',f'{i:04d}.tif'))
            visualize(savepath='{}/{}/{}'.format(args.out_dir,'fig',f'{i:04d}.png'),
                      x=decode_x,out=decode_out,att=decode_att)

            """
                        h = 1
                        w = 3
                        c = 3
                        p = config.width
            
                        allim = np.zeros((h, w, c, p, p))
                        x_ = x.cpu().numpy()[0]
                        out_ = out.cpu().numpy()[0]
                        in_rgb = x_[:3]
                        out_rgb = np.clip(out_[:3], 0, 1)
                        att_ = att.cpu().numpy()[0] * 255
                        heat_att = heatmap(att_.astype('uint8'))
                        
                        allim[0, 0, :] = in_rgb * 255
                        allim[0, 1, :] = out_rgb * 255
                        allim[0, 2, :] = heat_att
                        allim = allim.transpose(0, 3, 1, 4, 2)
                        allim = allim.reshape((h*p, w*p, c))
                        
                        allim = allim.astype(np.float32)
            
                        save_image(args.out_dir, allim, i, 1, filename=filename)
            """

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predoction')
    parser.add_argument('--config', default=r'D:/Wrz/SpA-GAN_for_cloud_removal-master/pretrained_models/RICE1/config.yml',
                        type=str, required=False)
    parser.add_argument('--test_dir',default=r'D:\Wrz\Dataset\test\train_image'
                        , type=str, required=False)
    parser.add_argument('--out_dir', default=r'D:\Wrz\Dataset\SpA-GAN\fitune_Dehaze'
                        ,type=str, required=False)
    parser.add_argument('--pretrained', default=r'D:/Wrz/SpA-GAN_for_cloud_removal-master/results/000001/models/gen_model_epoch_100.pth',
                        type=str, required=False)
    parser.add_argument('--cuda', default=True,action='store_true')
    parser.add_argument('--gpu_ids', type=int, default=[0])#
    parser.add_argument('--manualSeed', type=int, default=0)
    parser.add_argument("--mode", default='client')
    parser.add_argument("--port", default=52162)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='UTF-8') as f:
        config = yaml.safe_load(f)
    config = AttrMap(config)

    predict(config, args)
