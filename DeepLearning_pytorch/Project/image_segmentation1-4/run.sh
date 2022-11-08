    parser = argparse.ArgumentParser(description='1-4目标预测输出')
    parser.add_argument('--ckpt', type=str, metavar='', default=r'E:\Raiway_Demo\dataset_segmentation1-4\save_dir\model\0058model_obj.pth', help='预训练权重')
    parser.add_argument('--pre_img_dir', type=str, metavar='',default=False, help='图像切片文件夹')
    parser.add_argument('--pre_img', type=str, metavar='', default=r'E:\Raiway_Demo\dataset_segmentation1-4\image\0001.tif', help='整张图像路径(任意大小)')
    parser.add_argument('--output', type=str, metavar='',default=r'E:\Raiway_Demo\dataset_segmentation1-4\save_dir\pre', help='图像输出路径')
    parser.add_argument('--kernel', type=int, default=[512,512],nargs='+', metavar='', help='裁剪图像大小')
    parser.add_argument('--stride', type=int,default=256, metavar='', help='裁剪步长')
    parser.add_argument("--port", default=52162)
    parser.add_argument("--mode", default='client')

python ./modelpre.py --ckpt E:\Raiway_Demo\dataset_segmentation1-4\save_dir\model\0058model_obj.pth --pre_img 