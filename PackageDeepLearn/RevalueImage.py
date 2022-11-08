from PackageDeepLearn.utils import DataIOTrans,file_search_wash,Visualize
import numpy as np

ImagePath = file_search_wash.search_files_alldir(r'H:\sentinel-2_image_test\real_s2_concate',endwith='.tif')

def revalue_img(ImagePath):
    for each in ImagePath:

        img = DataIOTrans.DataIO.read_IMG(each)
        img[img==125] = 1
        img[img==255] = 2

        DataIOTrans.DataIO.save_Gdal(img,each.split('.')[0]+'.tif')

def chang_img_dtypeAndvalue(ImagePath):
    for each in ImagePath:
        img = DataIOTrans.DataIO.read_IMG(each)
        img = (img *10000).astype(np.uint16)
        DataIOTrans.DataIO.save_Gdal(img, each.split('.')[0] + '.tif')



#----------------------------------caculate value
B = DataIOTrans.DataIO.read_IMG(r'H:\sentinel-2_image_test\Cloud_free_R119_cl.tif')
B = B/10000
A = DataIOTrans.DataIO.read_IMG(r'H:\sentinel-2_image_test\real_s2_concate.tif')
cloudMaxDN = np.array([1.46879995, 1.47440004, 1.47360003])
Sub = A-B
alpha=Sub/(cloudMaxDN-B)
np.clip(alpha,0,1)
Sub[Sub<0.8]=0
Visualize.visualize(Sub=np.clip(alpha,0.2,1))