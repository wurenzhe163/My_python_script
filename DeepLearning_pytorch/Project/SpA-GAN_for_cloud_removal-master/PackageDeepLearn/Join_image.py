from PackageDeepLearn import ImageAfterTreatment
from PackageDeepLearn.utils import file_search_wash,DataIOTrans,Visualize
Img_Post =ImageAfterTreatment.Img_Post()
files = file_search_wash.search_files(r'C:\Users\Administrator\Desktop\SaveDir\real_sentinel\SPA-GAN')
ImageList = []
for eachpath in files:
    ImageList.append(Img_Post.read_IMG(eachpath))

#a_append,total_number_H,total_number_W = Img_Post.cut_image(Img_Post.read_IMG(r'H:\sentinel-2_image_test\real_s2_concate.tif'),kernel=[512, 512],stride=412)
Output = Img_Post.join_image2(ImageList, kernel=[512, 512], stride=412, H=26, W=26, S=3)

Visualize.save_img(r'C:\Users\Administrator\Desktop\SaveDir\real_sentinel\SPA-GAN',index=0,norm=False,endwith='.tif',Output=Output)