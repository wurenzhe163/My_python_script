# import cv2
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from PackageDeepLearn.utils.DataIOTrans import DataIO
#
# img=DataIO.read_IMG(r"C:\Users\11733\Desktop\ArcgisPro\project_1\导出\images\000000000.tif")
# plt.imshow(img)
#
# currentAxis=plt.gca()
# rect=patches.Rectangle((100, 50),200,100,linewidth=1,edgecolor='red',facecolor='none')#,linestyle='dotted'
# currentAxis.add_patch(rect)
# plt.show()
#
# # if __name__ == '__main__':
#     from torchvision import transforms
#
#     # load image
#     original_img =
#
#     # from pil image to tensor, do not normalize image
#     data_transform = transforms.Compose([transforms.ToTensor()])
#     img = data_transform(original_img)
#     draw = ImageDraw.Draw(original_img)
#
#
#     # draw_box(original_img,
#     #          predict_boxes,
#     #          predict_classes,
#     #          predict_scores,
#     #          category_index,
#     #          thresh=0.5,
#     #          line_thickness=3)
import os
import argparse
import xml.etree.cElementTree as ET
import random
from PIL import Image, ImageDraw, ImageFont
from PackageDeepLearn.utils.DataIOTrans import DataIO

FONT_SIZE = 13 * 2
WIDTH = 2
imgEndwith = '.tif'
SaveImgEndwith = '.png'  # 目前仅能够输出PNG
IMAGE_FONT = ImageFont.truetype(u"simhei.ttf", FONT_SIZE)
# COLOR_LIST = ["red", "green", "blue", "purple"]

COLOR_LIST = ["red", "green", "blue", "cyan", "yellow", "purple",
              "deeppink", "ghostwhite", "darkcyan", "olive",
              "orange", "orangered", "darkgreen"]
category = {'1':'轨道','2':'公路','3':'土路','4':'桥梁','5':'车站',
            '6':'广场','7':'隧道口','8':'站台','9':'电网','10':'列车'}

search_files = lambda path,endwith: sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(endwith)])
class ShowResult(object):
    def __init__(self, args , category):
        self.args = args
        self.annotation_path = self.args.annotation_path
        self.save_picture_path = self.args.save_picture_path
        self.origin_img_path = self.args.origin_img_path

        # 保存原有类别
        self.category = category#{'echinus': '海胆', 'holothurian': '海参', 'starfish': '海星', 'scallop': '扇贝'}

    def category_id_convert_to_name(self, category):
        return self.category[category]

    def GetAnnotBoxLoc(self, AnotPath):
        tree = ET.ElementTree(file=AnotPath)
        root = tree.getroot()
        ObjectSet = root.findall('object')
        ObjBndBoxSet = {}
        for Object in ObjectSet:
            ObjName = Object.find('name').text
            BndBox = Object.find('bndbox')
            x1 = int(float(BndBox.find('xmin').text))
            y1 = int(float(BndBox.find('ymin').text))
            x2 = int(float(BndBox.find('xmax').text))
            y2 = int(float(BndBox.find('ymax').text))
            BndBoxLoc = [x1, y1, x2, y2]
            if ObjName in ObjBndBoxSet:
                ObjBndBoxSet[ObjName].append(BndBoxLoc)
            else:
                ObjBndBoxSet[ObjName] = [BndBoxLoc]
        return ObjBndBoxSet

    def GetAnnotName(self, AnotPath):
        tree = ET.ElementTree(file=AnotPath)
        root = tree.getroot()
        path = root.find('path').text
        return path

    def Drawpic(self):
        if not os.path.exists(self.save_picture_path):
            os.mkdir(self.save_picture_path)

        xml_list = search_files(self.annotation_path,'.xml')
        # 读取每一个xml文件
        for idx, xml_file in enumerate(xml_list):

            # box是一个字典，以类别为key，box为值
            box = self.GetAnnotBoxLoc(xml_file)
            img_name = str(xml_file.split('\\')[-1]).replace(".xml", imgEndwith)

            img_path = os.path.join(self.origin_img_path, img_name)
            print("当前正在处理第-{0}-张图片, 总共需要处理-{1}-张, 完成百分比:{2:.2%}".format(idx + 1,
                                                                        len(xml_list),
                                                                        (idx + 1) / len(xml_list)))
            # 对每一个bbox标注
            img = DataIO.read_IMG(img_path)
            img = Image.fromarray(img)
            # img = Image.open(img_path, "r")  # img1.size返回的宽度和高度(像素表示)
            draw = ImageDraw.Draw(img)

            for classes in list(box.keys()):
                COLOR = random.choice(COLOR_LIST)
                category_name = self.category_id_convert_to_name(classes)
                for boxes in box[classes]:
                    x_left = int(boxes[0])
                    y_top = int(boxes[1])
                    x_right = int(boxes[2])
                    y_down = int(boxes[3])

                    top_left = (int(boxes[0]), int(boxes[1]))  # x1,y1
                    top_right = (int(boxes[2]), int(boxes[1]))  # x2,y1
                    down_right = (int(boxes[2]), int(boxes[3]))  # x2,y2
                    down_left = (int(boxes[0]), int(boxes[3]))  # x1,y2

                    draw.line([top_left, top_right, down_right, down_left, top_left], width=8, fill=COLOR)
                    draw.text((x_left + 30, y_top - FONT_SIZE), str(category_name), font=IMAGE_FONT, fill=COLOR)
            # 存储图片
            img_Savename = str(xml_file.split('\\')[-1]).replace(".xml", SaveImgEndwith)
            save_path = os.path.join(self.save_picture_path, img_Savename)

            img.save(save_path, "png")


def parse():
    parser = argparse.ArgumentParser()
    # annotation_path
    parser.add_argument('-annotation_path', default=r"C:\Users\11733\Desktop\ArcgisPro\project_1\导出\内江北站\labels",
                        help='the single img json file path')
    # save_picture_path
    parser.add_argument('-save_picture_path', default=r"C:\Users\11733\Desktop\ArcgisPro\project_1\导出\内江北站\save",
                        help='the val img result json file path')
    # origin_img_path
    parser.add_argument('-origin_img_path', default=r'C:\Users\11733\Desktop\ArcgisPro\project_1\导出\内江北站\images',
                        help='the val img path root')
    parser.add_argument("--port", default=52162)
    parser.add_argument("--mode", default='client')
    return parser.parse_args()



if __name__ == '__main__':
    args = parse()
    showresult = ShowResult(args,category)
    showresult.Drawpic()
