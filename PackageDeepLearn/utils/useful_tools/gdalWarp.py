import os
import argparse
from osgeo import gdal

def search_files(path, endwith='.tif'):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(endwith)]

def check_projection(input_group):
    projections = []
    for file_path in input_group:
        dataset = gdal.Open(file_path)
        projections.append(dataset.GetProjection())
        dataset = None
    return projections

def process_images(input_folder, to_epsg_4326=False):
    os.chdir(input_folder)
    input_group = search_files(path=input_folder, endwith='.tif')

    projections = check_projection(input_group)


    output_vrt = 'temp.vrt'
    output_tif = 'merge.tif'

    if to_epsg_4326:
        aligned_inputs = []
        for img in input_group:
            aligned_img = img.replace('.tif', '_aligned.tif')
            gdal.Warp(aligned_img, img, dstSRS='EPSG:4326')  # 转换至 WGS84 投影
            aligned_inputs.append(aligned_img)
    else:
        aligned_inputs = input_group

    # 创建虚拟数据集
    gdal.BuildVRT(output_vrt, aligned_inputs)
    # 使用期望的重采样算法进行图像拼接
    gdal.Warp(output_tif, output_vrt, resampleAlg='bilinear') #'average', 'nearest', 'bilinear', 'cubic', 'cubicspline', 'lanczos', 'average_magphase', 'mode'

    os.remove(output_vrt)
    if to_epsg_4326:
        for img in aligned_inputs:
            os.remove(img)


def main():
    parser = argparse.ArgumentParser(description='Merge TIFF images in a folder into a single TIFF.')
    parser.add_argument('input_folder', type=str, help='Path to the folder containing the TIFF images.')
    parser.add_argument('-t', '--to_epsg_4326', action='store_true', help='Transform image projection to EPSG:4326, default=false')

    args = parser.parse_args()

    process_images(args.input_folder, to_epsg_4326=args.to_epsg_4326)

if __name__ == '__main__':
    main()
