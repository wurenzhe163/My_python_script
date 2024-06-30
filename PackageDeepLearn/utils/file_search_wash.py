# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 19:20:10 2020

@author: 11733
"""
import os
import numpy as np

def delList(L):
    """"
    删除重复元素
    """
    L1 = []
    for i in L:
        if i not in L1:
            L1.append(i)
    return L1

# search_files = lambda path,endwith='.tif': [os.path.join(path,f) for f in os.listdir(path) if f.endswith(endwith) ]
def search_files(path,endwith='.tif'):
    """
    返回当前文件夹下文件
    Parameters
    ----------
    path : 路径
    endwith : The default is '.tif'.
    Returns: s ,   列表
    """
    s = []
    for f in os.listdir(path):
        if f.endswith(endwith):
            s.append(os.path.join(path,f))
    return s

def paixu(str_in,key=lambda info: int(info.split('_')[-1].split('.')[0])):
    test = sorted(str_in,key=key)
    return test

def search_files_alldir(path, endwith='.tif', return_type='relative'):
    """
    遍历文件夹下所有文件（包括子文件夹），若只需当前文件夹下文件使用search_files

    参数:
    - path: 要搜索的文件夹路径
    - endwith: 文件后缀名（默认是'.tif'）
    - return_type: 返回值类型，'absolute'表示返回绝对路径，'relative'表示返回文件夹名与文件名

    返回:
    - 包含所有符合条件的文件的列表，根据 return_type 参数返回绝对路径或有序字典形式的文件夹名与文件名
    """
    all_files = os.walk(path)  # os.walk遍历所有文件
    if return_type == 'absolute':
        s = []
    elif return_type == 'relative':
        s = OrderedDict()

    for dirpath, dirnames, filenames in all_files:
        for each_file in filenames:
            if each_file.endswith(endwith):
                if return_type == 'absolute':
                    s.append(os.path.join(dirpath, each_file))
                elif return_type == 'relative':
                    if dirpath not in s:
                        s[dirpath] = []
                    s[dirpath].append(each_file)

    if return_type == 'absolute':
        print('文件数=%d' % len(s))
    elif return_type == 'relative':
        file_count = sum(len(files) for files in s.values())
        print('文件数=%d' % file_count)
        for folder, files in s.items():
            print(f"{folder}: {len(files)} files")
        
    return s


def filter_(img_array,label_array):
    """"
    根据label过滤img
    img_array-------->影像组
    label_array------>标签组
    """
    new_labelarray=[]
    new_imgarray=[]
    for i in range(len(label_array)):
        label_sum = np.sum(label_array[i])
        if label_sum !=0:
            new_labelarray.append(label_array[i])
            new_imgarray.append(img_array[i])
    return new_imgarray,new_labelarray
