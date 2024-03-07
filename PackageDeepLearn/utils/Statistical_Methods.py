from tqdm import tqdm
from . import DataIOTrans
import numpy as np

def StatisticValues(ImgPaths):
    '''
    查看所有图像每个波段包含的数值以及数量
    ImgPaths: list
    '''
    pbar = tqdm(ImgPaths)
    for i, each in enumerate(pbar):
        array = DataIOTrans.DataIO.read_IMG(each)
        array = np.round(array, decimals=0)
        h, w, c = array.shape
        if i == 0:
            ImgDict = dict([(str(each), 0) for each in range(c)])
        # 单波段
        for each_band in range(c):
            unique, count = np.unique(array[:, :, each_band], return_counts=True)
            data_count = dict(zip(unique.astype(str), count))
            if i == 0:
                ImgDict[str(each_band)] = data_count
            else:
                keys = ImgDict[str(each_band)].keys()
                for key_, value_ in data_count.items():
                    if key_ in keys:
                        ImgDict[str(each_band)][key_] = ImgDict[str(each_band)][key_] + data_count[key_]
                    else:
                        ImgDict[str(each_band)].update({key_: value_})

        if i % 1000 == 0:
            print('len0={},len1={}'.format(len(ImgDict['0']), len(ImgDict['1'])))
    return ImgDict

def caculate_σ_μ_(ImgPaths, ignoreValue=65535):
    '''
    计算数据集每个通道的均值和标准差，以及最大值最小值
    ImgPaths: list, 包含所有需要计算的影像路径
    ignoreValue： 忽略的值
    '''
    σ_ALL = [];
    μ_ALL = [];

    pbar = tqdm(ImgPaths)
    for i, each in enumerate(pbar):
        array = DataIOTrans.DataIO.read_IMG(each)
        if len(array.shape) == 2:
            array = array[..., np.newaxis]
            print('Your dataset shape=2')
        array = array.astype(np.float32)
        array[array == ignoreValue] = np.nan

        if i == 0:
            h, w, c = array.shape
            array = array.reshape(h * w, c)
            arrayMax = array.max(axis=0)
            arrayMin = array.min(axis=0)
        else:
            array = array.reshape(h * w, c)
            max = array.max(axis=0)
            min = array.min(axis=0)

            arrayMax[max > arrayMax] = max[max > arrayMax]
            arrayMin[min < arrayMin] = min[min < arrayMin]
        if ignoreValue:
            # # 这里出了bug大矩阵无法适用
            # σ = np.nanmean(array, axis=axis)
            # μ = np.nanstd(array, axis=axis)
            σ = np.array([np.nanmean(array[:, each]) for each in range(c)])
            μ = np.array([np.nanstd(array[:, each]) for each in range(c)])

        σ_ALL.append(σ)
        μ_ALL.append(μ)
        pbar.set_description(
            '轮次 :{}/{}.max:{},min:{},σ: {},μ:{}'.format(i + 1, len(ImgPaths), arrayMax, arrayMin, σ, μ))

    σ = np.mean(np.array(σ_ALL), axis=0)
    μ = np.mean(np.array(μ_ALL), axis=0)
    return σ, μ

def Cal_HistBoundary(counts, y=100):
    '''
    counts：分布直方图计数
    y=300是百分比截断数
    '''
    Boundaryvalue = int(sum(counts) / y)
    countFront = 0
    countBack = 0

    for index, count in enumerate(counts):
        if countFront + count >= Boundaryvalue:
            indexFront = index  # 记录Index
            break
        else:
            countFront += count

    for count, index in zip(counts, range(len(counts))[::-1]):
        if countBack + count >= Boundaryvalue:
            indexBack = index  # 记录Index
            break
        else:
            countBack += count

    return {'countFront': countFront, 'indexFront': indexFront,
            'countBack': countBack, 'indexBack': indexBack}

def Cal_median(df, ColumnName):
    median_count = df[ColumnName].sum() / 2
    counts = 0
    for i, count in enumerate(df[ColumnName]):
        counts += count
        if counts >= median_count:
            print('中位数计算：band{},索引为{},最终计数{},累计计数{}'.format(ColumnName, i, count, counts))
            return i
    print('check your inter')

