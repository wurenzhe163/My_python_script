import geopandas as gpd
from tqdm import tqdm

def add_field(shp_path, add_dict):
	'''
	向矢量文件添加属性
	'''
    gdf = gpd.read_file(shp_path)
    gdf = gdf.assign(**add_dict)
    gdf.to_file(shp_path, driver='ESRI Shapefile', mode='w', encoding='GBK')
    
def detect_spatial_relationships(gdf, label_gdf):
    """
    检测两个 GeoDataFrame 之间的空间关系，并返回结果 GeoDataFrame。
    
    参数:
    - gdf: GeoDataFrame，原始的地理数据
    - label_gdf: GeoDataFrame，用于与 gdf 比较的地理数据
    
    返回:
    - result_gdf: 包含空间关系结果的 GeoDataFrame
    """
    # 创建新的 DataFrame 用于存储结果
    result_rows = []
    
    # 创建一个集合，用于存储 gdf 中完全重合的几何对象索引
    gdf_full_overlap_indices = set()

    # 遍历 gdf
    for idx_gdf, row_gdf in tqdm(gdf.iterrows(), desc='Processing gdf', total=len(gdf)):
        intersected = False
        for idx_label, row_label in label_gdf.iterrows():
            # 完全重合
            if row_gdf.geometry.equals(row_label.geometry):
                gdf_full_overlap_indices.add(idx_gdf)
                intersected = True
                break
            # 空间相交但不完全一致
            elif row_gdf.geometry.intersects(row_label.geometry):
                new_row = row_label.copy()
                new_row['关系'] = '空间相交不相同'
                result_rows.append(new_row)
                intersected = True
        # 没有空间相交的情况
        if not intersected:
            new_row = row_gdf.copy()
            new_row['关系'] = '空间不相交-Origin'  # 数据删除
            result_rows.append(new_row)

    # 检查 label_gdf 中的几何对象是否在 gdf 中
    for idx_label, row_label in tqdm(label_gdf.iterrows(), desc='Processing label_gdf', total=len(label_gdf)):
        if not any(row_label.geometry.intersects(row_gdf.geometry) for idx_gdf, row_gdf in gdf.iterrows()):
            new_row = row_label.copy()
            new_row['关系'] = '空间不相交-Label'
            result_rows.append(new_row)
        
    # 添加 gdf 中完全重合的几何对象到 result_gdf，并设置关系为 '空间一致'
    for idx in tqdm(gdf_full_overlap_indices, desc='Processing full overlaps'):
        row = gdf.loc[idx].copy()
        row['关系'] = '空间一致'
        result_rows.append(row)

    # 将结果转换为 GeoDataFrame
    result_gdf = gpd.GeoDataFrame(result_rows, columns=gdf.columns.tolist() + ['关系'], crs=gdf.crs)

    return result_gdf
    
