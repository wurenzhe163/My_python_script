import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import explain_validity
import pandas as pd
from tqdm import tqdm

# 读取 shapefile
shapefile_path = r'D:\BaiduSyncdisk\02_论文相关\在写\SAM冰湖\数据\2023_05_31_to_2023_09_15_样本.shp'
gdf = gpd.read_file(shapefile_path)

# 创建一个 DataFrame 来存储所有的拓扑错误
topology_errors = []

# 检查几何对象是否有效
for idx, row in tqdm(gdf.iterrows(), desc='Processing gdf', total=len(gdf)):
    geom = row.geometry
    
    # 检查是否存在自相交拓扑错误
    if not geom.is_valid:
        topology_errors.append({
            'Index': idx,
            'Error Type': '自相交',
            'Validity Reason': explain_validity(geom)
        })
        
    # 检查重叠错误（仅适用于多边形和多重多边形）
    if geom.geom_type in ['Polygon', 'MultiPolygon']:
        for other_idx, other_row in gdf.iterrows():
            if idx != other_idx and geom.intersects(other_row.geometry):
                topology_errors.append({
                    'Index': idx,
                    'Error Type': '重叠',
                    'Overlap With': other_idx
                })
    
    # 检查空洞错误
    if geom.geom_type == 'Polygon' and len(geom.interiors) > 0:
        for interior in geom.interiors:
            if not Polygon(interior).is_valid:
                topology_errors.append({
                    'Index': idx,
                    'Error Type': '空洞',
                    'Validity Reason': explain_validity(Polygon(interior))
                })

# 打印所有拓扑错误
if topology_errors:
    errors_df = pd.DataFrame(topology_errors)
    print("Topology errors found:")
    print(errors_df)
else:
    print("No topology errors found.")
