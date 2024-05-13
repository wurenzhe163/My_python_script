import geopandas as gpd

def add_field(shp_path, add_dict):
	'''
	向矢量文件添加属性
	'''
    gdf = gpd.read_file(shp_path)
    gdf = gdf.assign(**add_dict)
    gdf.to_file(shp_path, driver='ESRI Shapefile', mode='w', encoding='GBK')