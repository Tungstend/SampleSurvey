import geopandas as gpd
import os

def reproject_shapefile(input_shp_path, output_shp_path, target_epsg=4540):
    # 读取原始shp文件（默认带有.prj信息）
    gdf = gpd.read_file(input_shp_path)

    # 显示原始坐标系
    print(f"原始坐标系: {gdf.crs}")

    # 明确设置源坐标系为WGS84（若读取失败）
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.set_crs(epsg=4326)

    # 投影转换
    gdf_proj = gdf.to_crs(epsg=target_epsg)

    # 创建输出文件夹
    os.makedirs(os.path.dirname(output_shp_path), exist_ok=True)

    # 保存投影后的shapefile（包含所有shp附属文件）
    gdf_proj.to_file(output_shp_path, encoding='utf-8')

    print(f"已保存为：{output_shp_path}")

if __name__ == "__main__":
    # 输入原始shp文件路径（.shp 文件路径即可，其它附属文件自动处理）
    input_path = "C:\\Users\\hanji\\Desktop\\tmp\\湖泊面积\\巴里坤范围ARCGIS\\巴里坤盆地范围wgs84.shp"

    # 输出目标路径（.shp 文件路径，文件夹会自动创建）
    output_path = "C:\\Users\\hanji\\Desktop\\tmp\\湖泊面积\\巴里坤范围2000\\巴里坤盆地范围.shp"

    # 调用函数
    reproject_shapefile(input_path, output_path)