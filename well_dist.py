import geopandas as gpd
from shapely.geometry import Point
from pyproj import Transformer
import pandas as pd


def calculate_distances_to_lake_center(well_coords, lake_shp_path):
    """
    参数:
        well_coords: 包含5口井经纬度的列表 [(lon1, lat1), (lon2, lat2), ...]
        lake_shp_path: 湖泊边界的shapefile路径，投影为 EPSG:4540

    返回:
        DataFrame，包含井编号、坐标、距离（米）
    """
    # 1. 读取湖泊边界 shp 文件（坐标系已为 EPSG:4540）
    lake_gdf = gpd.read_file(lake_shp_path)
    lake_gdf = lake_gdf.to_crs(epsg=4540)

    # 2. 计算湖泊的质心（中心点）
    lake_center = lake_gdf.unary_union.centroid

    # 3. 创建经纬度 -> EPSG:4540 的坐标转换器
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:4540", always_xy=True)

    # 4. 转换井坐标并计算距离
    records = []
    for i, (lon, lat) in enumerate(well_coords, start=1):
        x, y = transformer.transform(lon, lat)
        well_point = Point(x, y)
        distance = well_point.distance(lake_center)  # 单位为米（EPSG:4540 坐标系单位）
        records.append({
            'Well': f'Well{i}',
            'Longitude': lon,
            'Latitude': lat,
            'Distance_m': distance
        })

    return pd.DataFrame(records)

if __name__ == "__main__":
    wells = [
        (92.6232, 43.6522),
        (93.0931, 43.7478),
        (93.0395, 43.6441),
        (93.3101, 43.6115),
        (93.5143, 43.4932)
    ]

    lake_shp = "C:\\Users\\hanji\\Desktop\\tmp\\湖泊面积\\zzzz\\boundary.shp"

    df = calculate_distances_to_lake_center(wells, lake_shp)
    print(df)