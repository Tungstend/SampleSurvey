from rasterio.features import rasterize
import rasterio
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, LineString

def ensure_polygon(gdf):
    new_geoms = []
    for geom in gdf.geometry:
        if geom.geom_type == 'Polygon':
            new_geoms.append(geom)
        elif geom.geom_type == 'LineString':
            if not geom.is_ring:
                geom = LineString(list(geom.coords) + [geom.coords[0]])
            poly = Polygon(geom)
            new_geoms.append(poly)
        else:
            raise ValueError("不支持的几何类型: " + geom.geom_type)
    gdf.geometry = new_geoms
    return gdf

def mask_by_rasterization(tif_path, shp_path, out_tif_path):
    gdf = gpd.read_file(shp_path)
    gdf = ensure_polygon(gdf)

    with rasterio.open(tif_path) as src:
        profile = src.profile
        data = src.read(1)
        transform = src.transform

        # 将所有 geometry 转换为 (geometry, value) 的元组列表
        shapes = [(geom, 1) for geom in gdf.geometry]

        # rasterize 所有多边形
        mask = rasterize(
            shapes=shapes,
            out_shape=(src.height, src.width),
            transform=transform,
            fill=0,
            all_touched=True,  # 关键参数：只要碰边就算在内
            dtype='uint8'
        )

        # 设置 nodata（若未设）
        if profile['nodata'] is None:
            profile['nodata'] = -9999
        data[mask == 1] = profile['nodata']

        # 写出结果
        with rasterio.open(out_tif_path, 'w', **profile) as dst:
            dst.write(data, 1)


if __name__ == "__main__":
    mask_by_rasterization(
        tif_path='C:\\Users\\hanji\\Desktop\\tmp\\湖泊面积\\E_BLK_DEM30m_2000.tif',
        shp_path='C:\\Users\\hanji\\Desktop\\tmp\\湖泊面积\\zzzz\\boundary.shp',
        out_tif_path='C:\\Users\\hanji\\Desktop\\tmp\\湖泊面积\\E_BLK_DEM30m_exclude.tif'
    )
