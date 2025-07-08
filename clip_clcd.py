import os
import re
import rasterio
from rasterio.mask import mask
from rasterio.features import geometry_mask
import geopandas as gpd
import numpy as np

def get_valid_nodata(dtype: str):
    """根据数据类型返回合法的 nodata 值"""
    if dtype == 'uint8':
        return 255
    elif dtype == 'int16':
        return -9999
    elif dtype == 'float32':
        return -9999.0
    else:
        raise ValueError(f"不支持的数据类型: {dtype}")

def clip_clcd_by_shapefile(input_folder, shapefile_path, output_folder):
    # 读取边界 shapefile
    shapefile = gpd.read_file(shapefile_path)
    os.makedirs(output_folder, exist_ok=True)

    # 匹配 CLCD 文件名
    pattern = re.compile(r"CLCD_v01_(\d{4})_2000\.tif")

    for file in os.listdir(input_folder):
        match = pattern.match(file)
        if not match:
            continue

        yyyy = match.group(1)
        output_filename = f"CLCD_BLK_{yyyy}.tif"
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, output_filename)

        with rasterio.open(input_path) as src:
            dtype = src.meta['dtype']
            nodata_value = get_valid_nodata(dtype)

            # 矢量坐标系转换
            shapes = shapefile.to_crs(src.crs) if shapefile.crs != src.crs else shapefile
            geoms = [feature["geometry"] for feature in shapes.__geo_interface__["features"]]

            # 裁剪（不填充 nodata）
            out_image, out_transform = mask(
                src, geoms, crop=True, filled=False, nodata=None
            )

            # 构建掩膜，仅保留 shape 内像素
            out_mask = geometry_mask(
                geoms,
                transform=out_transform,
                invert=True,
                out_shape=(out_image.shape[1], out_image.shape[2]),
                all_touched=False
            )

            out_image = np.where(out_mask[None, :, :], out_image, nodata_value)

            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "nodata": nodata_value
            })

            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image)

        print(f"✔ 已裁剪并输出: {output_filename}")

if __name__ == "__main__":
    input_folder = r"E:\\数据\\CLCD中国土地覆盖数据集1985_2020年_30米"
    shapefile_path = r"C:\\Users\\hanji\\Desktop\\tmp\\湖泊面积\\巴里坤范围2000\\巴里坤盆地范围.shp"
    output_folder = r"E:\\数据\\巴里坤耕地"

    clip_clcd_by_shapefile(input_folder, shapefile_path, output_folder)
