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

def clip_tifs_strict_mask(input_folder, shapefile_path, output_folder):
    # 读取矢量边界
    shapefile = gpd.read_file(shapefile_path)

    # 创建输出目录
    os.makedirs(output_folder, exist_ok=True)

    # 文件名格式匹配
    pattern = re.compile(r"EVI_china_(\d{6})_2000\.tif")

    for file in os.listdir(input_folder):
        match = pattern.match(file)
        if not match:
            continue

        yyyymm = match.group(1)
        yyyy, mm = yyyymm[:4], yyyymm[4:]
        output_filename = f"EVI_BLK_{yyyy}_{mm}.tif"
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, output_filename)

        with rasterio.open(input_path) as src:
            dtype = src.meta['dtype']
            nodata_value = get_valid_nodata(dtype)

            # 坐标系统一
            shapes = shapefile.to_crs(src.crs) if shapefile.crs != src.crs else shapefile
            geoms = [feature["geometry"] for feature in shapes.__geo_interface__["features"]]

            # 裁剪图像（不填充 nodata）
            out_image, out_transform = mask(
                src, geoms, crop=True, filled=False, nodata=None
            )

            # 构建掩膜（False 表示 shape 外，True 表示 shape 内）
            out_mask = geometry_mask(
                geoms,
                transform=out_transform,
                invert=True,  # True 表示 shape 内部为 True
                out_shape=(out_image.shape[1], out_image.shape[2]),
                all_touched=False
            )

            # 掩膜 shape 外的像素为 nodata
            out_image = np.where(out_mask[None, :, :], out_image, nodata_value)

            # 更新元数据
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "nodata": nodata_value
            })

            # 写入裁剪结果
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image)

        print(f"✔ 已裁剪并输出: {output_filename}")


if __name__ == "__main__":
    input_folder = r"E:\\数据\\中国EVI月均值产品数据MOD13Q1"
    shapefile_path = r"C:\\Users\\hanji\\Desktop\\tmp\\湖泊面积\\巴里坤范围2000\\巴里坤盆地范围.shp"
    output_folder = r"E:\\数据\\巴里坤EVI"

    clip_tifs_strict_mask(input_folder, shapefile_path, output_folder)