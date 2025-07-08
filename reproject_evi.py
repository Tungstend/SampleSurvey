import os
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import fiona


def clip_and_reproject_tifs(input_folder, shapefile_path, output_folder):
    """
    批量裁剪并重投影 WGS84 tif 文件为 CGCS2000 / EPSG:4540（CM 93E）

    参数:
        input_folder (str): 包含 WGS84 tif 文件的文件夹
        shapefile_path (str): 用于裁剪的矢量边界文件（shp）
        output_folder (str): 输出裁剪并重投影后的 tif 的文件夹
    """
    os.makedirs(output_folder, exist_ok=True)

    src_crs = "EPSG:4326"
    dst_crs = "EPSG:4540"  # CGCS2000 / 3-degree Gauss-Kruger CM 93E

    # 加载裁剪边界
    with fiona.open(shapefile_path, "r") as shapefile:
        geoms = [feature["geometry"] for feature in shapefile]

    for filename in os.listdir(input_folder):
        if filename.startswith("EVI_china_") and filename.endswith(".tif"):
            input_path = os.path.join(input_folder, filename)
            output_name = filename.replace(".tif", "_2000.tif")
            output_path = os.path.join(output_folder, output_name)

            with rasterio.open(input_path) as src:
                # 裁剪影像
                clipped_image, clipped_transform = mask(src, geoms, crop=True)
                clipped_meta = src.meta.copy()
                clipped_meta.update({
                    "height": clipped_image.shape[1],
                    "width": clipped_image.shape[2],
                    "transform": clipped_transform
                })

                # 创建内存文件保存裁剪结果
                with rasterio.io.MemoryFile() as memfile:
                    with memfile.open(**clipped_meta) as clipped_src:
                        clipped_src.write(clipped_image)

                        # 计算投影变换
                        transform, width, height = calculate_default_transform(
                            src_crs, dst_crs, clipped_src.width, clipped_src.height, *clipped_src.bounds)
                        dst_meta = clipped_meta.copy()
                        dst_meta.update({
                            'crs': dst_crs,
                            'transform': transform,
                            'width': width,
                            'height': height
                        })

                        # 写入重投影文件
                        with rasterio.open(output_path, 'w', **dst_meta) as dst:
                            for i in range(1, clipped_src.count + 1):
                                reproject(
                                    source=rasterio.band(clipped_src, i),
                                    destination=rasterio.band(dst, i),
                                    src_transform=clipped_src.transform,
                                    src_crs=src_crs,
                                    dst_transform=transform,
                                    dst_crs=dst_crs,
                                    resampling=Resampling.nearest
                                )
            print(f"处理完成: {filename} → {output_name}")


if __name__ == "__main__":
    clip_and_reproject_tifs(
        input_folder=r"D:\\数据\\中国EVI月均值产品数据MOD13Q1",
        shapefile_path=r"D:\\Hydrogeology\\QGIS\\blk\\clip_pri.shp",
        output_folder=r"E:\\数据\\中国EVI月均值产品数据MOD13Q1"
    )