import os
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import fiona
from fiona.transform import transform_geom

def clip_and_reproject_CLCD(input_folder, shapefile_path, output_folder):
    """
    裁剪并重投影 CLCD 年度数据为 EPSG:4540（CGCS2000 / CM93E）
    自动识别年份决定原始坐标系（EPSG:4326 或 Albers）
    """
    os.makedirs(output_folder, exist_ok=True)

    dst_crs = "EPSG:4540"  # CGCS2000 / 3° Gauss-Kruger, CM 93E

    # 加载 shapefile 并投影为 EPSG:4326（因为 rasterio.mask 要求 geoms 与影像一致）
    with fiona.open(shapefile_path, "r") as shapefile:
        vector_crs = shapefile.crs
        if not vector_crs:
            raise ValueError("Shapefile 坐标系未定义，请确保存在 .prj 文件")
        geoms = [transform_geom(vector_crs, "EPSG:4326", feature["geometry"]) for feature in shapefile]

    for filename in os.listdir(input_folder):
        if filename.startswith("CLCD_v01_") and filename.endswith(".tif"):
            year_str = filename.replace("CLCD_v01_", "").replace(".tif", "")
            try:
                year = int(year_str)
            except ValueError:
                print(f"[跳过] 文件名年份解析失败: {filename}")
                continue
            if not (1990 <= year <= 2023):
                print(f"[跳过] 不在目标年份范围内: {filename}")
                continue

            # 设置原始 CRS
            if year <= 2020:
                src_crs = "EPSG:4326"
            else:
                # WGS 1984 Albers 没有 EPSG 编码，用 proj4 字符串定义
                src_crs = {
                    'proj': 'aea',
                    'lat_1': 25,
                    'lat_2': 47,
                    'lat_0': 0,
                    'lon_0': 105,
                    'x_0': 0,
                    'y_0': 0,
                    'datum': 'WGS84',
                    'units': 'm',
                    'no_defs': True
                }

            input_path = os.path.join(input_folder, filename)
            output_name = filename.replace(".tif", "_2000.tif")
            output_path = os.path.join(output_folder, output_name)

            with rasterio.open(input_path) as src:
                try:
                    # 将 shapefile 投影为 src_crs
                    local_geoms = [
                        transform_geom("EPSG:4326", src.crs, geom) for geom in geoms
                    ]

                    # 裁剪影像
                    clipped_image, clipped_transform = mask(src, local_geoms, crop=True)
                except Exception as e:
                    print(f"[跳过] {filename}: 裁剪失败（{e}）")
                    continue

                clipped_meta = src.meta.copy()
                clipped_meta.update({
                    "height": clipped_image.shape[1],
                    "width": clipped_image.shape[2],
                    "transform": clipped_transform
                })

                # 使用内存写入裁剪结果
                with rasterio.io.MemoryFile() as memfile:
                    with memfile.open(**clipped_meta) as clipped_src:
                        clipped_src.write(clipped_image)

                        # 重投影参数
                        transform, width, height = calculate_default_transform(
                            clipped_src.crs, dst_crs, clipped_src.width, clipped_src.height, *clipped_src.bounds)
                        dst_meta = clipped_meta.copy()
                        dst_meta.update({
                            'crs': dst_crs,
                            'transform': transform,
                            'width': width,
                            'height': height
                        })

                        with rasterio.open(output_path, 'w', **dst_meta) as dst:
                            for i in range(1, clipped_src.count + 1):
                                reproject(
                                    source=rasterio.band(clipped_src, i),
                                    destination=rasterio.band(dst, i),
                                    src_transform=clipped_src.transform,
                                    src_crs=clipped_src.crs,
                                    dst_transform=transform,
                                    dst_crs=dst_crs,
                                    resampling=Resampling.nearest
                                )

                print(f"[完成] {filename} → {output_name}")


if __name__ == "__main__":
    clip_and_reproject_CLCD(
        input_folder=r"D:\\数据\\CLCD中国土地覆盖数据集1985_2020年_30米",
        shapefile_path=r"D:\\Hydrogeology\\QGIS\\blk\\clip_pri.shp",
        output_folder=r"E:\\数据\\CLCD中国土地覆盖数据集1985_2020年_30米"
    )