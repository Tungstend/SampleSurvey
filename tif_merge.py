import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import from_origin
from skimage.transform import resize
import numpy as np
import math


def align_bounds_to_resolution(bounds, res):
    """将边界向外扩展并对齐到指定分辨率网格"""
    xmin = math.floor(bounds[0] / res) * res
    ymin = math.floor(bounds[1] / res) * res
    xmax = math.ceil(bounds[2] / res) * res
    ymax = math.ceil(bounds[3] / res) * res
    return (xmin, ymin, xmax, ymax)


def extract_mask_and_cut(large_tif_path, small_tif_path,
                         output_large_tif_path, output_small_tif_path):
    # === 1. 打开小图（1m，CGCS2000） ===
    with rasterio.open(small_tif_path) as src_small:
        small_data = src_small.read(1)
        small_crs = src_small.crs
        small_transform = src_small.transform
        small_bounds = src_small.bounds
        small_profile = src_small.profile
        small_res = small_transform.a
        small_nodata = src_small.nodata if src_small.nodata is not None else -9999

    # === 2. 打开大图（30m，CGCS2000） ===
    with rasterio.open(large_tif_path) as src_large:
        large_data = src_large.read(1)
        large_transform = src_large.transform
        large_profile = src_large.profile
        large_res = large_transform.a

    # === 3. 将小图边界向外扩展到与大图像素对齐 ===
    aligned_bounds = align_bounds_to_resolution(small_bounds, large_res)
    aligned_left, aligned_bottom, aligned_right, aligned_top = aligned_bounds

    # === 4. 提取大图中覆盖区域 ===
    window = from_bounds(*aligned_bounds, transform=large_transform)
    window = window.round_offsets().round_lengths()

    row_off = int(window.row_off)
    col_off = int(window.col_off)
    nrows = int(window.height)
    ncols = int(window.width)

    large_patch = large_data[row_off:row_off + nrows, col_off:col_off + ncols]

    # === 5. 上采样大图 patch 到 1m 分辨率，确保精确匹配形状 ===
    new_width = int((aligned_right - aligned_left) / small_res)
    new_height = int((aligned_top - aligned_bottom) / small_res)
    upsampled_large = resize(
        large_patch,
        output_shape=(new_height, new_width),
        order=0,
        preserve_range=True,
        anti_aliasing=False
    ).astype('float32')

    # === 6. 构造补齐后的小图数组 ===
    expanded_small = np.full((new_height, new_width), np.nan, dtype='float32')

    row_shift = int((aligned_top - small_bounds[3]) / small_res)
    col_shift = int((small_bounds[0] - aligned_left) / small_res)

    expanded_small[row_shift:row_shift + small_data.shape[0],
                   col_shift:col_shift + small_data.shape[1]] = small_data

    # === 7. 替换无效像素为上采样大图值 ===
    invalid_mask = (expanded_small == small_nodata) | np.isnan(expanded_small)
    expanded_small[invalid_mask] = upsampled_large[invalid_mask]

    # === 8. 将该区域从大图中挖空 ===
    large_data[row_off:row_off + nrows, col_off:col_off + ncols] = np.nan

    # === 9. 保存输出 ===
    # 9.1 保存挖空后的大图
    large_profile.update({
        'dtype': 'float32',
        'nodata': np.nan
    })
    with rasterio.open(output_large_tif_path, 'w', **large_profile) as dst:
        dst.write(large_data.astype('float32'), 1)

    # 9.2 保存补齐后的小图
    new_transform = from_origin(aligned_left, aligned_top, small_res, small_res)
    small_profile.update({
        'height': new_height,
        'width': new_width,
        'transform': new_transform,
        'crs': small_crs,
        'dtype': 'float32',
        'nodata': np.nan
    })
    with rasterio.open(output_small_tif_path, 'w', **small_profile) as dst:
        dst.write(expanded_small, 1)


if __name__ == "__main__":
    extract_mask_and_cut(
        large_tif_path='C:\\Users\\hanji\\Desktop\\tmp\\湖泊面积\\E_BLK_DEM30m_2000.tif',
        small_tif_path='C:\\Users\\hanji\\Desktop\\tmp\\湖泊面积\\re_100cm.tif',
        output_large_tif_path='C:\\Users\\hanji\\Desktop\\tmp\\湖泊面积\\output_large_exclude_small.tif',
        output_small_tif_path='C:\\Users\\hanji\\Desktop\\tmp\\湖泊面积\\output_small_covering_large.tif'
    )
