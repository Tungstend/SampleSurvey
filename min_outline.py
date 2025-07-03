import rasterio
import numpy as np
from scipy.ndimage import binary_erosion

def get_tif_effective_boundary_min_with_position(filepath):
    """
    查找 tif 文件中有效区域（不规则多边形）边界上的最小值及其位置

    参数:
        filepath (str): tif 文件路径

    返回:
        (min_val, row, col, x, y): 最小值、行列号、地理坐标
    """
    with rasterio.open(filepath) as src:
        data = src.read(1)
        nodata = src.nodata
        transform = src.transform  # 地理坐标转换信息

        # 有效像素掩膜（非 nodata）
        if nodata is not None:
            valid_mask = data != nodata
        else:
            valid_mask = ~np.isnan(data)

        # 腐蚀后做边界提取
        eroded = binary_erosion(valid_mask)
        boundary_mask = valid_mask & ~eroded

        # 获取边界中最小值及其位置
        boundary_data = np.where(boundary_mask, data, np.nan)
        min_val = np.nanmin(boundary_data)
        min_pos = np.unravel_index(np.nanargmin(boundary_data), data.shape)  # (row, col)

        # 转换为地理坐标
        row, col = min_pos
        x, y = src.transform * (col, row)

    return min_val, row, col, x, y

# 示例调用
if __name__ == "__main__":
    path = 'C:\\Users\\hanji\\Desktop\\tmp\\湖泊面积\\re_100cm.tif'
    minval, row, col, x, y = get_tif_effective_boundary_min_with_position(path)
    print(f"边界最小值: {minval}")
    print(f"位置: 行={row}, 列={col}")
    print(f"地理坐标: X={x}, Y={y}")
