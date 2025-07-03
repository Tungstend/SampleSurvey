import rasterio
import numpy as np

def get_tif_min_value_and_position(filepath):
    """
    获取 tif 文件的最小值及其在栅格中的位置（行列索引）

    返回:
        (min_value, row, col)
    """
    with rasterio.open(filepath) as src:
        data = src.read(1)
        nodata = src.nodata

        if nodata is not None:
            data = np.where(data == nodata, np.nan, data)

        min_val = np.nanmin(data)
        pos = np.unravel_index(np.nanargmin(data), data.shape)  # 返回(row, col)

    return min_val, pos[0], pos[1]

# 示例调用
if __name__ == "__main__":
    file_path = "C:\\Users\\hanji\\Desktop\\tmp\\湖泊面积\\re_100cm.tif"
    val, row, col = get_tif_min_value_and_position(file_path)
    print(f"最小值为 {val}，位置为 行={row}，列={col}")
