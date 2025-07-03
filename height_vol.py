import rasterio
import numpy as np
import pandas as pd

def compute_area_volume(filepath, water_level, pixel_area=1.0):
    """
    根据高程图计算指定水位下的湖泊覆盖面积与库容
    """
    with rasterio.open(filepath) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodata

        # 有效区域掩膜
        if nodata is not None:
            mask = data != nodata
        else:
            mask = ~np.isnan(data)

        valid_data = data[mask]

        # 被淹没的像素
        submerged = valid_data < water_level
        submerged_elevation = valid_data[submerged]

        # 面积（像素数量 × 单像素面积）
        area_m2 = submerged.sum() * pixel_area

        # 库容 = ∑(水位 - 高程) × 面积
        volume_m3 = np.sum(water_level - submerged_elevation) * pixel_area

    return area_m2, volume_m3

def export_area_volume_table(filepath, min_level, max_level, step=0.2, pixel_area=1.0, output_excel='area_volume.xlsx'):
    """
    循环计算不同水位的面积和库容，并导出为 Excel 表格
    """
    levels = np.arange(min_level, max_level + step, step)
    results = []

    for wl in levels:
        area, volume = compute_area_volume(filepath, wl, pixel_area)
        results.append({
            '水位高程 (m)': wl,
            '面积 (m²)': round(area, 2),
            '库容 (m³)': round(volume, 2)
        })

        # 打印进度
        print(f"处理水位 {wl:.2f} m，面积: {area:.2f} m²，库容: {volume:.2f} m³")

    df = pd.DataFrame(results)
    df.to_excel(output_excel, index=False)
    print(f"已保存至: {output_excel}")

# 示例调用
if __name__ == "__main__":
    dem_path = 'C:\\Users\\hanji\\Documents\\WeChat Files\\wxid_0xiky9xxszp622\\FileStorage\\File\\2025-07\\zzzz\\湖泊东.tif'        # 高程tif路径
    min_wl = 1579                    # 最小水位
    max_wl = 1582                    # 最大水位
    step = 0.1                       # 水位间隔
    output = 'C:\\Users\\hanji\\Documents\\WeChat Files\\wxid_0xiky9xxszp622\\FileStorage\\File\\2025-07\\zzzz\\lake_area_volume_east.xlsx'  # 输出路径

    export_area_volume_table(dem_path, min_wl, max_wl, step, pixel_area=1.0, output_excel=output)
