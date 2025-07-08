import os
import re
import rasterio
import numpy as np
import pandas as pd

def extract_year_from_filename(filename):
    """
    从文件名中提取年份，格式为 CLCD_BLK_YYYY.tif
    """
    match = re.search(r"CLCD_BLK_(\d{4})\.tif$", filename)
    return int(match.group(1)) if match else None

def calculate_cultivated_area(tif_path):
    """
    读取栅格文件，统计值为1的像素个数，并根据分辨率计算耕地面积（平方千米）
    """
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        res_x, res_y = src.res  # 自动获取分辨率（米）
        pixel_area_km2 = res_x * res_y / 1e6  # 像素面积（平方千米）
        cultivated_pixels = np.sum(data == 1)  # 值为1的像素数
        return cultivated_pixels * pixel_area_km2

def generate_monthly_area_table(input_folder, output_excel):
    """
    批量读取tif文件，生成逐月耕地面积表并保存为Excel
    """
    year_area_map = {}

    # 遍历文件夹，读取每年的耕地面积
    for file in os.listdir(input_folder):
        year = extract_year_from_filename(file)
        if year:
            tif_path = os.path.join(input_folder, file)
            area_km2 = calculate_cultivated_area(tif_path)
            year_area_map[year] = area_km2
            print(f"{year}年耕地面积: {area_km2:.3f} 平方千米")

    # 构造逐月数据
    rows = []
    for year in sorted(year_area_map):
        area = year_area_map[year]
        for month in range(1, 13):
            date_str = f"{year}-{month:02d}"
            rows.append({"时间": date_str, "面积": area})

    df = pd.DataFrame(rows)
    df.to_excel(output_excel, index=False)
    print(f"\n✅ 耕地面积表已保存到：{output_excel}")


# === 使用方法 ===
if __name__ == "__main__":
    input_folder = r"E:\\数据\\巴里坤耕地"  # 修改为你的tif文件夹路径
    output_excel = r"E:\\数据\\耕地面积结果.xlsx"  # 修改为你想保存的Excel路径
    generate_monthly_area_table(input_folder, output_excel)
