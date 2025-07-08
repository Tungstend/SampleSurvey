import os
import rasterio
import pandas as pd
from datetime import datetime

def process_ndvi_tifs(input_folder, output_excel):
    results = []

    for file in os.listdir(input_folder):
        if file.endswith(".tif") and file.startswith("EVI_BLK_"):
            try:
                # 提取年月
                parts = file.replace(".tif", "").split("_")
                year = int(parts[2])
                month = int(parts[3])
                date_str = f"{year}-{month:02d}"

                # 读取tif
                file_path = os.path.join(input_folder, file)
                with rasterio.open(file_path) as src:
                    data = src.read(1).astype(float)
                    nodata = src.nodata

                    # 掩膜掉 nodata 像素
                    if nodata is not None:
                        valid_mask = data != nodata
                        valid_data = data[valid_mask]
                    else:
                        valid_data = data.flatten()

                    # 转换为 NDVI
                    ndvi = valid_data / 255.0

                    # 计算统计值
                    pixel_count = len(ndvi)
                    ndvi_sum = ndvi.sum()
                    ndvi_mean = ndvi.mean()

                    results.append((date_str, ndvi_sum, ndvi_mean, pixel_count))

            except Exception as e:
                print(f"跳过文件 {file}，错误信息: {e}")

    # 排序
    results.sort(key=lambda x: datetime.strptime(x[0], "%Y-%m"))

    # 写入 Excel
    df = pd.DataFrame(results, columns=["时间", "NDVI总和", "平均NDVI", "像素总数"])
    df.to_excel(output_excel, index=False)
    print(f"结果已保存至 {output_excel}")


if __name__ == "__main__":
    input_folder = r"E:\\数据\\巴里坤EVI"
    output_excel = r"E:\\数据\\ndvi结果.xlsx"
    process_ndvi_tifs(input_folder, output_excel)