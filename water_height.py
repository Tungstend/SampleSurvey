import pandas as pd
import numpy as np

def generate_time_height_table(area_time_path, area_height_path, output_path):
    # 表一：时间-面积
    df_time_area = pd.read_excel(area_time_path)
    time_col = df_time_area.iloc[:, 0].to_numpy()
    area_vals = df_time_area.iloc[:, 1].to_numpy()

    # 表二：面积-高度
    df_area_height = pd.read_excel(area_height_path)
    area_ref = df_area_height.iloc[:, 1].to_numpy()
    height_ref = df_area_height.iloc[:, 0].to_numpy()

    # 按面积升序排序
    sort_idx = np.argsort(area_ref)
    area_ref = area_ref[sort_idx]
    height_ref = height_ref[sort_idx]

    # 构建分段插值函数（平台 + 线性）
    def get_height(area):
        for i in range(len(area_ref) - 1):
            a1, a2 = area_ref[i], area_ref[i + 1]
            h1, h2 = height_ref[i], height_ref[i + 1]

            if a1 <= area < a2:
                if h1 == h2:
                    return h1  # 平台段，直接返回
                else:
                    # 线性插值
                    return h1 + (h2 - h1) * (area - a1) / (a2 - a1)
        return height_ref[-1]  # 超出最大面积时

    # 应用插值
    height_vals = [get_height(a) for a in area_vals]

    # 构建输出表
    df_output = pd.DataFrame({
        '时间': time_col,
        '高度': height_vals
    })

    df_output.to_excel(output_path, index=False)
    print(f"✅ 表三已生成：{output_path}")

if __name__ == "__main__":
    table1_path = "C:\\Users\\hanji\\Desktop\\tmp\\湖泊面积\\时间-面积.xlsx"
    table2_path = "C:\\Users\\hanji\\Desktop\\tmp\\湖泊面积\\面积-高度.xlsx"
    output_path = "C:\\Users\\hanji\\Desktop\\tmp\\湖泊面积\\时间-高度.xlsx"

    generate_time_height_table(table1_path, table2_path, output_path)
