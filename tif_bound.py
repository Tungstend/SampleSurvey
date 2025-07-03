import rasterio
import numpy as np
from skimage import measure
from shapely.geometry import LineString
import geopandas as gpd

def extract_precise_edge_line(tif_path, shp_path):
    print("📂 打开 TIFF...")
    with rasterio.open(tif_path) as src:
        image = src.read(1)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    print("🧼 构建有效区域掩膜...")
    mask = (image != nodata) if nodata is not None else (image > 0)
    mask = mask.astype(np.uint8)

    print("🔍 提取边缘轮廓（子像素级别）...")
    contours = measure.find_contours(mask, level=0.5)

    if not contours:
        print("⚠️ 无边界轮廓，可能为空图。")
        return

    print(f"✅ 发现 {len(contours)} 条轮廓，转换为坐标...")
    line_geoms = []
    for contour in contours:
        # contour: (row, col) → map to (x, y)
        coords = [
            rasterio.transform.xy(transform, y, x, offset='center')
            for y, x in contour
        ]
        line = LineString(coords)
        if line.is_valid:
            line_geoms.append(line)

    print("💾 保存为 LineString Shapefile...")
    gdf = gpd.GeoDataFrame(geometry=line_geoms, crs=crs)
    gdf.to_file(shp_path)
    print(f"🎉 成功导出：{shp_path}")

if __name__ == "__main__":
    input = "C:\\Users\\hanji\\Desktop\\tmp\\湖泊面积\\zzzz\\re_100cm.tif"
    output = "C:\\Users\\hanji\\Desktop\\tmp\\湖泊面积\\zzzz\\boundary.shp"
    extract_precise_edge_line(input, output)
