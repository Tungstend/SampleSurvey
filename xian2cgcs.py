import rasterio
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling

def reproject_large_to_2000(input_path, output_path, target_crs="EPSG:4540", resolution=30):
    with rasterio.open(input_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds, resolution=resolution
        )

        profile = src.profile.copy()
        profile.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'dtype': 'float32',
            'nodata': None
        })

        data = np.empty((height, width), dtype='float32')
        reproject(
            source=src.read(1),
            destination=data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest
        )

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data, 1)

if __name__ == "__main__":
    reproject_large_to_2000(
        input_path='C:\\Users\\hanji\\Desktop\\tmp\\湖泊面积\\E_BLK_DEM30m.tif',
        output_path='C:\\Users\\hanji\\Desktop\\tmp\\湖泊面积\\E_BLK_DEM30m_2000.tif',
        target_crs='EPSG:4540',  # CGCS2000
        resolution=30
    )
