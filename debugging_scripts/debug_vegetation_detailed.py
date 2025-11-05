#!/usr/bin/env python3
"""
Detailed debug vegetation indices tile verification
"""

import rasterio
from datetime import datetime
from pathlib import Path
from rasterio.warp import transform
from rasterio.crs import CRS

# Test parameters
lat, lon = 30.3165, 78.0322
date = datetime(2020, 3, 15)
doy = date.timetuple().tm_yday

print(f"Testing vegetation indices tile verification:")
print(f"  Point: ({lat}, {lon})")
print(f"  Date: {date.date()}")
print(f"  DOY: {doy}")

# Calculate tile
h = int((lon + 180) / 10)
v = int((90 - lat) / 10)
tile = f"h{h:02d}v{v:02d}"
print(f"  Calculated tile: {tile}")

# Test the specific files that should exist
band_mapping = {
    'red': 'B01',
    'nir': 'B02', 
    'blue': 'B03',
    'swir1': 'B06',
    'swir2': 'B07'
}

year_dir = Path("data/raw/modis_sr/2020")

for band_name, band_code in band_mapping.items():
    band_file = f"MYD09GA.AA{date.year}{doy:03d}.{tile}.061.{band_code}.tif"
    filepath = year_dir / band_file
    
    print(f"\nTesting {band_name} ({band_code}):")
    print(f"  File: {band_file}")
    print(f"  Exists: {filepath.exists()}")
    
    if filepath.exists():
        try:
            with rasterio.open(filepath) as src:
                print(f"  CRS: {src.crs}")
                print(f"  Bounds: {src.bounds}")
                print(f"  Shape: {src.shape}")
                
                # Transform coordinates
                x, y = transform(
                    CRS.from_epsg(4326),  # WGS84
                    src.crs,              # MODIS Sinusoidal
                    [lon], [lat]
                )
                
                print(f"  Transformed coordinates: ({x[0]:.2f}, {y[0]:.2f})")
                
                # Check if point is within bounds
                bounds = src.bounds
                in_bounds = (bounds.left <= x[0] <= bounds.right and 
                            bounds.bottom <= y[0] <= bounds.top)
                print(f"  Within bounds: {in_bounds}")
                
                if in_bounds:
                    row, col = src.index(x[0], y[0])
                    print(f"  Row, Col: ({row}, {col})")
                    
                    if 0 <= row < src.height and 0 <= col < src.width:
                        value = src.read(1)[row, col]
                        print(f"  Raw value: {value}")
                        if value != src.nodata:
                            scaled_value = value / 10000.0
                            print(f"  Scaled value: {scaled_value}")
                        else:
                            print(f"  NoData value")
                    else:
                        print(f"  Row/Col out of bounds")
                else:
                    print(f"  Point outside raster bounds")
                    
        except Exception as e:
            print(f"  Error: {e}")
    else:
        print(f"  File not found!")


