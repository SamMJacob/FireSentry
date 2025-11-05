#!/usr/bin/env python3
"""
Detailed debug LST extraction
"""

import rasterio
import numpy as np
from datetime import datetime
from pathlib import Path

# Test parameters
lat, lon = 30.3165, 78.0322
date = datetime(2020, 3, 15)
doy = date.timetuple().tm_yday

# Calculate tile
h = int((lon + 180) / 10)
v = int((90 - lat) / 10)
tile = f"h{h:02d}v{v:02d}"

# Construct file path
lst_file = Path(f"data/raw/modis_lst/2020/MOD11A1.AA{date.year}{doy:03d}.{tile}.061.LST_Day_1km.tif")

print(f"Testing LST extraction for:")
print(f"  Point: ({lat}, {lon})")
print(f"  Date: {date.date()}")
print(f"  DOY: {doy}")
print(f"  Tile: {tile}")
print(f"  File: {lst_file}")
print(f"  File exists: {lst_file.exists()}")

if lst_file.exists():
    try:
        with rasterio.open(lst_file) as src:
            print(f"\nRaster info:")
            print(f"  CRS: {src.crs}")
            print(f"  Bounds: {src.bounds}")
            print(f"  Shape: {src.shape}")
            print(f"  NoData: {src.nodata}")
            print(f"  Data type: {src.dtypes[0]}")
            
            # Get row, col for the point
            row, col = src.index(lon, lat)
            print(f"\nCoordinate conversion:")
            print(f"  Input: ({lat}, {lon})")
            print(f"  Row, Col: ({row}, {col})")
            print(f"  Bounds check: 0 <= {row} < {src.height} and 0 <= {col} < {src.width}")
            
            if 0 <= row < src.height and 0 <= col < src.width:
                # Read the value
                value = src.read(1)[row, col]
                print(f"  Raw value: {value}")
                print(f"  Is NoData: {value == src.nodata}")
                print(f"  Is NaN: {np.isnan(value)}")
                
                if value != src.nodata and not np.isnan(value):
                    # MODIS LST values are in Kelvin * 0.02 scale factor
                    lst_kelvin = value * 0.02
                    print(f"  LST Kelvin: {lst_kelvin}")
                    print(f"  Reasonable range (200-350K): {200 <= lst_kelvin <= 350}")
                else:
                    print("  Value is NoData or NaN")
            else:
                print("  Point is outside raster bounds")
                
    except Exception as e:
        print(f"Error reading raster: {e}")
else:
    print("File does not exist!")


