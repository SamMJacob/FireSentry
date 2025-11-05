#!/usr/bin/env python3
"""
Debug LST coordinate transformation
"""

import rasterio
import numpy as np
from datetime import datetime
from pathlib import Path
from rasterio.warp import transform
from rasterio.crs import CRS

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

print(f"Testing LST coordinate transformation:")
print(f"  Input WGS84: ({lat}, {lon})")

if lst_file.exists():
    try:
        with rasterio.open(lst_file) as src:
            print(f"  Raster CRS: {src.crs}")
            print(f"  Raster bounds: {src.bounds}")
            
            # Transform coordinates
            x, y = transform(
                CRS.from_epsg(4326),  # WGS84
                src.crs,              # MODIS Sinusoidal
                [lon], [lat]
            )
            
            print(f"  Transformed coordinates: ({x[0]:.2f}, {y[0]:.2f})")
            
            # Check if transformed point is within bounds
            bounds = src.bounds
            in_bounds = (bounds.left <= x[0] <= bounds.right and 
                        bounds.bottom <= y[0] <= bounds.top)
            print(f"  Within bounds: {in_bounds}")
            
            if in_bounds:
                # Get row, col
                row, col = src.index(x[0], y[0])
                print(f"  Row, Col: ({row}, {col})")
                print(f"  Raster shape: {src.shape}")
                
                if 0 <= row < src.height and 0 <= col < src.width:
                    # Read the value
                    value = src.read(1)[row, col]
                    print(f"  Raw value: {value}")
                    print(f"  NoData: {src.nodata}")
                    print(f"  Is NoData: {value == src.nodata}")
                    print(f"  Is NaN: {np.isnan(value)}")
                    
                    if value != src.nodata and not np.isnan(value):
                        lst_kelvin = value * 0.02
                        print(f"  LST Kelvin: {lst_kelvin}")
                        print(f"  Reasonable range: {200 <= lst_kelvin <= 350}")
                    else:
                        print("  Value is NoData or NaN")
                else:
                    print("  Row/Col out of bounds")
            else:
                print("  Transformed point is outside raster bounds")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
else:
    print("File does not exist!")


