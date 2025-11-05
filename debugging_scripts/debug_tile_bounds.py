#!/usr/bin/env python3
"""
Debug which tile contains the test point
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

print(f"Test point: ({lat}, {lon})")
print(f"Date: {date.date()} (DOY: {doy})")

# Available tiles for this date
tiles = ['h24v05', 'h24v06', 'h25v05', 'h25v06']

for tile in tiles:
    lst_file = Path(f"data/raw/modis_lst/2020/MOD11A1.AA{date.year}{doy:03d}.{tile}.061.LST_Day_1km.tif")
    
    if lst_file.exists():
        try:
            with rasterio.open(lst_file) as src:
                # Transform coordinates
                x, y = transform(
                    CRS.from_epsg(4326),  # WGS84
                    src.crs,              # MODIS Sinusoidal
                    [lon], [lat]
                )
                
                bounds = src.bounds
                in_bounds = (bounds.left <= x[0] <= bounds.right and 
                            bounds.bottom <= y[0] <= bounds.top)
                
                print(f"\nTile {tile}:")
                print(f"  Bounds: {bounds}")
                print(f"  Transformed point: ({x[0]:.2f}, {y[0]:.2f})")
                print(f"  Contains point: {in_bounds}")
                
                if in_bounds:
                    row, col = src.index(x[0], y[0])
                    print(f"  Row, Col: ({row}, {col})")
                    print(f"  Shape: {src.shape}")
                    
                    if 0 <= row < src.height and 0 <= col < src.width:
                        value = src.read(1)[row, col]
                        print(f"  Raw value: {value}")
                        if value != src.nodata:
                            lst_kelvin = value * 0.02
                            print(f"  LST Kelvin: {lst_kelvin}")
                        else:
                            print(f"  NoData value")
                    else:
                        print(f"  Row/Col out of bounds")
                        
        except Exception as e:
            print(f"Error with tile {tile}: {e}")
    else:
        print(f"Tile {tile}: File not found")


