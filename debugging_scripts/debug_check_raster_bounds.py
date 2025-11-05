#!/usr/bin/env python3
"""
Check the actual geographic bounds of the downloaded MODIS SR files.
"""

import rasterio
from rasterio.warp import transform
from rasterio.crs import CRS
from pathlib import Path

# Test point
test_lat, test_lon = 30.3165, 78.0322  # Dehradun

# File to check
test_file = Path("data/raw/modis_sr/2020/MYD09GA.AA2020061.h25v05.061.B01.tif")

print(f"Test Point: ({test_lat}, {test_lon})")
print(f"File: {test_file.name}")
print()

if not test_file.exists():
    print(f"❌ File not found: {test_file}")
    exit(1)

with rasterio.open(test_file) as src:
    print(f"CRS: {src.crs}")
    print(f"Bounds (native CRS): {src.bounds}")
    print(f"  Left: {src.bounds.left}")
    print(f"  Bottom: {src.bounds.bottom}")
    print(f"  Right: {src.bounds.right}")
    print(f"  Top: {src.bounds.top}")
    print()
    
    # Transform test point to raster CRS
    x, y = transform(
        CRS.from_epsg(4326),  # WGS84
        src.crs,              # MODIS Sinusoidal
        [test_lon], [test_lat]
    )
    
    print(f"Test point in raster CRS: ({x[0]}, {y[0]})")
    print()
    
    # Check if point is within bounds
    within = (src.bounds.left <= x[0] <= src.bounds.right and 
              src.bounds.bottom <= y[0] <= src.bounds.top)
    
    if within:
        print("✓ Point IS within raster bounds")
        
        # Get row, col
        row, col = src.index(x[0], y[0])
        print(f"  Row: {row}, Col: {col}")
        print(f"  Raster shape: {src.shape}")
        
        # Check if row, col is valid
        if 0 <= row < src.height and 0 <= col < src.width:
            value = src.read(1)[row, col]
            print(f"  Value at point: {value}")
        else:
            print(f"  ❌ Row/Col outside raster dimensions!")
    else:
        print("✗ Point IS NOT within raster bounds")
        print(f"  X: {x[0]} (bounds: {src.bounds.left} to {src.bounds.right})")
        print(f"  Y: {y[0]} (bounds: {src.bounds.bottom} to {src.bounds.top})")
        print()
        print("This means the raster was aggressively clipped during download.")


