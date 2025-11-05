#!/usr/bin/env python3
"""
Check the actual geographic bounds of MODIS SR rasters
"""

import rasterio
from rasterio.warp import transform_bounds
from rasterio.crs import CRS
from pathlib import Path

test_file = Path("data/raw/modis_sr/2020/MYD09GA.AA2020061.h25v05.061.B01.tif")

if test_file.exists():
    with rasterio.open(test_file) as src:
        print(f"Testing file: {test_file.name}")
        print("="*60)
        
        print(f"\nRaster CRS: {src.crs}")
        print(f"Raster bounds (native CRS):")
        print(f"  Left: {src.bounds.left:.2f}")
        print(f"  Right: {src.bounds.right:.2f}")
        print(f"  Bottom: {src.bounds.bottom:.2f}")
        print(f"  Top: {src.bounds.top:.2f}")
        print(f"  Width: {src.width}, Height: {src.height}")
        
        # Transform bounds to WGS84
        wgs84_bounds = transform_bounds(
            src.crs,
            CRS.from_epsg(4326),
            src.bounds.left,
            src.bounds.bottom,
            src.bounds.right,
            src.bounds.top
        )
        
        print(f"\nRaster bounds (WGS84):")
        print(f"  West: {wgs84_bounds[0]:.4f}°E")
        print(f"  South: {wgs84_bounds[1]:.4f}°N")
        print(f"  East: {wgs84_bounds[2]:.4f}°E")
        print(f"  North: {wgs84_bounds[3]:.4f}°N")
        
        # Expected h25v05 bounds
        print(f"\nExpected h25v05 tile bounds:")
        print(f"  Longitude: 70°E to 80°E")
        print(f"  Latitude: 30°N to 40°N")
        
        # Test point
        test_lat, test_lon = 30.3165, 78.0322
        print(f"\nTest point: ({test_lat}°N, {test_lon}°E)")
        
        # Check if test point is within geographic bounds
        is_within = (wgs84_bounds[0] <= test_lon <= wgs84_bounds[2] and 
                    wgs84_bounds[1] <= test_lat <= wgs84_bounds[3])
        print(f"Is test point within raster geographic bounds? {is_within}")
        
        if not is_within:
            print(f"\n❌ PROBLEM: Test point is OUTSIDE the raster's geographic bounds!")
            print(f"This means the raster was clipped and doesn't cover the full tile extent.")
            print(f"\nThe raster only covers:")
            print(f"  {wgs84_bounds[0]:.4f}°E to {wgs84_bounds[2]:.4f}°E")
            print(f"  {wgs84_bounds[1]:.4f}°N to {wgs84_bounds[3]:.4f}°N")
            print(f"\nBut the test point is at:")
            print(f"  {test_lon}°E, {test_lat}°N")
            
            # Check which bound is violated
            if test_lon < wgs84_bounds[0]:
                print(f"  Longitude is too far WEST (need {wgs84_bounds[0]:.4f}°E, got {test_lon}°E)")
            elif test_lon > wgs84_bounds[2]:
                print(f"  Longitude is too far EAST")
            if test_lat < wgs84_bounds[1]:
                print(f"  Latitude is too far SOUTH")
            elif test_lat > wgs84_bounds[3]:
                print(f"  Latitude is too far NORTH (need ≤ {wgs84_bounds[3]:.4f}°N, got {test_lat}°N)")
        else:
            print(f"\n✅ Test point is within raster bounds!")
else:
    print(f"File not found: {test_file}")


