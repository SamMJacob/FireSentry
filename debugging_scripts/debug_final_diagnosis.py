#!/usr/bin/env python3
"""
Final diagnosis - tile calculation vs available data
"""

import rasterio
from rasterio.warp import transform_bounds
from rasterio.crs import CRS
from pathlib import Path

print("="*80)
print("FINAL DIAGNOSIS - MODIS SR TILE ISSUE")
print("="*80)

# Check what tiles we actually have
year_dir = Path("data/raw/modis_sr/2020")
all_files = list(year_dir.glob("MYD09GA.AA2020061.*.B01.tif"))
available_tiles = set()
for f in all_files:
    # Extract tile from filename
    parts = f.name.split('.')
    for part in parts:
        if part.startswith('h') and 'v' in part:
            available_tiles.add(part)
            break

print(f"\n1. AVAILABLE TILES IN DATASET:")
print(f"   {sorted(available_tiles)}")

# Check geographic bounds of each available tile
print(f"\n2. GEOGRAPHIC BOUNDS OF AVAILABLE TILES:")
for tile in sorted(available_tiles):
    test_file = year_dir / f"MYD09GA.AA2020061.{tile}.061.B01.tif"
    if test_file.exists():
        with rasterio.open(test_file) as src:
            wgs84_bounds = transform_bounds(
                src.crs,
                CRS.from_epsg(4326),
                src.bounds.left,
                src.bounds.bottom,
                src.bounds.right,
                src.bounds.top
            )
            print(f"   {tile}: {wgs84_bounds[0]:.2f}°E to {wgs84_bounds[2]:.2f}°E, "
                  f"{wgs84_bounds[1]:.2f}°N to {wgs84_bounds[3]:.2f}°N")

# Uttarakhand bounds
print(f"\n3. UTTARAKHAND BOUNDS:")
print(f"   28.7°N to 31.4°N, 77.6°E to 81.0°E")

# Check what tiles SHOULD cover Uttarakhand
print(f"\n4. TILES THAT SHOULD COVER UTTARAKHAND:")
print(f"   Based on standard MODIS grid:")
print(f"   - h25v05: 70-80°E, 30-40°N")
print(f"   - h25v06: 70-80°E, 20-30°N")
print(f"   - h26v05: 80-90°E, 30-40°N")
print(f"   - h26v06: 80-90°E, 20-30°N")

# Test points
test_points = [
    ("Original test point", 30.3165, 78.0322),
    ("Within h25v05 bounds", 31.0, 81.5),
    ("Center of Uttarakhand", 30.0, 79.4),
]

print(f"\n5. TEST POINTS ANALYSIS:")
for name, lat, lon in test_points:
    h = int((lon + 180) / 10)
    v = int((90 - lat) / 10)
    calc_tile = f"h{h:02d}v{v:02d}"
    
    # Check if point is in any available tile
    found_in_tile = None
    for tile in sorted(available_tiles):
        test_file = year_dir / f"MYD09GA.AA2020061.{tile}.061.B01.tif"
        if test_file.exists():
            try:
                with rasterio.open(test_file) as src:
                    wgs84_bounds = transform_bounds(
                        src.crs,
                        CRS.from_epsg(4326),
                        src.bounds.left,
                        src.bounds.bottom,
                        src.bounds.right,
                        src.bounds.top
                    )
                    if (wgs84_bounds[0] <= lon <= wgs84_bounds[2] and 
                        wgs84_bounds[1] <= lat <= wgs84_bounds[3]):
                        found_in_tile = tile
                        break
            except:
                pass
    
    print(f"   {name}:")
    print(f"     Coordinates: ({lat}°N, {lon}°E)")
    print(f"     Calculated tile: {calc_tile}")
    print(f"     Tile available: {calc_tile in available_tiles}")
    print(f"     Actually found in: {found_in_tile if found_in_tile else 'NONE'}")

print(f"\n6. DIAGNOSIS:")
if len(available_tiles) < 4:
    print(f"   ❌ DATA ISSUE: Missing tiles!")
    print(f"   Expected tiles for Uttarakhand: h25v05, h25v06, h26v05, h26v06")
    print(f"   Available tiles: {sorted(available_tiles)}")
    missing = set(['h25v05', 'h25v06', 'h26v05', 'h26v06']) - available_tiles
    print(f"   Missing tiles: {sorted(missing)}")
    print(f"\n   SOLUTION: Need to re-download MODIS SR data to include missing tiles")
else:
    print(f"   ✅ All expected tiles available")
    print(f"   Issue is likely with tile calculation or coordinate transformation")


