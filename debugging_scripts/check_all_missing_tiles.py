#!/usr/bin/env python3
"""
Comprehensive check of ALL fire points to find ALL missing MODIS tiles.
This will ensure we download everything needed in one go.
"""

import pandas as pd
from collections import defaultdict

# Load fire data
print("Loading fire data...")
df = pd.read_csv('data/raw/firms/firms_uttarakhand_2020_2025.csv')
df = df.rename(columns={'latitude': 'lat', 'longitude': 'lon'})

print(f"Total fire points: {len(df)}")
print()

# Available tiles (what we currently have)
available_sr_tiles = ['h24v05', 'h24v06', 'h25v06']
available_lst_tiles = ['h24v05', 'h24v06', 'h25v05', 'h25v06']

# Tile coverage (approximate geographic bounds)
TILE_COVERAGE = {
    'h23v05': {'lon_min': 67.0, 'lon_max': 78.3, 'lat_min': 30.0, 'lat_max': 40.0},
    'h23v06': {'lon_min': 67.0, 'lon_max': 78.3, 'lat_min': 20.0, 'lat_max': 30.0},
    'h24v05': {'lon_min': 78.3, 'lon_max': 80.8, 'lat_min': 30.0, 'lat_max': 40.0},
    'h24v06': {'lon_min': 69.3, 'lon_max': 74.5, 'lat_min': 20.0, 'lat_max': 30.0},
    'h25v05': {'lon_min': 91.4, 'lon_max': 92.4, 'lat_min': 30.0, 'lat_max': 40.0},
    'h25v06': {'lon_min': 80.8, 'lon_max': 85.1, 'lat_min': 20.0, 'lat_max': 30.0},
    'h26v05': {'lon_min': 102.0, 'lon_max': 104.0, 'lat_min': 30.0, 'lat_max': 40.0},
    'h26v06': {'lon_min': 92.5, 'lon_max': 96.8, 'lat_min': 20.0, 'lat_max': 30.0},
}

def find_tile_for_point(lat, lon):
    """Find which tile(s) a point falls into."""
    matching_tiles = []
    for tile, bounds in TILE_COVERAGE.items():
        if (bounds['lon_min'] <= lon <= bounds['lon_max'] and 
            bounds['lat_min'] <= lat <= bounds['lat_max']):
            matching_tiles.append(tile)
    return matching_tiles

# Check each fire point
print("Checking all fire points for required tiles...")
print()

tile_counts = defaultdict(int)
points_by_tile = defaultdict(list)
points_without_tile = []

for idx, row in df.iterrows():
    lat, lon = row['lat'], row['lon']
    tiles = find_tile_for_point(lat, lon)
    
    if not tiles:
        points_without_tile.append((lat, lon))
    else:
        for tile in tiles:
            tile_counts[tile] += 1
            points_by_tile[tile].append((lat, lon))

# Report results
print("="*80)
print("TILE REQUIREMENT ANALYSIS")
print("="*80)
print()

print("Fire points by required tile:")
print("-" * 80)
for tile in sorted(tile_counts.keys()):
    count = tile_counts[tile]
    percentage = (count / len(df)) * 100
    in_sr = "✅" if tile in available_sr_tiles else "❌"
    in_lst = "✅" if tile in available_lst_tiles else "❌"
    status = "MISSING" if (tile not in available_sr_tiles or tile not in available_lst_tiles) else "Available"
    
    print(f"{tile}: {count:5d} fires ({percentage:5.1f}%) | SR: {in_sr} | LST: {in_lst} | {status}")

print()
print("="*80)
print("MISSING TILES SUMMARY")
print("="*80)
print()

# Find missing SR tiles
missing_sr_tiles = set()
for tile in tile_counts.keys():
    if tile not in available_sr_tiles:
        missing_sr_tiles.add(tile)

# Find missing LST tiles  
missing_lst_tiles = set()
for tile in tile_counts.keys():
    if tile not in available_lst_tiles:
        missing_lst_tiles.add(tile)

print("Missing MODIS SR tiles (MYD09GA):")
if missing_sr_tiles:
    for tile in sorted(missing_sr_tiles):
        count = tile_counts[tile]
        percentage = (count / len(df)) * 100
        print(f"  ❌ {tile}: {count} fires ({percentage:.1f}%)")
    total_missing_sr = sum(tile_counts[t] for t in missing_sr_tiles)
    print(f"  Total: {total_missing_sr} fires ({total_missing_sr/len(df)*100:.1f}%)")
else:
    print("  ✅ None - all tiles available!")

print()

print("Missing MODIS LST tiles (MOD11A1):")
if missing_lst_tiles:
    for tile in sorted(missing_lst_tiles):
        count = tile_counts[tile]
        percentage = (count / len(df)) * 100
        print(f"  ❌ {tile}: {count} fires ({percentage:.1f}%)")
    total_missing_lst = sum(tile_counts[t] for t in missing_lst_tiles)
    print(f"  Total: {total_missing_lst} fires ({total_missing_lst/len(df)*100:.1f}%)")
else:
    print("  ✅ None - all tiles available!")

print()

# Points without any tile match
if points_without_tile:
    print(f"⚠️  WARNING: {len(points_without_tile)} points don't match any known tile!")
    print("Sample points:")
    for lat, lon in points_without_tile[:5]:
        print(f"  ({lat:.4f}, {lon:.4f})")
    print()

print("="*80)
print("DOWNLOAD REQUIREMENTS")
print("="*80)
print()

all_missing = missing_sr_tiles.union(missing_lst_tiles)
if all_missing:
    print("You need to download these tiles:")
    for tile in sorted(all_missing):
        needs_sr = "✅" if tile in missing_sr_tiles else "  "
        needs_lst = "✅" if tile in missing_lst_tiles else "  "
        count = tile_counts[tile]
        percentage = (count / len(df)) * 100
        print(f"  {tile}: SR:{needs_sr} LST:{needs_lst} | {count} fires ({percentage:.1f}%)")
    
    print()
    print("Products to download:")
    if missing_sr_tiles:
        print(f"  • MYD09GA (Surface Reflectance): {sorted(missing_sr_tiles)}")
    if missing_lst_tiles:
        print(f"  • MOD11A1 (Land Surface Temperature): {sorted(missing_lst_tiles)}")
    
    print()
    print("Period: Dry season (Feb-Jun) for years 2020-2024")
    
    total_affected = len(set().union(*[set(points_by_tile[t]) for t in all_missing]))
    print(f"\nTotal unique fire points affected: {total_affected} ({total_affected/len(df)*100:.1f}%)")
else:
    print("✅ All required tiles are available!")
    print("No downloads needed!")

print()
print("="*80)

