#!/usr/bin/env python3
"""
Final coverage analysis with ACTUAL tile bounds from real files.
"""

import pandas as pd

# Load fire data
df = pd.read_csv('data/raw/firms/firms_uttarakhand_2020_2025.csv')
df = df.rename(columns={'latitude': 'lat', 'longitude': 'lon'})

# ACTUAL tile coverage (from real MODIS files)
ACTUAL_TILE_COVERAGE = {
    'h24v05': {'lon_min': 69.28, 'lon_max': 91.38, 'lat_min': 30.00, 'lat_max': 40.00},
    'h24v06': {'lon_min': 63.85, 'lon_max': 80.83, 'lat_min': 20.00, 'lat_max': 30.00},
    'h25v06': {'lon_min': 74.49, 'lon_max': 92.38, 'lat_min': 20.00, 'lat_max': 30.00},
}

# Uttarakhand bounds
uk_west, uk_east = 77.575402, 81.044789
uk_south, uk_north = 28.709556, 31.459016

print("="*80)
print("FINAL COVERAGE ANALYSIS WITH ACTUAL TILE BOUNDS")
print("="*80)
print()

print(f"Uttarakhand: {uk_west:.2f}-{uk_east:.2f}°E, {uk_south:.2f}-{uk_north:.2f}°N")
print()

print("Available tiles (ACTUAL bounds):")
print("-" * 80)
for tile, bounds in sorted(ACTUAL_TILE_COVERAGE.items()):
    print(f"{tile}:")
    print(f"  Longitude: {bounds['lon_min']:.2f}°E - {bounds['lon_max']:.2f}°E")
    print(f"  Latitude:  {bounds['lat_min']:.2f}°N - {bounds['lat_max']:.2f}°N")
print()

# Check coverage
print("="*80)
print("COVERAGE CHECK")
print("="*80)
print()

def point_in_tile(lat, lon, tile_bounds):
    return (tile_bounds['lon_min'] <= lon <= tile_bounds['lon_max'] and
            tile_bounds['lat_min'] <= lat <= tile_bounds['lat_max'])

# Check each fire point
covered = 0
uncovered = []
coverage_by_tile = {tile: 0 for tile in ACTUAL_TILE_COVERAGE.keys()}

for idx, row in df.iterrows():
    lat, lon = row['lat'], row['lon']
    found = False
    
    for tile, bounds in ACTUAL_TILE_COVERAGE.items():
        if point_in_tile(lat, lon, bounds):
            coverage_by_tile[tile] += 1
            found = True
            break  # Count each point only once
    
    if found:
        covered += 1
    else:
        uncovered.append((lat, lon))

print(f"Total fire points: {len(df)}")
print(f"Covered by existing tiles: {covered} ({covered/len(df)*100:.1f}%)")
print(f"NOT covered: {len(uncovered)} ({len(uncovered)/len(df)*100:.1f}%)")
print()

print("Coverage by tile:")
for tile in sorted(coverage_by_tile.keys()):
    count = coverage_by_tile[tile]
    print(f"  {tile}: {count} fires ({count/len(df)*100:.1f}%)")
print()

if uncovered:
    uncovered_df = pd.DataFrame(uncovered, columns=['lat', 'lon'])
    print("="*80)
    print("UNCOVERED POINTS ANALYSIS")
    print("="*80)
    print()
    print(f"Uncovered points: {len(uncovered)}")
    print(f"Latitude range:  {uncovered_df['lat'].min():.4f}°N - {uncovered_df['lat'].max():.4f}°N")
    print(f"Longitude range: {uncovered_df['lon'].min():.4f}°E - {uncovered_df['lon'].max():.4f}°E")
    print()
    
    # Determine which tiles are needed
    print("Required tiles for uncovered points:")
    print("-" * 80)
    
    # Check if h23v05 or h23v06 would cover them
    # h23v05: ~67-78.3°E, 30-40°N
    # h23v06: ~67-78.3°E, 20-30°N
    
    need_h23v05 = uncovered_df[(uncovered_df['lon'] < 69.28) & (uncovered_df['lat'] >= 30.0)]
    need_h23v06 = uncovered_df[(uncovered_df['lon'] < 63.85) & (uncovered_df['lat'] < 30.0)]
    
    print(f"Need h23v05 (<69.28°E, ≥30°N): {len(need_h23v05)} fires")
    print(f"Need h23v06 (<63.85°E, <30°N): {len(need_h23v06)} fires")
    
    print()
    print("Sample uncovered points:")
    for lat, lon in uncovered[:10]:
        print(f"  ({lat:.4f}°N, {lon:.4f}°E)")
else:
    print("="*80)
    print("✅ ALL FIRE POINTS ARE COVERED!")
    print("="*80)
    print()
    print("No additional tiles needed!")
    print("The existing tiles (h24v05, h24v06, h25v06) cover 100% of Uttarakhand fires.")

print()
print("="*80)

