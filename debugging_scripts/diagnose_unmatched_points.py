#!/usr/bin/env python3
"""
Diagnose why so many points don't match any tile.
Check if the tile coverage bounds are wrong.
"""

import pandas as pd
import numpy as np

# Load fire data
df = pd.read_csv('data/raw/firms/firms_uttarakhand_2020_2025.csv')
df = df.rename(columns={'latitude': 'lat', 'longitude': 'lon'})

# Current tile coverage (from modis_tiles.py)
TILE_COVERAGE = {
    'h23v05': {'lon_min': 67.0, 'lon_max': 78.3, 'lat_min': 30.0, 'lat_max': 40.0},
    'h23v06': {'lon_min': 67.0, 'lon_max': 78.3, 'lat_min': 20.0, 'lat_max': 30.0},
    'h24v05': {'lon_min': 78.3, 'lon_max': 80.8, 'lat_min': 30.0, 'lat_max': 40.0},
    'h24v06': {'lon_min': 69.3, 'lon_max': 74.5, 'lat_min': 20.0, 'lat_max': 30.0},
    'h25v05': {'lon_min': 91.4, 'lon_max': 92.4, 'lat_min': 30.0, 'lat_max': 40.0},
    'h25v06': {'lon_min': 80.8, 'lon_max': 85.1, 'lat_min': 20.0, 'lat_max': 30.0},
}

def find_tile_for_point(lat, lon):
    """Find which tile(s) a point falls into."""
    matching_tiles = []
    for tile, bounds in TILE_COVERAGE.items():
        if (bounds['lon_min'] <= lon <= bounds['lon_max'] and 
            bounds['lat_min'] <= lat <= bounds['lat_max']):
            matching_tiles.append(tile)
    return matching_tiles

# Find unmatched points
unmatched = []
for idx, row in df.iterrows():
    lat, lon = row['lat'], row['lon']
    tiles = find_tile_for_point(lat, lon)
    if not tiles:
        unmatched.append({'lat': lat, 'lon': lon})

unmatched_df = pd.DataFrame(unmatched)

print("="*80)
print("UNMATCHED POINTS ANALYSIS")
print("="*80)
print()

print(f"Total fire points: {len(df)}")
print(f"Unmatched points: {len(unmatched)} ({len(unmatched)/len(df)*100:.1f}%)")
print()

if len(unmatched) > 0:
    print("Geographic distribution of unmatched points:")
    print("-" * 80)
    print(f"Latitude range:  {unmatched_df['lat'].min():.4f}°N - {unmatched_df['lat'].max():.4f}°N")
    print(f"Longitude range: {unmatched_df['lon'].min():.4f}°E - {unmatched_df['lon'].max():.4f}°E")
    print()
    
    # Check which region they fall into
    print("Breakdown by region:")
    print("-" * 80)
    
    # Northern (>30N)
    northern = unmatched_df[unmatched_df['lat'] > 30.0]
    # Southern (<30N)
    southern = unmatched_df[unmatched_df['lat'] <= 30.0]
    
    print(f"Northern (>30°N): {len(northern)} points")
    if len(northern) > 0:
        print(f"  Lat: {northern['lat'].min():.2f}°N - {northern['lat'].max():.2f}°N")
        print(f"  Lon: {northern['lon'].min():.2f}°E - {northern['lon'].max():.2f}°E")
    
    print(f"Southern (≤30°N): {len(southern)} points")
    if len(southern) > 0:
        print(f"  Lat: {southern['lat'].min():.2f}°N - {southern['lat'].max():.2f}°N")
        print(f"  Lon: {southern['lon'].min():.2f}°E - {southern['lon'].max():.2f}°E")
    
    print()
    print("Longitude breakdown:")
    print("-" * 80)
    
    # Check longitude ranges
    west_of_h23 = unmatched_df[unmatched_df['lon'] < 67.0]
    h23_h24_gap = unmatched_df[(unmatched_df['lon'] >= 78.3) & (unmatched_df['lon'] < 78.3)]  # Should be none
    between_h24_h25 = unmatched_df[(unmatched_df['lon'] > 78.3) & (unmatched_df['lon'] < 80.8)]
    h24_h25_gap = unmatched_df[(unmatched_df['lon'] >= 80.8) & (unmatched_df['lon'] < 80.8)]  # Should be none
    east_of_h25 = unmatched_df[unmatched_df['lon'] >= 85.1]
    
    print(f"West of h23 (<67°E): {len(west_of_h23)}")
    print(f"Between h24v05 and h25v06 (78.3-80.8°E): {len(between_h24_h25)}")
    print(f"East of h25 (>85.1°E): {len(east_of_h25)}")
    
    # Most common case - between tiles
    between_tiles = unmatched_df[(unmatched_df['lon'] >= 78.3) & (unmatched_df['lon'] <= 80.8)]
    print()
    print(f"⚠️  ISSUE: {len(between_tiles)} points are in 78.3-80.8°E range")
    print("   This range SHOULD be covered by h24v05 or h24v06!")
    print()
    
    # Check latitude for these points
    if len(between_tiles) > 0:
        northern_gap = between_tiles[between_tiles['lat'] > 30.0]
        southern_gap = between_tiles[between_tiles['lat'] <= 30.0]
        
        print(f"   Northern (>30°N): {len(northern_gap)} - should be h24v05")
        print(f"   Southern (≤30°N): {len(southern_gap)} - should be h24v06")
        print()
        
        if len(southern_gap) > 0:
            print(f"   ❌ PROBLEM: h24v06 only covers 69.3-74.5°E")
            print(f"      But we have points at 78.3-80.8°E, ≤30°N")
            print(f"      These points need a different tile!")
            print()
            print("   Sample southern gap points:")
            for _, row in southern_gap.head(5).iterrows():
                print(f"      ({row['lat']:.4f}°N, {row['lon']:.4f}°E)")

print()
print("="*80)
print("DIAGNOSIS")
print("="*80)
print()

print("The tile coverage map has ERRORS!")
print()
print("Issues found:")
print("1. h24v06 bounds are wrong (69.3-74.5°E doesn't cover 78.3-80.8°E)")
print("2. Many points in the 78.3-80.8°E, <30°N range are unmatched")
print()
print("Likely solution:")
print("- h24v05 should cover BOTH >30°N AND ≤30°N in the 78.3-80.8°E range")
print("- OR there's another tile we're missing for the southern part")
print()
print("Need to verify actual MODIS tile boundaries!")

