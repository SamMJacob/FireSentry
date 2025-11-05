#!/usr/bin/env python3
"""
Debug the MODIS tile calculation to see if we're using the right tile.
"""

import math

def get_modis_tile_standard(lat, lon):
    """Standard MODIS tile calculation."""
    h = int((lon + 180) / 10)
    v = int((90 - lat) / 10)
    return f"h{h:02d}v{v:02d}"

def get_modis_tile_detailed(lat, lon):
    """More detailed MODIS tile calculation with bounds."""
    h = int((lon + 180) / 10)
    v = int((90 - lat) / 10)
    
    # Calculate tile bounds
    h_min = h * 10 - 180
    h_max = (h + 1) * 10 - 180
    v_min = 90 - (v + 1) * 10
    v_max = 90 - v * 10
    
    return f"h{h:02d}v{v:02d}", (h_min, h_max, v_min, v_max)

# Test points
test_points = [
    (30.3165, 78.0322, "Dehradun"),
    (30.0668, 79.0193, "Tehri"),
    (29.3803, 79.4636, "Nainital"),
    (30.7268, 79.0744, "Chamoli"),
]

print("MODIS Tile Calculation Debug")
print("=" * 60)

for lat, lon, name in test_points:
    tile, bounds = get_modis_tile_detailed(lat, lon)
    h_min, h_max, v_min, v_max = bounds
    
    print(f"\n{name} ({lat}, {lon}):")
    print(f"  Tile: {tile}")
    print(f"  Tile bounds: Lon {h_min:.2f} to {h_max:.2f}, Lat {v_min:.2f} to {v_max:.2f}")
    print(f"  Point within tile? {h_min <= lon <= h_max and v_min <= lat <= v_max}")

print("\n" + "=" * 60)
print("Checking if we need adjacent tiles...")

# Check if Dehradun might be in an adjacent tile
dehradun_lat, dehradun_lon = 30.3165, 78.0322

print(f"\nDehradun ({dehradun_lat}, {dehradun_lon}):")
print("Checking adjacent tiles:")

for h_offset in [-1, 0, 1]:
    for v_offset in [-1, 0, 1]:
        h = int((dehradun_lon + 180) / 10) + h_offset
        v = int((90 - dehradun_lat) / 10) + v_offset
        
        if h < 0 or h > 35 or v < 0 or v > 17:
            continue
            
        tile, bounds = get_modis_tile_detailed(dehradun_lat, dehradun_lon)
        h_min, h_max, v_min, v_max = bounds
        
        # Adjust bounds for offset
        h_min += h_offset * 10
        h_max += h_offset * 10
        v_min -= v_offset * 10
        v_max -= v_offset * 10
        
        tile_name = f"h{h:02d}v{v:02d}"
        within = h_min <= dehradun_lon <= h_max and v_min <= dehradun_lat <= v_max
        
        print(f"  {tile_name}: bounds ({h_min:.2f}, {h_max:.2f}, {v_min:.2f}, {v_max:.2f}) - {'✓' if within else '✗'}")

print("\n" + "=" * 60)
print("MODIS Tile System Reference:")
print("  - Tiles are 10° x 10°")
print("  - h tiles: 0-35 (longitude -180° to +180°)")
print("  - v tiles: 0-17 (latitude +90° to -90°)")
print("  - h25v05: longitude 70°-80°, latitude 40°-50°")
print("  - h26v05: longitude 80°-90°, latitude 40°-50°")
print("  - h25v06: longitude 70°-80°, latitude 30°-40°")
print("  - h26v06: longitude 80°-90°, latitude 30°-40°")