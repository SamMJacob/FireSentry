#!/usr/bin/env python3
"""
Verify if h23v06 tile will complete the western Uttarakhand coverage gap.
"""

# Uttarakhand bounds
uk_west = 77.575402
uk_east = 81.044789
uk_south = 28.709556
uk_north = 31.459016

print("="*70)
print("UTTARAKHAND COVERAGE VERIFICATION")
print("="*70)

print(f"\nUttarakhand Bounding Box:")
print(f"  West:  {uk_west:.3f}°E")
print(f"  East:  {uk_east:.3f}°E")
print(f"  South: {uk_south:.3f}°N")
print(f"  North: {uk_north:.3f}°N")

# Current tile coverage
tiles = {
    'h23v06': {'lon_min': 67.0, 'lon_max': 78.3, 'lat_min': 20.0, 'lat_max': 30.0},
    'h24v05': {'lon_min': 78.3, 'lon_max': 80.8, 'lat_min': 30.0, 'lat_max': 40.0},
    'h24v06': {'lon_min': 69.3, 'lon_max': 74.5, 'lat_min': 20.0, 'lat_max': 30.0},
    'h25v06': {'lon_min': 80.8, 'lon_max': 85.1, 'lat_min': 20.0, 'lat_max': 30.0},
}

print(f"\n{'='*70}")
print("TILE COVERAGE:")
print(f"{'='*70}")
for tile, bounds in sorted(tiles.items()):
    print(f"\n{tile}:")
    print(f"  Longitude: {bounds['lon_min']:.1f}°E - {bounds['lon_max']:.1f}°E")
    print(f"  Latitude:  {bounds['lat_min']:.1f}°N - {bounds['lat_max']:.1f}°N")

# Check longitude coverage
print(f"\n{'='*70}")
print("LONGITUDE COVERAGE ANALYSIS:")
print(f"{'='*70}")

print(f"\nUttarakhand needs: {uk_west:.3f}°E - {uk_east:.3f}°E")
print(f"\nTile coverage:")
print(f"  h23v06: 67.0°E - 78.3°E")
print(f"  h24v05: 78.3°E - 80.8°E")
print(f"  h25v06: 80.8°E - 85.1°E")

# Check if h23v06 covers western edge
if uk_west >= tiles['h23v06']['lon_min'] and uk_west <= tiles['h23v06']['lon_max']:
    print(f"\n✅ h23v06 COVERS western edge ({uk_west:.3f}°E)")
else:
    print(f"\n❌ h23v06 does NOT cover western edge ({uk_west:.3f}°E)")

# Check for gaps
if tiles['h23v06']['lon_max'] == tiles['h24v05']['lon_min']:
    print(f"✅ No gap between h23v06 and h24v05 (both at 78.3°E)")
else:
    gap = tiles['h24v05']['lon_min'] - tiles['h23v06']['lon_max']
    print(f"❌ Gap between h23v06 and h24v05: {gap:.1f}°")

if tiles['h24v05']['lon_max'] == tiles['h25v06']['lon_min']:
    print(f"✅ No gap between h24v05 and h25v06 (both at 80.8°E)")
else:
    gap = tiles['h25v06']['lon_min'] - tiles['h24v05']['lon_max']
    print(f"❌ Gap between h24v05 and h25v06: {gap:.1f}°")

# Check latitude coverage
print(f"\n{'='*70}")
print("LATITUDE COVERAGE ANALYSIS:")
print(f"{'='*70}")

print(f"\nUttarakhand needs: {uk_south:.3f}°N - {uk_north:.3f}°N")

# Check h23v06 latitude coverage
h23_covers_south = uk_south >= tiles['h23v06']['lat_min']
h23_covers_north = uk_north <= tiles['h23v06']['lat_max']

print(f"\nh23v06 covers: {tiles['h23v06']['lat_min']:.1f}°N - {tiles['h23v06']['lat_max']:.1f}°N")
print(f"  South boundary ({uk_south:.3f}°N): {'✅ Covered' if h23_covers_south else '❌ NOT covered'}")
print(f"  North boundary ({uk_north:.3f}°N): {'✅ Covered' if h23_covers_north else '❌ NOT covered (exceeds 30°N)'}")

# Check if we need additional tiles for northern part
if uk_north > tiles['h23v06']['lat_max']:
    missing_north = uk_north - tiles['h23v06']['lat_max']
    print(f"\n⚠️  WARNING: Northern Uttarakhand ({tiles['h23v06']['lat_max']:.1f}°N - {uk_north:.3f}°N) NOT covered by h23v06!")
    print(f"   Missing: {missing_north:.2f}° latitude")
    print(f"   Need tile: h23v05 (covers 30°N - 40°N)")

# Final verdict
print(f"\n{'='*70}")
print("FINAL VERDICT:")
print(f"{'='*70}")

if uk_west >= tiles['h23v06']['lon_min'] and uk_west <= tiles['h23v06']['lon_max']:
    if uk_north <= tiles['h23v06']['lat_max']:
        print("\n✅ h23v06 ALONE will complete the coverage")
    else:
        print("\n⚠️  h23v06 will help but NOT complete the coverage")
        print("   Also need: h23v05 for northern part (30°N - 31.5°N)")
else:
    print("\n❌ h23v06 will NOT help - wrong longitude range")

print(f"\n{'='*70}")

