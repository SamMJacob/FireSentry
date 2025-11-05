#!/usr/bin/env python3
"""
Check which MODIS tiles our test points fall into.
"""

def get_modis_tile(lat, lon):
    """Calculate MODIS tile for given coordinates."""
    h = int((lon + 180) / 10)
    v = int((90 - lat) / 10)
    return f"h{h:02d}v{v:02d}"

# Test points
test_points = [
    (30.3165, 78.0322, "Dehradun"),
    (30.0668, 79.0193, "Tehri"),
    (29.3803, 79.4636, "Nainital"),
    (30.7268, 79.0744, "Chamoli"),
]

print("Test Point Tile Coverage:")
print("=" * 60)

tiles_needed = set()
for lat, lon, name in test_points:
    tile = get_modis_tile(lat, lon)
    tiles_needed.add(tile)
    print(f"{name:15} ({lat:7.4f}, {lon:7.4f}) -> {tile}")

print("\n" + "=" * 60)
print(f"Tiles needed: {sorted(tiles_needed)}")
print(f"Tiles available: ['h24v05', 'h24v06', 'h25v05', 'h25v06']")

print("\nConclusion:")
if tiles_needed.issubset({'h24v05', 'h24v06', 'h25v05', 'h25v06'}):
    print("✓ All test points are covered by available tiles!")
    print("\nThe issue is likely with:")
    print("1. Coordinate transformation in the extraction code")
    print("2. Clipping that removed data coverage")
    print("3. File pattern matching")
else:
    print("✗ Some test points need tiles that are not available")
    missing = tiles_needed - {'h24v05', 'h24v06', 'h25v05', 'h25v06'}
    print(f"Missing tiles: {missing}")

# Check Uttarakhand bounding box
print("\n" + "=" * 60)
print("Uttarakhand Bounding Box Tile Coverage:")
uttarakhand_bbox = {
    'north': 31.44,
    'south': 28.72,
    'east': 81.02,
    'west': 77.57
}

corners = [
    (uttarakhand_bbox['north'], uttarakhand_bbox['west'], "NW"),
    (uttarakhand_bbox['north'], uttarakhand_bbox['east'], "NE"),
    (uttarakhand_bbox['south'], uttarakhand_bbox['west'], "SW"),
    (uttarakhand_bbox['south'], uttarakhand_bbox['east'], "SE"),
]

bbox_tiles = set()
for lat, lon, corner in corners:
    tile = get_modis_tile(lat, lon)
    bbox_tiles.add(tile)
    print(f"{corner:3} corner ({lat:7.4f}, {lon:7.4f}) -> {tile}")

print(f"\nBounding box spans tiles: {sorted(bbox_tiles)}")
print(f"Tiles available in NASA: ['h24v05', 'h24v06', 'h25v05', 'h25v06']")

if bbox_tiles.issubset({'h24v05', 'h24v06', 'h25v05', 'h25v06'}):
    print("\n✓ All of Uttarakhand is covered by available tiles!")
else:
    missing = bbox_tiles - {'h24v05', 'h24v06', 'h25v05', 'h25v06'}
    print(f"\n✗ Missing tiles for complete coverage: {missing}")


