#!/usr/bin/env python3
"""
Debug: Check if the MODIS Sinusoidal projection is causing the misalignment.
Convert the 10° tile boundaries to MODIS Sinusoidal to see where they should be.
"""

from rasterio.warp import transform
from rasterio.crs import CRS

# MODIS Sinusoidal CRS
modis_crs = CRS.from_string(
    'PROJCS["unnamed",GEOGCS["Unknown datum based upon the custom spheroid",'
    'DATUM["Not specified (based on custom spheroid)",'
    'SPHEROID["Custom spheroid",6371007.181,0]],'
    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],'
    'PROJECTION["Sinusoidal"],PARAMETER["longitude_of_center",0],'
    'PARAMETER["false_easting",0],PARAMETER["false_northing",0],'
    'UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
)

print("MODIS Projection Analysis")
print("=" * 70)
print()

# h25v05 should cover: 70°-80°E, 30°-40°N
print("h25v05 Tile - Geographic bounds: 70°-80°E, 30°-40°N")
print()

# Convert tile corners to MODIS Sinusoidal
corners = [
    (70.0, 30.0, "SW"),
    (80.0, 30.0, "SE"),
    (70.0, 40.0, "NW"),
    (80.0, 40.0, "NE"),
]

print("Tile corners in MODIS Sinusoidal projection:")
modis_bounds = {'left': float('inf'), 'right': float('-inf'), 
                'bottom': float('inf'), 'top': float('-inf')}

for lon, lat, corner in corners:
    x, y = transform(CRS.from_epsg(4326), modis_crs, [lon], [lat])
    print(f"  {corner} ({lon}°E, {lat}°N) -> ({x[0]:.2f}, {y[0]:.2f})")
    
    modis_bounds['left'] = min(modis_bounds['left'], x[0])
    modis_bounds['right'] = max(modis_bounds['right'], x[0])
    modis_bounds['bottom'] = min(modis_bounds['bottom'], y[0])
    modis_bounds['top'] = max(modis_bounds['top'], y[0])

print()
print(f"Expected tile bounds in MODIS Sinusoidal:")
print(f"  Left: {modis_bounds['left']:.2f}")
print(f"  Right: {modis_bounds['right']:.2f}")
print(f"  Bottom: {modis_bounds['bottom']:.2f}")
print(f"  Top: {modis_bounds['top']:.2f}")

print()
print("Actual downloaded tile bounds (from test):")
actual_bounds = {
    'left': 7783653.64,
    'right': 8895604.16,
    'bottom': 3335851.56,
    'top': 4447802.08
}
print(f"  Left: {actual_bounds['left']:.2f}")
print(f"  Right: {actual_bounds['right']:.2f}")
print(f"  Bottom: {actual_bounds['bottom']:.2f}")
print(f"  Top: {actual_bounds['top']:.2f}")

print()
print("=" * 70)
print("Comparison:")
print(f"  Left:   Expected {modis_bounds['left']:.2f}, Got {actual_bounds['left']:.2f}, Diff {actual_bounds['left'] - modis_bounds['left']:.2f} m")
print(f"  Right:  Expected {modis_bounds['right']:.2f}, Got {actual_bounds['right']:.2f}, Diff {actual_bounds['right'] - modis_bounds['right']:.2f} m")
print(f"  Bottom: Expected {modis_bounds['bottom']:.2f}, Got {actual_bounds['bottom']:.2f}, Diff {actual_bounds['bottom'] - modis_bounds['bottom']:.2f} m")
print(f"  Top:    Expected {modis_bounds['top']:.2f}, Got {actual_bounds['top']:.2f}, Diff {actual_bounds['top'] - modis_bounds['top']:.2f} m")

print()
print("=" * 70)

# Test Dehradun
dehradun_lon, dehradun_lat = 78.0322, 30.3165
x, y = transform(CRS.from_epsg(4326), modis_crs, [dehradun_lon], [dehradun_lat])

print(f"\nDehradun ({dehradun_lon}°E, {dehradun_lat}°N):")
print(f"  MODIS coords: ({x[0]:.2f}, {y[0]:.2f})")
print(f"  Within expected bounds? {modis_bounds['left'] <= x[0] <= modis_bounds['right'] and modis_bounds['bottom'] <= y[0] <= modis_bounds['top']}")
print(f"  Within actual bounds? {actual_bounds['left'] <= x[0] <= actual_bounds['right'] and actual_bounds['bottom'] <= y[0] <= actual_bounds['top']}")

print()
if actual_bounds['left'] > modis_bounds['left']:
    print("⚠️  ISSUE FOUND: The actual tile starts EAST of where it should!")
    print(f"   The tile is missing {(actual_bounds['left'] - modis_bounds['left'])/1000:.1f} km on the western edge.")
    print()
    print("   This could be because:")
    print("   1. NASA's MODIS tiles don't align perfectly with 10° geographic boundaries")
    print("   2. The tile was pre-processed/clipped by NASA")
    print("   3. We need to download MULTIPLE tiles and mosaic them")


