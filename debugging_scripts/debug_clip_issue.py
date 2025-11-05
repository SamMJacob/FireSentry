#!/usr/bin/env python3
"""
Debug why the clipping is removing our test point.
"""

from rasterio.warp import transform_bounds
from rasterio.crs import CRS

# Uttarakhand bbox
uttarakhand_bbox = (77.575402, 28.709556, 81.044789, 31.459016)
print(f"Uttarakhand bbox (WGS84): {uttarakhand_bbox}")
print(f"  West: {uttarakhand_bbox[0]}, South: {uttarakhand_bbox[1]}")
print(f"  East: {uttarakhand_bbox[2]}, North: {uttarakhand_bbox[3]}")
print()

# Test point
test_lon, test_lat = 78.0322, 30.3165
print(f"Test point (Dehradun): ({test_lon}, {test_lat})")
print(f"  Is within Uttarakhand bbox? {uttarakhand_bbox[0] <= test_lon <= uttarakhand_bbox[2] and uttarakhand_bbox[1] <= test_lat <= uttarakhand_bbox[3]}")
print()

# MODIS Sinusoidal CRS
modis_crs = CRS.from_string('PROJCS["unnamed",GEOGCS["Unknown datum based upon the custom spheroid",DATUM["Not specified (based on custom spheroid)",SPHEROID["Custom spheroid",6371007.181,0]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Sinusoidal"],PARAMETER["longitude_of_center",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]')

# Transform bbox
bbox_transformed = transform_bounds('EPSG:4326', modis_crs, *uttarakhand_bbox)
print(f"Uttarakhand bbox in MODIS Sinusoidal:")
print(f"  {bbox_transformed}")
print(f"  Left: {bbox_transformed[0]}, Bottom: {bbox_transformed[1]}")
print(f"  Right: {bbox_transformed[2]}, Top: {bbox_transformed[3]}")
print()

# Transform test point
from rasterio.warp import transform
test_x, test_y = transform(
    CRS.from_epsg(4326),
    modis_crs,
    [test_lon], [test_lat]
)
print(f"Test point in MODIS Sinusoidal: ({test_x[0]}, {test_y[0]})")
print(f"  Is within transformed bbox? {bbox_transformed[0] <= test_x[0] <= bbox_transformed[2] and bbox_transformed[1] <= test_y[0] <= bbox_transformed[3]}")
print()

# Clipped raster bounds (from previous debug)
clipped_bounds = (7783653.637667, 3335851.559, 7904114.94396415, 3498474.3225012985)
print(f"Clipped raster bounds (from file):")
print(f"  {clipped_bounds}")
print(f"  Left: {clipped_bounds[0]}, Bottom: {clipped_bounds[1]}")
print(f"  Right: {clipped_bounds[2]}, Top: {clipped_bounds[3]}")
print()

print("Issue Analysis:")
print(f"  Test point X: {test_x[0]}")
print(f"  Transformed bbox left: {bbox_transformed[0]}")
print(f"  Clipped raster left: {clipped_bounds[0]}")
print()
print(f"  Difference between bbox and clipped left: {clipped_bounds[0] - bbox_transformed[0]}")
print()

if test_x[0] < clipped_bounds[0]:
    print("❌ Test point is WEST of the clipped raster")
    print(f"   The clipping removed {(clipped_bounds[0] - test_x[0]) / 1000:.1f} km of data on the western edge!")
else:
    print("✓ Test point should be within clipped raster")


