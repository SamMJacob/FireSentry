#!/usr/bin/env python3
"""
Test with a point that's within the clipped MODIS SR bounds
"""

from datetime import datetime
from pathlib import Path
import rasterio
from rasterio.warp import transform
from rasterio.crs import CRS
import numpy as np

# Test with a point within the clipped MODIS SR bounds
# Clipped bounds: 80.8290°E to 83.3331°E, 30.0000°N to 31.5042°N
lat, lon = 31.0, 81.5  # Within clipped bounds
date = datetime(2020, 3, 1)  # DOY 61 (known to have data)

print(f"Testing with point within clipped bounds:")
print(f"  Point: ({lat}°N, {lon}°E)")
print(f"  Date: {date.date()}")
print("="*60)

# Calculate tile
h = int((lon + 180) / 10)
v = int((90 - lat) / 10)
tile = f"h{h:02d}v{v:02d}"
print(f"Calculated tile: {tile}")

# Test vegetation indices
print(f"\n1. Testing vegetation indices:")
from features.indices import VegetationIndices
vi = VegetationIndices()

# Test single date
indices = vi.calculate_indices_for_date(lat, lon, date)
print(f"   NDVI: {indices['ndvi']}")
print(f"   EVI: {indices['evi']}")
print(f"   NDWI: {indices['ndwi']}")

if not all(np.isnan([indices['ndvi'], indices['evi'], indices['ndwi']])):
    print(f"   ✅ Vegetation indices working!")
else:
    print(f"   ❌ Vegetation indices still returning NaN")

# Test DTW features
print(f"\n2. Testing DTW features:")
from features.dtw import DynamicTimeWindow
dtw = DynamicTimeWindow(thcp=30.0, thdp=10.0, max_window_days=90)

# Calculate DTW window
fire_date = datetime(2020, 3, 15)  # Use a fire date
dtw_start, dtw_end = dtw.calculate_dtw(
    fire_date, lat, lon, "data/raw/chirps"
)

if dtw_start and dtw_end:
    print(f"   DTW window: {dtw_start.date()} to {dtw_end.date()}")
    
    # Test vegetation DTW features
    vi_features = vi.extract_dtw_features(lat, lon, dtw_start, dtw_end)
    print(f"   Vegetation DTW features:")
    for feature, value in vi_features.items():
        if not np.isnan(value):
            print(f"     {feature}: {value:.4f}")
        else:
            print(f"     {feature}: NaN")
    
    # Test LST features
    print(f"\n3. Testing LST features:")
    from features.lst import LSTFeatures
    lst = LSTFeatures()
    
    lst_features = lst.extract_dtw_features(lat, lon, dtw_start, dtw_end)
    print(f"   LST features:")
    for feature, value in lst_features.items():
        if not np.isnan(value):
            print(f"     {feature}: {value:.2f} K")
        else:
            print(f"     {feature}: NaN")
    
    # Test terrain features
    print(f"\n4. Testing terrain features:")
    from features.terrain import TerrainFeatures
    tf = TerrainFeatures()
    
    terrain_features = tf.extract_terrain_features(lat, lon)
    print(f"   Terrain features:")
    for feature, value in terrain_features.items():
        print(f"     {feature}: {value:.2f}")
    
    # Test precipitation features
    print(f"\n5. Testing precipitation features:")
    precip_series = dtw.get_precipitation_series(
        lat, lon, dtw_start, dtw_end, "data/raw/chirps"
    )
    if len(precip_series) > 0:
        precip_features = {
            'prec_min': float(precip_series.min()),
            'prec_median': float(precip_series.median()),
            'prec_mean': float(precip_series.mean()),
            'prec_max': float(precip_series.max()),
            'prec_sum': float(precip_series.sum())
        }
        print(f"   Precipitation features:")
        for feature, value in precip_features.items():
            print(f"     {feature}: {value:.2f} mm")
    else:
        print(f"   No precipitation data")
    
    # Test complete pipeline
    print(f"\n6. Testing complete pipeline:")
    from features.pipeline import FeaturePipeline
    pipeline = FeaturePipeline()
    
    all_features = pipeline.extract_all_features(lat, lon, dtw_start, dtw_end)
    print(f"   Total features: {len(all_features)}")
    
    # Count non-NaN features
    non_nan_count = sum(1 for v in all_features.values() if not np.isnan(v))
    print(f"   Non-NaN features: {non_nan_count}/24")
    
    if non_nan_count >= 20:  # Allow for some missing data
        print(f"   ✅ Pipeline working with good test point!")
    else:
        print(f"   ❌ Still missing too many features")
        
    # Show all features
    print(f"\n   All features:")
    for feature, value in all_features.items():
        if not np.isnan(value):
            print(f"     {feature}: {value:.4f}")
        else:
            print(f"     {feature}: NaN")
else:
    print(f"   ❌ Could not calculate DTW window")


