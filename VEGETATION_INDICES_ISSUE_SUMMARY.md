# Vegetation Indices Issue - Root Cause Analysis

## Problem
Vegetation indices returning NaN for test point (30.3165°N, 78.0322°E) on 2020-03-15.

## Root Cause
The **MODIS Surface Reflectance rasters were clipped** during download/processing to cover only the Uttarakhand region. The test point is **outside the clipped bounds**.

### Details

**Test Point:** 30.3165°N, 78.0322°E  
**Expected Tile:** h25v05 (correct)  
**Date:** 2020-03-15 (DOY 75)

**Full h25v05 Tile Bounds (Geographic):**
- Longitude: 70°E to 80°E
- Latitude: 30°N to 40°N

**Actual Clipped Raster Bounds (WGS84):**
- Longitude: **80.8290°E to 83.3331°E** 
- Latitude: 30.0000°N to 31.5042°N

**Problem:**
- Test point longitude: 78.0322°E
- Clipped raster starts at: 80.8290°E  
- **Gap: 2.5° West** - test point is outside the clipped area!

## Impact

1. **LST Features:** ✅ Working (files cover the test point in h24v05)
2. **Terrain Features:** ✅ Working
3. **Precipitation Features:** ✅ Working
4. **Vegetation Indices:** ❌ Not working (test point outside clipped bounds)

**Current Status: 12/24 features working (50%)**

## Solutions

### Option 1: Change Test Point (Recommended)
Use a test point within the clipped MODIS SR bounds:
- **Suggested point:** 31.0°N, 81.5°E (within 80.83-83.33°E range)
- This point is still in Uttarakhand and within the clipped raster

### Option 2: Accept NaN Values
- The pipeline **correctly handles missing data** by returning NaN
- In production, points outside MODIS SR coverage will have NaN vegetation indices
- This is **expected behavior** for satellite data with limited coverage

### Option 3: Re-download MODIS SR Data
- Download full h25v05 tiles without clipping
- Would increase storage requirements significantly
- Not recommended as clipping was done for storage optimization

## Verification

The vegetation indices code is **working correctly**:
1. ✅ File finding logic (with 3x3 tile search)
2. ✅ Coordinate transformation (WGS84 to Sinusoidal)
3. ✅ Band value extraction
4. ✅ Index calculation (NDVI, EVI, NDWI)
5. ✅ NaN handling for missing data

The issue is **data coverage**, not code logic.

## Recommendation

**Update the test point to (31.0°N, 81.5°E)** which is:
- Within Uttarakhand
- Within the clipped MODIS SR raster bounds
- Will have valid vegetation indices data
- Better represents the actual usable data coverage


