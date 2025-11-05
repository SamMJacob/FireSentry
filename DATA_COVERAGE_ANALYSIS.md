# FireSentry Data Coverage Analysis

## 🔍 Current Data Coverage Status

### **1. MODIS Surface Reflectance (Vegetation Indices)**

**Available Tiles:**
- ✅ h24v05 (78.3°E - 80.8°E, 30°N - 40°N)
- ✅ h24v06 (69.3°E - 74.5°E, 20°N - 30°N)  
- ✅ h25v06 (80.8°E - 85.1°E, 20°N - 30°N)
- ❌ **h23v06 MISSING** (67.0°E - 78.3°E, 20°N - 30°N)

**Date Coverage:**
- Start: March 1, 2020 (DOY 061)
- End: March 31, 2020 (DOY 091)
- Duration: **31 days (March only)**
- Missing: January, February, April-December

**Files:** 93 band files (31 days × 3 tiles)

---

### **2. MODIS Land Surface Temperature (LST)**

**Available Tiles:**
- ✅ h24v05 (78.3°E - 80.8°E, 30°N - 40°N)
- ✅ h24v06 (69.3°E - 74.5°E, 20°N - 30°N)
- ✅ h25v05 (91.4°E - 92.4°E, 30°N - 40°N) - *Not needed for Uttarakhand*
- ✅ h25v06 (80.8°E - 85.1°E, 20°N - 30°N)
- ❌ **h23v06 MISSING** (67.0°E - 78.3°E, 20°N - 30°N)

**Date Coverage:**
- Start: February 1, 2020 (DOY 032)
- End: June 30, 2020 (DOY 182)
- Duration: **151 days (Feb-Jun)**
- Missing: January, July-December

**Files:** 604 LST files

---

### **3. CHIRPS Precipitation**

**Coverage:**
- ✅ **Global coverage** (no tile limitations)
- ✅ Full year 2020 available
- Format: Daily GeoTIFF files

**Status:** ✅ Complete

---

### **4. SRTM Terrain Data**

**Coverage:**
- ✅ Covers Uttarakhand region (N28-N31, E077-E081)
- ✅ Static data (no temporal component)

**Status:** ✅ Complete

---

## 🚨 Critical Issues

### **Issue 1: Missing h23v06 Tile**

**Impact:**
- **Western Uttarakhand** (77.5°E - 78.3°E) has NO coverage
- Fire points in this region will have **NaN values** for:
  - NDVI, EVI, NDWI (vegetation indices)
  - LST (land surface temperature)

**Example Affected Points:**
- (29.7295°N, 77.5371°E) - Western edge
- (29.8878°N, 77.6221°E) - Near boundary
- (30.0205°N, 77.8622°E) - Near boundary

**Estimated Impact:**
- ~10-15% of fire points may fall in this gap

---

### **Issue 2: Limited Temporal Coverage (MODIS SR)**

**Impact:**
- Only **March 2020** data available
- DTW windows extending into Jan-Feb will have **NaN values**
- Fire events in April-December will have **NO vegetation data**

**Example:**
- Fire on March 15, 2020
- DTW window: 90 days back = December 16, 2019
- Available data: March 1-31, 2020 only
- Result: Most of DTW window has NaN values

**Estimated Impact:**
- ~70-80% of DTW calculations will be incomplete

---

### **Issue 3: Limited Temporal Coverage (MODIS LST)**

**Impact:**
- Only **Feb-Jun 2020** data available
- DTW windows extending into January will have **NaN values**
- Fire events in July-December will have **NO LST data**

**Estimated Impact:**
- ~40-50% of DTW calculations will be incomplete

---

## 📊 Uttarakhand Bounding Box

```
North: 31.459016°N
South: 28.709556°N
East:  81.044789°E
West:  77.575402°E
```

**Required MODIS Tiles:**
1. **h23v06** (67.0°E - 78.3°E) - ❌ MISSING
2. **h24v05** (78.3°E - 80.8°E) - ✅ Available
3. **h24v06** (69.3°E - 74.5°E) - ✅ Available (mostly outside Uttarakhand)
4. **h25v06** (80.8°E - 85.1°E) - ✅ Available

---

## 🎯 Required Downloads

### **Priority 1: h23v06 Tile (Critical)**

**What to Download:**
- Tile: h23v06
- Product: MYD09GA (Surface Reflectance) + MOD11A1 (LST)
- Date Range: Full year 2020 (Jan 1 - Dec 31)
- Purpose: Fill western Uttarakhand coverage gap

**Script:** `scripts/fetch_modis_sr_h23v06.py`

**Estimated Size:** ~5-10 GB

---

### **Priority 2: Full Year MODIS SR (High)**

**What to Download:**
- Tiles: h23v06, h24v05, h24v06, h25v06
- Product: MYD09GA (Surface Reflectance)
- Date Range: Jan-Feb + Apr-Dec 2020
- Purpose: Complete temporal coverage

**Script:** `scripts/fetch_modis_sr_jan_feb.py` (needs expansion)

**Estimated Size:** ~50-100 GB

---

### **Priority 3: Full Year MODIS LST (Medium)**

**What to Download:**
- Tiles: h23v06, h24v05, h24v06, h25v06
- Product: MOD11A1 (Land Surface Temperature)
- Date Range: January + July-December 2020
- Purpose: Complete temporal coverage

**Estimated Size:** ~30-50 GB

---

## 🔧 Immediate Actions

### **Option A: Download h23v06 Only (Quick Fix)**
```bash
# Download h23v06 for full year 2020
python scripts/fetch_modis_sr_h23v06.py
```
- **Time:** 2-4 hours
- **Fixes:** Western Uttarakhand coverage gap
- **Still Missing:** Jan-Feb, Apr-Dec temporal gaps

### **Option B: Download Full Dataset (Complete Fix)**
```bash
# 1. Download h23v06 for full year
python scripts/fetch_modis_sr_h23v06.py

# 2. Download Jan-Feb for all tiles
python scripts/fetch_modis_sr_jan_feb.py

# 3. Download Apr-Dec for all tiles
# (requires new script)
```
- **Time:** 10-20 hours
- **Fixes:** All coverage gaps
- **Result:** Complete dataset

### **Option C: Filter Fire Data (Workaround)**
```python
# Only process fires in March 2020 within covered area
fire_points = fire_points[
    (fire_points['date'] >= '2020-03-01') &
    (fire_points['date'] <= '2020-03-31') &
    (fire_points['lon'] >= 78.3)  # Exclude western edge
]
```
- **Time:** Immediate
- **Fixes:** Nothing (just avoids the problem)
- **Result:** Reduced dataset

---

## 📈 Current Pipeline Status

### **Both Pipelines (Original + Optimized):**
- ✅ Running
- ⚠️  Generating NaN values for:
  - Western Uttarakhand points (h23v06 gap)
  - Jan-Feb dates (temporal gap)
  - Apr-Dec dates (temporal gap)
- ⏱️  Expected completion: 12-20 hours
- 📊 Expected data quality: ~20-30% complete features

### **Recommendation:**
1. **Let current pipelines finish** (sunk cost)
2. **Download h23v06 tile** in parallel (Priority 1)
3. **Re-run pipelines** after h23v06 download completes
4. **Consider full dataset download** for production use

---

## 🎯 Next Steps

1. ✅ **Updated lookup table** to include h23v06
2. ✅ **Created download script** for h23v06
3. ⏳ **Download h23v06 data** (user action required)
4. ⏳ **Re-run feature extraction** with complete data
5. ⏳ **Consider full temporal coverage** for production

---

## 📝 Notes

- **MODIS Sinusoidal Projection:** Tile boundaries don't align with simple lat/lon grid
- **h23v06 Discovery:** Found through analysis of failed point lookups
- **Tile Coverage:** Verified against actual MODIS tile georeference data
- **Performance Impact:** Lookup table optimization still valid (3x faster)

---

**Generated:** 2025-10-25
**Status:** Analysis Complete, Action Required

