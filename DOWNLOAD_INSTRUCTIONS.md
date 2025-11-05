# FireSentry Data Download Instructions

## 🎯 Summary

You have **100% spatial coverage** but need to download **temporal data** for the complete dry season.

## ✅ What You Have

- **Tiles**: h24v05, h24v06, h25v06 (100% Uttarakhand coverage - VERIFIED)
- **Current data**: March only for years 2020-2024
- **Coverage**: 11,399 fire points (100% spatial)

## 📥 What You Need to Download

### **Complete Dry Season Data**
- **Months**: January-June (dry season + DTW lookback)
- **Years**: 2020-2024
- **Tiles**: h24v05, h24v06, h25v06

## 🚀 Single Command to Download Everything

```bash
python scripts/fetch_modis_sr_complete.py
```

This ONE script downloads:
- ✅ January-February (for DTW lookback windows)
- ✅ March (will skip - already have it)
- ✅ April-June (rest of dry season)
- ✅ All years 2020-2024
- ✅ All 3 tiles with 100% coverage

## 📊 Download Details

### **What to Expect:**
- **Total months**: 6 months × 5 years = 30 month-years
- **Total granules**: ~900-1200 granules
- **Estimated size**: 40-60 GB
- **Estimated time**: 6-10 hours
- **Resumable**: Yes (skips existing files)

### **Progress Tracking:**
The script will show:
```
YEAR 2020
--- January 2020 ---
  [1/45] Downloading...
  [2/45] Downloading...
  ...
✅ January complete: 45 granules
--- February 2020 ---
  ...
```

## 🔧 After Download

### **1. Verify Data:**
```bash
# Check what you have
python3 << 'EOF'
import os
from datetime import datetime

for year in [2020, 2021, 2022, 2023, 2024]:
    files = [f for f in os.listdir(f'data/raw/modis_sr/{year}') if f.endswith('.B01.tif')]
    if files:
        doys = sorted(set([int(f.split('.')[1][6:9]) for f in files]))
        start = datetime.strptime(f'{year}{doys[0]:03d}', '%Y%j')
        end = datetime.strptime(f'{year}{doys[-1]:03d}', '%Y%j')
        print(f"{year}: {start.strftime('%b %d')} - {end.strftime('%b %d')} ({len(doys)} days)")
EOF
```

**Expected output:**
```
2020: Jan 01 - Jun 30 (181 days)
2021: Jan 01 - Jun 30 (181 days)
2022: Jan 01 - Jun 30 (181 days)
2023: Jan 01 - Jun 30 (181 days)
2024: Jan 01 - Jun 30 (182 days)
```

### **2. Run Feature Extraction:**
```bash
# Use the optimized version with fixed tile lookup
python scripts/build_features_optimized.py
```

### **3. Expected Results:**
- ✅ **100% spatial coverage** (all 11,399 fire points)
- ✅ **Complete temporal coverage** (Jan-Jun for all years)
- ✅ **All 24 features** with valid data
- ✅ **No NaN values** (except for genuinely missing satellite data)

## 📝 Important Notes

### **Tile Coverage (VERIFIED):**
- **h24v05**: 69.28°E - 91.38°E, 30°N - 40°N (covers 4,106 fires)
- **h24v06**: 63.85°E - 80.83°E, 20°N - 30°N (covers 7,293 fires)
- **h25v06**: 74.49°E - 92.38°E, 20°N - 30°N (backup, 0 fires)

These bounds are from **actual MODIS files**, not estimates.

### **Why These Months:**
- **Jan-Feb**: DTW lookback windows (fires in March look back 90 days)
- **Mar-Jun**: Dry season (actual fire occurrence period)

### **What About July-December:**
Not needed! Your training focuses on dry season fires only.

## ⚠️ Troubleshooting

### **If Download Fails:**
- Script is resumable - just run it again
- Already downloaded files will be skipped
- Check NASA Earthdata credentials in ~/.netrc

### **If Running Out of Space:**
- Each month is ~2-3 GB
- Total: 40-60 GB needed
- Can download year-by-year if needed

### **If Too Slow:**
- Expected: 6-10 hours for complete download
- Rate limited to 1 second between granules
- Running overnight is recommended

## ✅ Verification Checklist

After download completes:

- [ ] All years (2020-2024) have data
- [ ] Each year has Jan-Jun coverage
- [ ] Each month has ~30 days of data
- [ ] All 3 tiles present for each day
- [ ] Total size: 40-60 GB
- [ ] Manifest file created: `data/raw/modis_sr/download_manifest_complete.json`

## 🎉 Ready to Go!

Once download completes, you'll have:
- ✅ Complete spatial coverage (100% of Uttarakhand)
- ✅ Complete temporal coverage (Jan-Jun 2020-2024)
- ✅ All data needed for training
- ✅ No additional downloads required

---

**Questions?** Check `COVERAGE_ISSUE_RESOLVED.md` for details on how we verified 100% coverage.

