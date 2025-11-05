# FireSentry Feature Extraction - Restart Instructions

## What Happened

The feature extraction got stuck in an infinite loop because:
- Pseudo fire points near the eastern edge of Uttarakhand (lon ~81.0°E) were outside CHIRPS coverage
- For each out-of-bounds point, the DTW algorithm checks ~90 days of data
- This caused the same warning to repeat 90 times per point
- The logging overwhelmed the process

## What Was Fixed

✅ Modified `features/dtw.py` to only warn ONCE per out-of-bounds point
✅ The algorithm still handles these points gracefully (returns 0.0 for precipitation)
✅ Feature extraction will now continue without warning spam

## How to Restart

### Step 1: Kill the Current Process
```bash
# Press Ctrl+C in the terminal running the script
# Or find and kill the process:
ps aux | grep build_features
kill <PID>
```

### Step 2: Clean Up Partial Results (Optional)
```bash
# If a partial features.parquet was created, remove it:
rm -f data/features.parquet
rm -f data/features_metadata.json
```

### Step 3: Restart Feature Extraction
```bash
python scripts/build_features.py --skip-training
```

## What to Expect

✅ **Reduced warnings**: You'll see warnings for out-of-bounds points, but only once per point (not 90 times!)
✅ **Progress**: The script will process ~22,248 points (11,124 fire + 11,124 pseudo)
✅ **Time**: Should take ~30-60 minutes depending on your system
✅ **Some missing features**: A small percentage of NDWI values will be NaN due to cloud cover (normal!)

## Expected Output

```
INFO:features.pipeline:Loaded 11124 fire points
INFO:features.pipeline:Generating pseudo fire points with ratio 1.0
WARNING:features.dtw:Point (30.788, 81.032) outside raster bounds  # Only ONCE per point!
INFO:features.dtw:DTW calculation complete: 21950 successful, 298 failed
INFO:features.pipeline:Extracting features for 21950 points...
...
✅ FireSentry build completed successfully!
```

## Monitoring Progress

Watch for these log messages to track progress:
- `Calculating DTW for X fire points` - DTW phase
- `Extracting features for X points...` - Feature extraction phase
- Progress updates every 100 points

## If It Gets Stuck Again

If you see the same warning repeating many times for the same point, the fix didn't work. In that case:
1. Stop the process (Ctrl+C)
2. Check that `features/dtw.py` has the `self._warned_points` set
3. Re-run the script

## Next Steps After Completion

Once feature extraction completes successfully:
```bash
# Analyze the results
python -c "
import pandas as pd
df = pd.read_parquet('data/features.parquet')
print(f'Dataset: {len(df)} samples')
print(f'Features: {len(df.columns)-1}')
print(df.info())
"

# Then proceed to model training
python scripts/build_features.py  # Without --skip-training
```


