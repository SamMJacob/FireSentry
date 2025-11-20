# FireSentry Complete Setup Guide 🚀

Complete step-by-step guide to set up and run `build_features_5000.py` from scratch.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Environment Setup (.env)](#environment-setup-env)
4. [NASA Earthdata Authentication](#nasa-earthdata-authentication)
5. [Data Downloads](#data-downloads)
   - [Manual Downloads](#manual-downloads)
   - [Automated Downloads](#automated-downloads)
6. [Running the Pipeline](#running-the-pipeline)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended) or Windows 10+
- **RAM**: 16GB minimum (32GB recommended for full pipeline)
- **CPU**: 4+ cores (6+ cores recommended)
- **Storage**: 100GB+ free space (for raw data + processed features)
- **Python**: 3.9 or 3.10

### Required Accounts
1. **NASA Earthdata Account** (Free)
   - Sign up at: https://urs.earthdata.nasa.gov/register
   - Required for downloading MODIS satellite data
   
2. **FIRMS Account** (Optional - manual download available)
   - Sign up at: https://firms.modaps.eosdis.nasa.gov/
   - Get API token for automated downloads

---

## Installation

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd Mini_Project
```

### Step 2: Install System Dependencies

**Ubuntu/Debian:**
```bash
# GDAL (for geospatial processing)
sudo apt-get update
sudo apt-get install -y gdal-bin libgdal-dev python3-gdal

# Other system dependencies
sudo apt-get install -y build-essential python3-dev
```

**Windows:**
```bash
# Download GDAL from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal
# Install using pip after downloading .whl file
pip install GDAL-*.whl
```

### Step 3: Install Python Dependencies
```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install parquet support (CRITICAL - prevents 12-hour failures!)
pip install fastparquet
# OR
pip install pyarrow
```

### Step 4: Verify Installation
```bash
# Test critical imports
python -c "import pandas, numpy, sklearn, rasterio, earthaccess; print('✅ All imports successful')"

# Test parquet support (CRITICAL!)
python -c "import pandas as pd; df = pd.DataFrame({'test': [1,2,3]}); df.to_parquet('test.parquet'); print('✅ Parquet support confirmed')"
```

---

## Environment Setup (.env)

### Step 1: Create .env File
```bash
# Copy template
cp env.example .env

# Edit .env file
nano .env  # or use your preferred editor
```

### Step 2: Fill in Required Values

**Required Variables:**
```bash
# NASA Earthdata Credentials (REQUIRED)
EARTHDATA_USER=your_earthdata_username
EARTHDATA_PASS=your_earthdata_password

# FIRMS API Token (OPTIONAL - manual download available)
FIRMS_MAP_KEY=your_firms_token_if_available

# Geographic Bounds (Uttarakhand, India) - Already set correctly
BBOX_N=31.459016
BBOX_S=28.709556
BBOX_E=81.044789
BBOX_W=77.575402

# Data Time Range
YEAR_START=2020
YEAR_END=2024
```

**Optional Variables (for production training):**
```bash
# Model Training Parameters
MSFS_N_REPEATS=10
AUTOSKLEARN_TIME_LIMIT=14400
AUTOSKLEARN_PER_RUN_LIMIT=300
AUTOSKLEARN_ENSEMBLE_SIZE=50
AUTOSKLEARN_ENSEMBLE_NBEST=10
```

### Step 3: Verify .env File
```bash
# Test that .env is loaded correctly
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('EARTHDATA_USER:', os.getenv('EARTHDATA_USER', 'NOT SET'))"
```

---

## NASA Earthdata Authentication

NASA Earthdata requires authentication via `.netrc` file for automated downloads.

### Ubuntu/Linux/Mac:

**Method 1: Using .env file (Automated)**
```bash
# The scripts will automatically create .netrc from .env
# Just make sure .env has EARTHDATA_USER and EARTHDATA_PASS set
```

**Method 2: Manual .netrc Setup**
```bash
# Create .netrc file in home directory
cat >> ~/.netrc << EOF
machine urs.earthdata.nasa.gov
login YOUR_EARTHDATA_USERNAME
password YOUR_EARTHDATA_PASSWORD
EOF

# Set proper permissions (CRITICAL - must be 600)
chmod 600 ~/.netrc

# Verify
cat ~/.netrc
```

### Windows:

**Method 1: Using .env file (Automated)**
```bash
# Scripts will use .env credentials automatically
# Make sure .env has EARTHDATA_USER and EARTHDATA_PASS
```

**Method 2: Manual _netrc Setup**
```bash
# On Windows, create _netrc file (note: underscore, not dot)
# Location: C:\Users\YourUsername\_netrc

# Create file with this content:
machine urs.earthdata.nasa.gov
login YOUR_EARTHDATA_USERNAME
password YOUR_EARTHDATA_PASSWORD
```

**Method 3: Using netsh (Windows Alternative)**
```bash
# Windows can use netsh for credential storage
# But .env method is preferred and works cross-platform
```

### Verify Authentication
```bash
# Test authentication
python -c "
from dotenv import load_dotenv
import os
import earthaccess

load_dotenv()
username = os.getenv('EARTHDATA_USER')
password = os.getenv('EARTHDATA_PASS')

if username and password:
    os.environ['EARTHDATA_USER'] = username
    os.environ['EARTHDATA_PASS'] = password
    auth = earthaccess.login()
    if auth:
        print('✅ Earthdata authentication successful!')
    else:
        print('❌ Authentication failed - check credentials')
else:
    print('❌ EARTHDATA_USER and EARTHDATA_PASS not set in .env')
"
```

---

## Data Downloads

### Directory Structure

The pipeline expects this directory structure:
```
data/
├── raw/
│   ├── chirps/          # CHIRPS precipitation data
│   ├── firms/           # FIRMS fire data
│   ├── modis_lst/       # MODIS Land Surface Temperature
│   ├── modis_sr/        # MODIS Surface Reflectance
│   └── srtm/            # SRTM elevation data
└── processed/           # Generated feature matrix
```

### Manual Downloads

These datasets require manual download:

#### 1. FIRMS Fire Data (REQUIRED)

**What it is:** NASA's database of actual fire locations (ground truth)

**Download Steps:**
1. Visit: https://firms.modaps.eosdis.nasa.gov/download/
2. Select **"MODIS C6.1"** (not VIIRS)
3. **Draw bounding box** over Uttarakhand region:
   - North: 31.5°N
   - South: 28.7°N
   - East: 81.0°E
   - West: 77.5°E
4. **Date range**: 2020-01-01 to 2024-12-31
5. **Format**: CSV
6. **Save as**: `data/raw/firms/firms_uttarakhand_2020_2025.csv`

**Verify:**
```bash
# Check file exists and has data
head -5 data/raw/firms/firms_uttarakhand_2020_2025.csv
# Should show columns: latitude, longitude, acq_date, confidence, etc.
```

#### 2. SRTM Elevation Data (REQUIRED)

**What it is:** Shuttle Radar Topography Mission - elevation/terrain data

**Download Steps:**

**Option A: Download from USGS (Recommended)**
1. Visit: https://earthexplorer.usgs.gov/
2. Search for "SRTM 1 Arc-Second Global" (30m resolution)
3. **Select area**: Uttarakhand bounding box (same as above)
4. **Download tiles**: 
   - N28E077 (if available)
   - N29E077, N29E078, N29E079, N29E080, N29E081
   - N30E077, N30E078, N30E079, N30E080, N30E081
   - N31E077, N31E078, N31E079, N31E080, N31E081
5. **Save to**: `data/raw/srtm/`

**Option B: Download from AWS (Automated)**
```bash
# SRTM tiles are available on AWS S3
# You can use a script to download them automatically
# Check if scripts/download_srtm.py exists
```

**Verify:**
```bash
# Check SRTM files exist
ls -lh data/raw/srtm/
# Should see .tif or .hgt files
```

### Automated Downloads

These scripts handle downloads automatically:

#### 1. CHIRPS Precipitation Data

**What it is:** Daily precipitation estimates (global coverage)

**Download:**
```bash
# Download CHIRPS data for 2020-2024
python scripts/fetch_chirps.py

# Or download specific years
python scripts/fetch_chirps.py --year 2020
python scripts/fetch_chirps.py --year 2021
# ... etc
```

**Expected Output:**
- Files: `data/raw/chirps/2020/chirps-v2.0.2020.061.tif` (one per day)
- Size: ~50-100MB per year
- Time: 10-30 minutes per year

**Verify:**
```bash
# Check CHIRPS files
ls data/raw/chirps/2020/ | head -10
# Should see .tif files with dates
```

#### 2. MODIS Surface Reflectance (Vegetation Indices)

**What it is:** Vegetation health data (NDVI, EVI, NDWI)

**Download:**
```bash
# Complete download (Jan-Jun 2020-2024) - RECOMMENDED
python scripts/fetch_modis_sr_complete.py

# Or download specific years/months
python scripts/fetch_modis_sr.py
```

**Expected Output:**
- Files: `data/raw/modis_sr/2020/MYD09GA.AA2020061.h24v05.061.B01.tif`
- Size: ~40-60GB total
- Time: 6-10 hours (can run overnight)
- **Resumable**: Skips existing files

**Important Notes:**
- Requires NASA Earthdata authentication (see above)
- Downloads 3 tiles: h24v05, h24v06, h25v06
- Downloads 6 months: Jan-Jun (dry season + DTW lookback)

**Verify:**
```bash
# Check MODIS SR files
ls data/raw/modis_sr/2020/ | head -10
# Should see .tif files with band codes (B01, B02, etc.)
```

#### 3. MODIS Land Surface Temperature (LST)

**What it is:** Temperature data from space

**Download:**
```bash
# Download MODIS LST data
python scripts/fetch_modis_lst.py
```

**Expected Output:**
- Files: `data/raw/modis_lst/2020/MOD11A1.AA2020061.h24v05.061.LST_Day_1km.tif`
- Size: ~10-20GB total
- Time: 2-4 hours

**Verify:**
```bash
# Check MODIS LST files
ls data/raw/modis_lst/2020/ | head -10
```

---

## Running the Pipeline

### Step 1: Verify All Data is Present

```bash
# Quick check script
python -c "
from pathlib import Path

checks = {
    'FIRMS': Path('data/raw/firms/firms_uttarakhand_2020_2025.csv').exists(),
    'SRTM': len(list(Path('data/raw/srtm').glob('*.tif'))) > 0 or len(list(Path('data/raw/srtm').glob('*.hgt'))) > 0,
    'CHIRPS': Path('data/raw/chirps/2020').exists() and len(list(Path('data/raw/chirps/2020').glob('*.tif'))) > 0,
    'MODIS SR': Path('data/raw/modis_sr/2020').exists() and len(list(Path('data/raw/modis_sr/2020').glob('*.tif'))) > 0,
    'MODIS LST': Path('data/raw/modis_lst/2020').exists() and len(list(Path('data/raw/modis_lst/2020').glob('*.tif'))) > 0,
}

print('Data Availability Check:')
for name, exists in checks.items():
    status = '✅' if exists else '❌'
    print(f'{status} {name}: {\"Found\" if exists else \"Missing\"}')

if all(checks.values()):
    print('\n✅ All data present! Ready to run pipeline.')
else:
    print('\n❌ Missing data. Please download missing datasets.')
"
```

### Step 2: Run the Build Script

**First Time (Full Pipeline):**
```bash
# Build everything from scratch
python scripts/build_features_5000.py --force

# This will:
# 1. Test parquet support (prevents 12-hour failures)
# 2. Build feature matrix (5000 fire + 5000 pseudo points)
# 3. Train model with temporal split
# 4. Evaluate model
# 5. Generate presentation
```

**Resume from Existing Features:**
```bash
# If features_5000.parquet exists, skip feature building
python scripts/build_features_5000.py

# This will:
# 1. Load existing features
# 2. Train model (if not exists)
# 3. Evaluate model (if not exists)
```

**Just Retrain Model:**
```bash
# Delete old model and retrain
rm models/model.joblib
python scripts/build_features_5000.py

# This will:
# 1. Load existing features
# 2. Retrain model with temporal split
# 3. Re-evaluate
```

### Step 3: Monitor Progress

The script will show:
- ✅ Parquet test results
- 📊 Feature extraction progress (parallel processing)
- 🤖 Model training progress (Auto-sklearn)
- 📈 Evaluation results
- 💾 Memory usage (if psutil installed)

**Expected Timeline:**
- Feature building: 2-3 hours (with parallel processing)
- Model training: 1-2 hours (Auto-sklearn)
- Evaluation: 5-10 minutes
- **Total: 3-5 hours**

### Step 4: Check Results

```bash
# Check output files
ls -lh data/processed/features_5000.parquet
ls -lh models/model.joblib
ls -lh data/processed/evaluation_5000.json

# View evaluation results
cat data/processed/evaluation_5000.json
```

---

## Troubleshooting

### Common Issues

#### 1. "Parquet library not available"
**Error:**
```
❌ PARQUET LIBRARY MISSING!
```

**Solution:**
```bash
pip install fastparquet
# OR
pip install pyarrow
```

#### 2. "No .netrc file found"
**Error:**
```
No .netrc file found. Please create one...
```

**Solution:**
```bash
# Make sure .env has EARTHDATA_USER and EARTHDATA_PASS
# Then scripts will auto-create .netrc
# OR manually create:
cat >> ~/.netrc << EOF
machine urs.earthdata.nasa.gov
login YOUR_USERNAME
password YOUR_PASSWORD
EOF
chmod 600 ~/.netrc
```

#### 3. "Authentication failed"
**Error:**
```
❌ Authentication failed
```

**Solution:**
- Verify credentials in `.env` file
- Check NASA Earthdata account is active
- Try logging in at: https://urs.earthdata.nasa.gov/
- Verify `.netrc` file permissions (must be 600)

#### 4. "No MODIS files found"
**Error:**
```
No MODIS SR files found for...
```

**Solution:**
- Check MODIS data is downloaded: `ls data/raw/modis_sr/2020/`
- Verify tile names match: h24v05, h24v06, h25v06
- Re-download missing data: `python scripts/fetch_modis_sr_complete.py`

#### 5. "System crashes during parallel processing"
**Error:**
```
System restart / Out of memory
```

**Solution:**
- Reduce parallel workers: Edit `build_features_5000.py`, change `self.n_jobs = 4` to `self.n_jobs = 2`
- Close other applications
- Monitor memory: `watch -n 1 free -h`

#### 6. "Feature extraction takes too long"
**Solution:**
- Enable parallel processing (already enabled by default)
- Check you have 4+ CPU cores
- Reduce number of points: Edit sampling from 5000 to 1000 for testing

#### 7. "Model training fails with 'No runs available'"
**Error:**
```
[WARNING] No runs were available to build an ensemble from
```

**Solution:**
- Increase memory limit: Edit `model/train.py`, change `memory_limit=6144` to `memory_limit=8192`
- Reduce time limits if system is slow
- Check data quality: `python -c "import pandas as pd; df = pd.read_parquet('data/processed/features_5000.parquet'); print(df.isnull().sum())"`

#### 8. "Missing FIRMS data"
**Error:**
```
FileNotFoundError: firms_uttarakhand_2020_2025.csv
```

**Solution:**
- Download FIRMS data manually (see Manual Downloads section)
- Save to: `data/raw/firms/firms_uttarakhand_2020_2025.csv`
- Verify CSV has columns: latitude, longitude, acq_date

#### 9. "Missing SRTM data"
**Error:**
```
No SRTM files found
```

**Solution:**
- Download SRTM tiles manually (see Manual Downloads section)
- Place in: `data/raw/srtm/`
- Verify files are .tif or .hgt format

#### 10. "Windows path issues"
**Error:**
```
Path not found / Permission denied
```

**Solution:**
- Use forward slashes in paths: `data/raw/chirps` not `data\raw\chirps`
- Run as administrator if needed
- Check file permissions

### Getting Help

**Check Logs:**
```bash
# View build log
tail -f build_5000.log

# Check for errors
grep -i error build_5000.log
```

**Verify Environment:**
```bash
# Check Python version
python --version  # Should be 3.9 or 3.10

# Check dependencies
pip list | grep -E "pandas|numpy|sklearn|rasterio|earthaccess"

# Check disk space
df -h  # Linux/Mac
# OR
dir  # Windows
```

**Test Individual Components:**
```bash
# Test data loading
python -c "import pandas as pd; df = pd.read_csv('data/raw/firms/firms_uttarakhand_2020_2025.csv'); print(f'Loaded {len(df)} fire points')"

# Test MODIS access
python -c "import earthaccess; auth = earthaccess.login(); print('✅ MODIS access OK' if auth else '❌ MODIS access failed')"

# Test geospatial tools
python -c "import rasterio; print('✅ Rasterio OK')"
```

---

## Quick Reference

### Essential Commands

```bash
# Setup
cp env.example .env
nano .env  # Edit with your credentials

# Download data
python scripts/fetch_chirps.py
python scripts/fetch_modis_sr_complete.py
python scripts/fetch_modis_lst.py

# Run pipeline
python scripts/build_features_5000.py

# Demo/presentation
python demo_for_presentation.py
```

### File Locations

```
data/processed/features_5000.parquet    # Feature matrix
models/model.joblib                     # Trained model
data/processed/evaluation_5000.json    # Evaluation results
presentation_plots/                     # Demo visualizations
build_5000.log                         # Build log
```

### Expected Sizes

- Features file: ~50-100 MB
- Model file: ~100-500 MB
- Raw data: ~100 GB total
- Processed data: ~1 GB

---

## Summary Checklist ✅

Before running `build_features_5000.py`, ensure:

- [ ] Python 3.9+ installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Parquet support installed (`pip install fastparquet` or `pyarrow`)
- [ ] `.env` file created with `EARTHDATA_USER` and `EARTHDATA_PASS`
- [ ] NASA Earthdata authentication working (test with `earthaccess.login()`)
- [ ] FIRMS data downloaded to `data/raw/firms/firms_uttarakhand_2020_2025.csv`
- [ ] SRTM data in `data/raw/srtm/`
- [ ] CHIRPS data downloaded (2020-2024)
- [ ] MODIS SR data downloaded (Jan-Jun 2020-2024)
- [ ] MODIS LST data downloaded (2020-2024)
- [ ] At least 16GB RAM available
- [ ] At least 100GB disk space free

Once all items are checked, you're ready to run:
```bash
python scripts/build_features_5000.py
```

**Good luck! 🚀**

