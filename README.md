# FireSentry: Forest Fire Risk Prediction for Uttarakhand

An AI-driven early warning system for forest fire risk prediction in Uttarakhand, India, using Multi-Stage Feature Selection (MSFS) and AutoML.

## Overview

FireSentry implements the methodology from the base paper "An Efficient Task Implementation Modeling Framework with Multi-Stage Feature Selection and AutoML: A Case Study in Forest Fire Risk Prediction" (Su et al., 2024). The system integrates satellite-derived meteorological, vegetation, and topographic features to predict fire risk with high accuracy.

## Key Features

- **Dynamic Time Window (DTW) Algorithm**: Identifies critical drying periods preceding fire events
- **Multi-Stage Feature Selection (MSFS)**: Combines Mutual Information Gain, RFECV, and voting aggregation
- **AutoML Training**: Automated algorithm selection and hyperparameter optimization
- **RESTful API**: Real-time fire risk prediction service
- **Comprehensive Evaluation**: ROC curves, confusion matrices, and feature importance analysis

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │    │  Feature Engine  │    │   ML Pipeline   │
├─────────────────┤    ├──────────────────┤    ├─────────────────┤
│ • FIRMS (Fire)  │───▶│ • DTW Algorithm  │───▶│ • MSFS Selection│
│ • CHIRPS (Rain) │    │ • Vegetation Idx │    │ • Auto-sklearn  │
│ • MODIS (LST)   │    │ • Terrain Feat.  │    │ • Ensemble      │
│ • SRTM (DEM)    │    │ • 24 Features    │    │ • FastAPI       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- GDAL (for geospatial processing)
- NASA Earthdata account (for MODIS data)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Mini_Project
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment**:
   ```bash
   cp env.example .env
   # Edit .env with your NASA Earthdata credentials
   ```

4. **Download data** (see Data Acquisition section below)

5. **Build the complete pipeline**:
   ```bash
   python scripts/build_features.py
   ```

6. **Start the API server**:
   ```bash
   python api/main.py
   ```

7. **Test the API**:
   ```bash
   curl -X POST "http://localhost:8000/predict" \
        -H "Content-Type: application/json" \
        -d '{"lat": 30.0, "lon": 79.0, "date": "2024-04-15"}'
   ```

## Data Acquisition

### 1. CHIRPS Precipitation (Automated)
```bash
python scripts/fetch_chirps.py
```
Downloads daily precipitation data (2020-2024) for Uttarakhand region.

### 2. FIRMS Fire Data (Manual)
1. Visit [FIRMS Download](https://firms.modaps.eosdis.nasa.gov/download/)
2. Select MODIS C6.1, draw bounding box over Uttarakhand
3. Date range: 2020-01-01 to 2024-12-31
4. Save as `data/raw/firms/modis_2020_2024.csv`

### 3. MODIS LST (Semi-Automated)
```bash
# Set up NASA Earthdata credentials
echo "machine urs.earthdata.nasa.gov login YOUR_USER password YOUR_PASS" >> ~/.netrc
chmod 600 ~/.netrc

# Download MODIS LST data
python scripts/fetch_modis_lst.py
```

### 4. SRTM DEM
Place your SRTM files in `data/raw/srtm/` directory.

## Feature Engineering

The system extracts 24 features:

### Dynamic Features (21)
- **Precipitation**: min, median, mean, max, sum (5 features)
- **LST**: min, median, mean, max (4 features)
- **NDVI**: min, median, mean, max (4 features)
- **EVI**: min, median, mean, max (4 features)
- **NDWI**: min, median, mean, max (4 features)

### Static Features (3)
- **Terrain**: elevation, slope, aspect

## Model Training

### Multi-Stage Feature Selection (MSFS)
1. **Stage 1**: Mutual Information Gain filtering (top 12 features)
2. **Stage 2**: Recursive Feature Elimination with Cross-Validation
3. **Stage 3**: Voting aggregation across 3 runs

### Auto-sklearn Training
- **Time Budget**: 600 seconds total, 30 seconds per run
- **Cross-Validation**: 5-fold stratified CV
- **Ensemble Size**: Up to 50 models
- **Algorithms**: Random Forest, XGBoost, SVM, MLP, etc.

## API Usage

### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "lat": 30.0,
       "lon": 79.0,
       "date": "2024-04-15"
     }'
```

### Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '[
       {"lat": 30.0, "lon": 79.0, "date": "2024-04-15"},
       {"lat": 30.1, "lon": 79.1, "date": "2024-04-16"}
     ]'
```

### Response Format
```json
{
  "probability": 0.75,
  "risk_level": "High",
  "confidence": 0.85,
  "features_used": 12
}
```

## Evaluation

Run comprehensive model evaluation:
```bash
python scripts/evaluate.py
```

This generates:
- ROC curve (`docs/fig_roc.png`)
- Confusion matrix (`docs/fig_cm.png`)
- Feature importance plot (`docs/fig_feature_importance.png`)
- Precision-recall curve (`docs/fig_pr_curve.png`)
- Results summary (`docs/results.json`)

## Project Structure

```
Mini_Project/
├── data/                    # Data storage
│   ├── raw/                # Raw satellite data
│   ├── derived/            # Processed data
│   └── features.parquet    # Feature matrix
├── features/               # Feature engineering
│   ├── dtw.py             # Dynamic Time Window
│   ├── indices.py         # Vegetation indices
│   ├── terrain.py         # Terrain features
│   └── pipeline.py        # Main pipeline
├── msfs/                  # Feature selection
│   └── selection.py       # MSFS implementation
├── model/                 # Model training
│   ├── train.py          # Auto-sklearn training
│   └── artifacts/        # Saved models
├── api/                   # FastAPI application
│   └── main.py           # API endpoints
├── scripts/               # Utility scripts
│   ├── fetch_chirps.py   # Data acquisition
│   ├── evaluate.py       # Model evaluation
│   └── build_features.py # Complete pipeline
├── tests/                 # Unit tests
└── docs/                  # Documentation
    ├── Mini_Project_Report.md
    └── Mini_Project_review_F1[1].pptx
```

## Configuration

Edit `.env` file for configuration:

```env
# Geographic bounds (Uttarakhand)
BBOX_N=31.5
BBOX_S=28.7
BBOX_E=81.0
BBOX_W=77.5

# Data range
YEAR_START=2020
YEAR_END=2024

# NASA Earthdata credentials
EARTHDATA_USER=your_username
EARTHDATA_PASS=your_password

# Model parameters
MSFS_N_REPEATS=3
AUTOSKLEARN_TIME_LIMIT=600
```

## Performance

Expected model performance:
- **Accuracy**: >93%
- **Precision**: >90%
- **Recall**: >85%
- **F1-Score**: >88%
- **AUC**: >91%

## Deployment

### Local Development
```bash
python api/main.py
```

### Production with Docker
```bash
# Start infrastructure services
docker-compose -f infra/docker-compose.yml up -d

# Run API server
python api/main.py
```

## Testing

Run unit tests:
```bash
python -m pytest tests/
```

Run specific test:
```bash
python -m pytest tests/test_dtw.py -v
```

## Troubleshooting

### Common Issues

1. **GDAL installation issues**:
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install gdal-bin libgdal-dev
   
   # On Windows with Conda
   conda install -c conda-forge gdal
   ```

2. **Auto-sklearn installation**:
   ```bash
   pip install auto-sklearn
   # May require additional system dependencies
   ```

3. **NASA Earthdata authentication**:
   - Ensure `.netrc` file is properly formatted
   - Check credentials are correct
   - Verify account has MODIS data access

### Logs

Check build logs:
```bash
tail -f build.log
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{su2024efficient,
  title={An Efficient Task Implementation Modeling Framework with Multi-Stage Feature Selection and AutoML: A Case Study in Forest Fire Risk Prediction},
  author={Su, Y. and Zhao, L. and Li, H. and Li, X. and Chen, J. and Ge, Y.},
  journal={Remote Sensing},
  volume={16},
  number={17},
  pages={3190},
  year={2024}
}
```

## Contact

For questions or support, please open an issue on GitHub.

## Acknowledgments

- NASA FIRMS for fire detection data
- UCSB CHIRPS for precipitation data
- NASA MODIS for land surface data
- Auto-sklearn team for the AutoML framework




