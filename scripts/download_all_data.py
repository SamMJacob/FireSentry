#!/usr/bin/env python3
"""
Complete Data Download Pipeline

Downloads all required data for FireSentry project:
1. CHIRPS precipitation data
2. FIRMS fire data  
3. MODIS LST data
4. MODIS Surface Reflectance data

Usage:
    python scripts/download_all_data.py

Requirements:
    - All dependencies from requirements.txt
    - NASA Earthdata credentials in .env
    - FIRMS API token in .env
"""

import os
import sys
import subprocess
from pathlib import Path
import logging
from datetime import datetime
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataDownloader:
    def __init__(self):
        """Initialize data downloader."""
        load_dotenv()
        self.check_credentials()
        self.setup_directories()
        
    def check_credentials(self):
        """Check if required credentials are available."""
        required_vars = ['EARTHDATA_USER', 'EARTHDATA_PASS', 'FIRMS_MAP_KEY']
        missing_vars = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            logger.error(f"Missing environment variables: {missing_vars}")
            logger.error("Please update your .env file with the required credentials")
            sys.exit(1)
        
        logger.info("✅ All required credentials found")
    
    def setup_directories(self):
        """Create data directories."""
        directories = [
            "data/raw/chirps",
            "data/raw/firms", 
            "data/raw/modis_lst",
            "data/raw/modis_sr",
            "data/raw/srtm",
            "data/derived/terrain"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ Data directories created")
    
    def setup_netrc(self):
        """Setup .netrc file for NASA Earthdata authentication."""
        netrc_path = Path.home() / ".netrc"
        
        username = os.getenv('EARTHDATA_USER')
        password = os.getenv('EARTHDATA_PASS')
        
        netrc_content = f"""machine urs.earthdata.nasa.gov
login {username}
password {password}
"""
        
        with open(netrc_path, 'w') as f:
            f.write(netrc_content)
        
        # Set proper permissions
        os.chmod(netrc_path, 0o600)
        
        logger.info("✅ .netrc file created for NASA Earthdata authentication")
    
    def download_chirps(self):
        """Download CHIRPS precipitation data."""
        logger.info("🌧️  Starting CHIRPS precipitation download...")
        
        try:
            result = subprocess.run([
                sys.executable, "scripts/fetch_chirps.py"
            ], capture_output=True, text=True, cwd=Path.cwd())
            
            if result.returncode == 0:
                logger.info("✅ CHIRPS download completed successfully")
                return True
            else:
                logger.error(f"❌ CHIRPS download failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ CHIRPS download error: {e}")
            return False
    
    def download_firms(self):
        """Download FIRMS fire data."""
        logger.info("🔥 Starting FIRMS fire data download...")
        
        try:
            result = subprocess.run([
                sys.executable, "scripts/fetch_firms.py"
            ], capture_output=True, text=True, cwd=Path.cwd())
            
            if result.returncode == 0:
                logger.info("✅ FIRMS download completed successfully")
                return True
            else:
                logger.error(f"❌ FIRMS download failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ FIRMS download error: {e}")
            return False
    
    def download_modis_lst(self):
        """Download MODIS LST data."""
        logger.info("🌡️  Starting MODIS LST download...")
        
        try:
            result = subprocess.run([
                sys.executable, "scripts/fetch_modis_lst.py"
            ], capture_output=True, text=True, cwd=Path.cwd())
            
            if result.returncode == 0:
                logger.info("✅ MODIS LST download completed successfully")
                return True
            else:
                logger.error(f"❌ MODIS LST download failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ MODIS LST download error: {e}")
            return False
    
    def download_modis_sr(self):
        """Download MODIS Surface Reflectance data."""
        logger.info("🌿 Starting MODIS Surface Reflectance download...")
        
        try:
            result = subprocess.run([
                sys.executable, "scripts/fetch_modis_sr.py"
            ], capture_output=True, text=True, cwd=Path.cwd())
            
            if result.returncode == 0:
                logger.info("✅ MODIS SR download completed successfully")
                return True
            else:
                logger.error(f"❌ MODIS SR download failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ MODIS SR download error: {e}")
            return False
    
    def process_srtm(self):
        """Process SRTM DEM data to create terrain products."""
        logger.info("🏔️  Starting SRTM terrain processing...")
        
        # Check if SRTM files exist
        srtm_dir = Path("data/raw/srtm")
        if not srtm_dir.exists() or not any(srtm_dir.iterdir()):
            logger.warning("⚠️  No SRTM files found in data/raw/srtm/")
            logger.warning("Please copy your SRTM .hgt.zip files to data/raw/srtm/ first")
            return False
        
        try:
            result = subprocess.run([
                sys.executable, "scripts/process_srtm.py"
            ], capture_output=True, text=True, cwd=Path.cwd())
            
            if result.returncode == 0:
                logger.info("✅ SRTM processing completed successfully")
                return True
            else:
                logger.error(f"❌ SRTM processing failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error running SRTM processing: {e}")
            return False
    
    def validate_downloads(self):
        """Validate downloaded data."""
        logger.info("🔍 Validating downloaded data...")
        
        validation_results = {}
        
        # Check CHIRPS data
        chirps_dir = Path("data/raw/chirps")
        if chirps_dir.exists() and any(chirps_dir.iterdir()):
            validation_results['chirps'] = True
            logger.info("✅ CHIRPS data found")
        else:
            validation_results['chirps'] = False
            logger.warning("⚠️  CHIRPS data not found")
        
        # Check FIRMS data
        firms_files = list(Path("data/raw/firms").glob("*.csv"))
        if firms_files:
            validation_results['firms'] = True
            logger.info(f"✅ FIRMS data found: {len(firms_files)} files")
        else:
            validation_results['firms'] = False
            logger.warning("⚠️  FIRMS data not found")
        
        # Check MODIS LST data
        lst_dir = Path("data/raw/modis_lst")
        if lst_dir.exists() and any(lst_dir.iterdir()):
            validation_results['modis_lst'] = True
            logger.info("✅ MODIS LST data found")
        else:
            validation_results['modis_lst'] = False
            logger.warning("⚠️  MODIS LST data not found")
        
        # Check MODIS SR data
        sr_dir = Path("data/raw/modis_sr")
        if sr_dir.exists() and any(sr_dir.iterdir()):
            validation_results['modis_sr'] = True
            logger.info("✅ MODIS SR data found")
        else:
            validation_results['modis_sr'] = False
            logger.warning("⚠️  MODIS SR data not found")
        
        # Check SRTM terrain products
        terrain_dir = Path("data/derived/terrain")
        required_terrain_files = ["elevation.tif", "slope.tif", "aspect.tif"]
        terrain_complete = all((terrain_dir / f).exists() for f in required_terrain_files)
        
        if terrain_complete:
            validation_results['srtm_terrain'] = True
            logger.info("✅ SRTM terrain products found")
        else:
            validation_results['srtm_terrain'] = False
            logger.warning("⚠️  SRTM terrain products not found")
        
        return validation_results
    
    def run_all_downloads(self):
        """Run all data downloads."""
        logger.info("🚀 Starting complete data download pipeline...")
        logger.info(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Setup authentication
        self.setup_netrc()
        
        # Download data
        results = {}
        
        results['chirps'] = self.download_chirps()
        results['firms'] = self.download_firms()
        results['modis_lst'] = self.download_modis_lst()
        results['modis_sr'] = self.download_modis_sr()
        results['srtm_terrain'] = self.process_srtm()
        
        # Validate downloads
        validation = self.validate_downloads()
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info("📊 DOWNLOAD SUMMARY")
        logger.info("="*50)
        
        for dataset, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            logger.info(f"{dataset.upper()}: {status}")
        
        logger.info("\n🔍 VALIDATION SUMMARY")
        for dataset, found in validation.items():
            status = "✅ FOUND" if found else "⚠️  NOT FOUND"
            logger.info(f"{dataset.upper()}: {status}")
        
        successful_downloads = sum(results.values())
        total_downloads = len(results)
        
        logger.info(f"\n📈 SUCCESS RATE: {successful_downloads}/{total_downloads} ({successful_downloads/total_downloads*100:.1f}%)")
        
        if successful_downloads == total_downloads:
            logger.info("🎉 All downloads completed successfully!")
            logger.info("➡️  Next step: Run feature engineering pipeline")
        else:
            logger.warning("⚠️  Some downloads failed. Check logs above for details.")
        
        logger.info(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main function."""
    try:
        downloader = DataDownloader()
        downloader.run_all_downloads()
        
    except KeyboardInterrupt:
        logger.info("\n⏹️  Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
