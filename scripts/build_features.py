#!/usr/bin/env python3
"""
Build Features Script (OPTIMIZED)

Main script to run the complete feature engineering pipeline with optimizations.
Downloads data, builds features, trains model, and generates evaluation.

OPTIMIZATIONS:
- Uses lookup table for vegetation indices (3x faster)
- Processes all years 2020-2024 (not just 2020)
- Batch processing for better memory management

Usage:
    python scripts/build_features.py [--skip-download] [--skip-training]

Author: FireSentry Team
"""

import argparse
import logging
import sys
from pathlib import Path
import time
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from features.pipeline_optimized import OptimizedFeaturePipeline
from model.train import FirePredictionTrainer

# Optional imports for evaluation and visualization
try:
    from scripts.evaluate import ModelEvaluator
    EVALUATOR_AVAILABLE = True
except ImportError as e:
    EVALUATOR_AVAILABLE = False
    logging.warning(f"ModelEvaluator not available: {e}")

try:
    from scripts.make_deck import PPTXGenerator
    PPTX_AVAILABLE = True
except ImportError as e:
    PPTX_AVAILABLE = False
    logging.warning(f"PPTXGenerator not available: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('build.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FireSentryBuilder:
    """
    Main builder class for FireSentry MVP.
    
    Orchestrates the complete pipeline from data acquisition to model deployment.
    """
    
    def __init__(self):
        """Initialize builder."""
        self.start_time = time.time()
        self.results = {}
        
        logger.info("FireSentry Builder initialized")
    
    def check_prerequisites(self):
        """Check if required files and directories exist."""
        logger.info("Checking prerequisites...")
        
        # Check if FIRMS data exists
        firms_file = Path("data/raw/firms/firms_uttarakhand_2020_2025.csv")
        if not firms_file.exists():
            logger.warning(f"FIRMS data not found: {firms_file}")
            logger.warning("Please download FIRMS data manually or run fetch script")
            return False
        
        # Check if CHIRPS data exists
        chirps_dir = Path("data/raw/chirps")
        if not chirps_dir.exists() or not any(chirps_dir.iterdir()):
            logger.warning("CHIRPS data not found. Run fetch_chirps.py first")
            return False
        
        # Check if SRTM data exists
        srtm_dir = Path("data/raw/srtm")
        if not srtm_dir.exists() or not any(srtm_dir.iterdir()):
            logger.warning("SRTM data not found. Please add SRTM files to data/raw/srtm/")
            return False
        
        # Check if processed terrain products exist
        terrain_dir = Path("data/derived/terrain")
        required_terrain_files = ["elevation.tif", "slope.tif", "aspect.tif"]
        missing_terrain = []
        
        for terrain_file in required_terrain_files:
            if not (terrain_dir / terrain_file).exists():
                missing_terrain.append(terrain_file)
        
        if missing_terrain:
            logger.warning(f"Missing terrain products: {missing_terrain}")
            logger.warning("Run 'python scripts/process_srtm.py' to generate terrain products")
            return False
        
        logger.info("Prerequisites check passed")
        return True
    
    def build_features(self):
        """Build feature matrix from raw data."""
        logger.info("Building feature matrix...")
        
        try:
            # Initialize OPTIMIZED pipeline
            pipeline = OptimizedFeaturePipeline()
            
            # Load fire data
            fire_points = pipeline.load_fire_data()
            if len(fire_points) == 0:
                logger.error("No fire data loaded")
                return False
            
            # Generate pseudo fire points
            pseudo_points = pipeline.generate_pseudo_fire_points(fire_points, ratio=1.0)
            
            # Build feature matrix
            feature_matrix = pipeline.build_feature_matrix(fire_points, pseudo_points)
            
            # Save feature matrix
            output_file, metadata_file = pipeline.save_feature_matrix(feature_matrix)
            
            self.results['feature_matrix'] = {
                'file': str(output_file),
                'samples': len(feature_matrix),
                'features': len(pipeline.feature_columns),
                'target_distribution': feature_matrix['target'].value_counts().to_dict()
            }
            
            logger.info(f"Feature matrix built: {len(feature_matrix)} samples, {len(pipeline.feature_columns)} features")
            return True
            
        except Exception as e:
            logger.error(f"Feature building failed: {e}")
            return False
    
    def train_model(self):
        """Train the fire prediction model."""
        logger.info("Training model...")
        
        try:
            # Initialize trainer
            trainer = FirePredictionTrainer(
                time_limit=600,
                per_run_time_limit=30,
                use_autosklearn=True  # Will fallback to sklearn if not available
            )
            
            # Train model
            training_results = trainer.train()
            
            # Save model
            trainer.save_model()
            
            self.results['training'] = training_results
            
            logger.info("Model training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False
    
    def evaluate_model(self):
        """Evaluate the trained model."""
        logger.info("Evaluating model...")
        
        try:
            # Initialize evaluator
            evaluator = ModelEvaluator()
            
            # Run complete evaluation
            summary = evaluator.run_complete_evaluation()
            
            self.results['evaluation'] = summary
            
            logger.info("Model evaluation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return False
    
    def generate_presentation(self):
        """Generate PowerPoint presentation."""
        logger.info("Generating presentation...")
        
        try:
            # Initialize generator
            generator = PPTXGenerator()
            
            # Generate presentation
            success = generator.generate_presentation()
            
            if success:
                logger.info("Presentation generated successfully")
                return True
            else:
                logger.warning("Presentation generation failed")
                return False
                
        except Exception as e:
            logger.error(f"Presentation generation failed: {e}")
            return False
    
    def print_summary(self):
        """Print build summary."""
        elapsed_time = time.time() - self.start_time
        
        print("\n" + "="*60)
        print("FIRESENTRY BUILD SUMMARY")
        print("="*60)
        print(f"Build completed in: {elapsed_time:.1f} seconds")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if 'feature_matrix' in self.results:
            fm = self.results['feature_matrix']
            print(f"\nFeature Matrix:")
            print(f"  • File: {fm['file']}")
            print(f"  • Samples: {fm['samples']}")
            print(f"  • Features: {fm['features']}")
            print(f"  • Target Distribution: {fm['target_distribution']}")
        
        if 'training' in self.results:
            tr = self.results['training']
            print(f"\nTraining Results:")
            print(f"  • Test Accuracy: {tr.get('test_accuracy', 'N/A'):.4f}")
            print(f"  • Test Precision: {tr.get('test_precision', 'N/A'):.4f}")
            print(f"  • Test Recall: {tr.get('test_recall', 'N/A'):.4f}")
            print(f"  • Test F1: {tr.get('test_f1', 'N/A'):.4f}")
            print(f"  • Test AUC: {tr.get('test_auc', 'N/A'):.4f}")
        
        if 'evaluation' in self.results:
            ev = self.results['evaluation']
            print(f"\nEvaluation Results:")
            print(f"  • Model Type: {ev.get('model_info', {}).get('model_type', 'N/A')}")
            print(f"  • Features Used: {ev.get('model_info', {}).get('features_used', 'N/A')}")
        
        print(f"\nGenerated Files:")
        print(f"  • Feature Matrix: data/features.parquet")
        print(f"  • Model Artifacts: model/artifacts/")
        print(f"  • Evaluation Results: docs/results.json")
        print(f"  • Figures: docs/fig_*.png")
        print(f"  • Presentation: docs/Mini_Project_review_F1[1].pptx")
        
        print(f"\nNext Steps:")
        print(f"  • Start API server: python api/main.py")
        print(f"  • Test predictions: curl -X POST http://localhost:8000/predict")
        print(f"  • View API docs: http://localhost:8000/docs")
        
        print("="*60)
    
    def run(self, skip_download=False, skip_training=False):
        """
        Run the complete build pipeline.
        
        Args:
            skip_download: Skip data download checks
            skip_training: Skip model training
        """
        logger.info("Starting FireSentry build pipeline")
        
        # Check prerequisites
        if not skip_download and not self.check_prerequisites():
            logger.error("Prerequisites check failed")
            return False
        
        # Build features
        if not self.build_features():
            logger.error("Feature building failed")
            return False
        
        # Train model
        if not skip_training:
            if not self.train_model():
                logger.error("Model training failed")
                return False
            
            # Evaluate model
            if not self.evaluate_model():
                logger.error("Model evaluation failed")
                return False
            
            # Generate presentation
            self.generate_presentation()
        
        # Print summary
        self.print_summary()
        
        logger.info("FireSentry OPTIMIZED build pipeline completed successfully")
        return True

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Build FireSentry MVP")
    parser.add_argument("--skip-download", action="store_true", 
                       help="Skip data download checks")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip model training and evaluation")
    
    args = parser.parse_args()
    
    builder = FireSentryBuilder()
    success = builder.run(
        skip_download=args.skip_download,
        skip_training=args.skip_training
    )
    
    if success:
        print("\n✅ FireSentry build completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ FireSentry build failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

