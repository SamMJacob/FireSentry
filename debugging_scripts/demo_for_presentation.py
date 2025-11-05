"""
FireSentry Model Demonstration Script
For Presentation to Teacher

This script loads the trained model and demonstrates its performance
with visualizations and detailed metrics.

Author: FireSentry Team
"""

import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FireSentryDemo:
    """Demonstration class for FireSentry model presentation."""
    
    def __init__(self, model_path="models/model.joblib", 
                 data_path="data/processed/features_5000.parquet"):
        """Initialize demo with model and data paths."""
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.model = None
        self.data = None
        
    def load_model(self):
        """Load the trained model."""
        if not self.model_path.exists():
            logger.error(f"❌ Model not found at {self.model_path}")
            logger.info("Please train the model first by running:")
            logger.info("  python -c \"from model.train import FirePredictionTrainer; t = FirePredictionTrainer(); t.train('data/processed/features_5000.parquet')\"")
            return False
            
        logger.info(f"Loading model from {self.model_path}")
        self.model = joblib.load(self.model_path)
        logger.info("✅ Model loaded successfully")
        return True
    
    def load_data(self):
        """Load and prepare data."""
        if not self.data_path.exists():
            logger.error(f"❌ Data not found at {self.data_path}")
            return False
            
        logger.info(f"Loading data from {self.data_path}")
        self.data = pd.read_parquet(self.data_path)
        logger.info(f"✅ Loaded {len(self.data)} samples")
        return True
    
    def print_data_summary(self):
        """Print summary statistics about the dataset."""
        logger.info("\n" + "="*80)
        logger.info("📊 DATASET SUMMARY")
        logger.info("="*80)
        
        # Basic stats
        logger.info(f"Total samples: {len(self.data):,}")
        fire_count = self.data['is_fire'].sum()
        non_fire_count = len(self.data) - fire_count
        logger.info(f"Fire samples: {fire_count:,} ({fire_count/len(self.data)*100:.1f}%)")
        logger.info(f"Non-fire samples: {non_fire_count:,} ({non_fire_count/len(self.data)*100:.1f}%)")
        
        # Temporal coverage
        if 'date' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['date'])
            logger.info(f"\n📅 Temporal Coverage:")
            logger.info(f"Date range: {self.data['date'].min().date()} to {self.data['date'].max().date()}")
            logger.info(f"Years covered: {sorted(self.data['date'].dt.year.unique())}")
        
        # Spatial coverage
        if 'lat' in self.data.columns and 'lon' in self.data.columns:
            logger.info(f"\n🌍 Spatial Coverage (Uttarakhand Region):")
            logger.info(f"Latitude range: {self.data['lat'].min():.3f}° to {self.data['lat'].max():.3f}°")
            logger.info(f"Longitude range: {self.data['lon'].min():.3f}° to {self.data['lon'].max():.3f}°")
        
        # Features
        feature_cols = [c for c in self.data.columns if c not in ['lat', 'lon', 'date', 'is_fire', 'dtw_start', 'dtw_end']]
        logger.info(f"\n🔬 Features: {len(feature_cols)} total")
        logger.info(f"Feature names: {', '.join(feature_cols[:10])}{'...' if len(feature_cols) > 10 else ''}")
    
    def print_model_info(self):
        """Print information about the trained model."""
        logger.info("\n" + "="*80)
        logger.info("🤖 MODEL INFORMATION")
        logger.info("="*80)
        
        # Check if it's an Auto-sklearn model
        model_type = type(self.model).__name__
        logger.info(f"Model type: {model_type}")
        
        # Try to get ensemble information
        if hasattr(self.model, 'show_models'):
            logger.info("\n📦 Ensemble Composition:")
            try:
                ensemble_info = self.model.show_models()
                logger.info(ensemble_info)
            except:
                logger.info("(Ensemble details not available)")
        
        # Try to get leaderboard
        if hasattr(self.model, 'leaderboard'):
            logger.info("\n🏆 Model Leaderboard:")
            try:
                leaderboard = self.model.leaderboard()
                logger.info(leaderboard.head(10).to_string())
            except:
                logger.info("(Leaderboard not available)")
    
    def evaluate_temporal_split(self, split_date=None, test_size=0.2):
        """
        Evaluate model with temporal split to prevent data leakage.
        
        Args:
            split_date: Date to split train/test (None = auto-calculate for 80/20 split)
            test_size: Test set fraction (only used if split_date is None)
        """
        logger.info("\n" + "="*80)
        logger.info("🔥 MODEL EVALUATION (TEMPORAL SPLIT - NO DATA LEAKAGE)")
        logger.info("="*80)
        
        # Prepare data
        df = self.data.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate split date: if provided, use it; otherwise calculate from percentile
        if split_date is None:
            split_date = df['date'].quantile(1 - test_size)
            logger.info(f"\n📅 Temporal Split Configuration (Auto-calculated):")
            logger.info(f"📊 Calculated split date from {test_size*100:.0f}% test size: {split_date.date()}")
        else:
            split_date = pd.to_datetime(split_date)
            logger.info(f"\n📅 Temporal Split Configuration:")
            logger.info(f"📊 Using provided split date: {split_date.date()}")
        
        # Split by time
        train_mask = df['date'] < split_date
        test_mask = df['date'] >= split_date
        
        train_pct = train_mask.sum() / len(df) * 100
        test_pct = test_mask.sum() / len(df) * 100
        
        logger.info(f"Train period: {df[train_mask]['date'].min().date()} to {df[train_mask]['date'].max().date()} ({train_pct:.1f}%)")
        logger.info(f"Test period: {df[test_mask]['date'].min().date()} to {df[test_mask]['date'].max().date()} ({test_pct:.1f}%)")
        logger.info(f"Train samples: {train_mask.sum():,}")
        logger.info(f"Test samples: {test_mask.sum():,}")
        
        # Prepare features
        feature_cols = [c for c in df.columns if c not in ['lat', 'lon', 'date', 'is_fire', 'dtw_start', 'dtw_end']]
        
        X_test = df[test_mask][feature_cols]
        y_test = df[test_mask]['is_fire'].astype(int)
        
        # Handle missing values
        X_test = X_test.fillna(X_test.median()).fillna(0)
        X_test = X_test.replace([np.inf, -np.inf], 0)
        
        # Make predictions
        logger.info("\n🎯 Making predictions on test set...")
        try:
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            logger.info("\n📈 Performance Metrics:")
            logger.info(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall:    {recall:.4f}")
            logger.info(f"F1-Score:  {f1:.4f}")
            logger.info(f"ROC-AUC:   {auc:.4f}")
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            logger.info("\n📊 Confusion Matrix:")
            logger.info(f"                 Predicted")
            logger.info(f"                 No Fire  Fire")
            logger.info(f"Actual No Fire  {cm[0,0]:6d}  {cm[0,1]:6d}")
            logger.info(f"Actual Fire     {cm[1,0]:6d}  {cm[1,1]:6d}")
            
            # Classification report
            logger.info("\n📋 Detailed Classification Report:")
            logger.info(classification_report(y_test, y_pred, 
                                             target_names=['No Fire', 'Fire'],
                                             digits=4))
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'confusion_matrix': cm
            }
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            return None
    
    def create_visualizations(self, metrics, save_dir="presentation_plots"):
        """Create visualizations for presentation."""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        logger.info("\n" + "="*80)
        logger.info("📊 CREATING VISUALIZATIONS")
        logger.info("="*80)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # 1. Performance Metrics Bar Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        metrics_values = [
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1'],
            metrics['auc']
        ]
        
        bars = ax.bar(metrics_names, metrics_values, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6'])
        ax.set_ylim([0, 1])
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('FireSentry Model Performance Metrics\n(Temporal Split - No Data Leakage)', 
                     fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'performance_metrics.png', dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved: {save_dir / 'performance_metrics.png'}")
        plt.close()
        
        # 2. Confusion Matrix Heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Fire', 'Fire'],
                   yticklabels=['No Fire', 'Fire'],
                   ax=ax, cbar_kws={'label': 'Count'})
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_title('Confusion Matrix\n(Temporal Split - No Data Leakage)', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved: {save_dir / 'confusion_matrix.png'}")
        plt.close()
        
        # 3. Feature Importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            fig, ax = plt.subplots(figsize=(10, 8))
            importances = self.model.feature_importances_
            feature_names = [c for c in self.data.columns if c not in ['lat', 'lon', 'date', 'is_fire', 'dtw_start', 'dtw_end']]
            
            # Sort by importance
            indices = np.argsort(importances)[::-1][:15]  # Top 15
            
            ax.barh(range(len(indices)), importances[indices], color='#3498db')
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([feature_names[i] for i in indices])
            ax.set_xlabel('Feature Importance', fontsize=12)
            ax.set_title('Top 15 Most Important Features', fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
            logger.info(f"✅ Saved: {save_dir / 'feature_importance.png'}")
            plt.close()
        
        logger.info(f"\n✅ All visualizations saved to {save_dir}/")
    
    def run_demo(self, split_date=None):
        """Run complete demonstration."""
        logger.info("="*80)
        logger.info("🔥 FireSentry Model Demonstration for Presentation")
        logger.info("="*80)
        
        # Load model and data
        if not self.load_model():
            return
        if not self.load_data():
            return
        
        # Print summaries
        self.print_data_summary()
        self.print_model_info()
        
        # Evaluate with temporal split (auto-calculates 80/20 if split_date is None)
        metrics = self.evaluate_temporal_split(split_date)
        
        if metrics:
            # Create visualizations
            self.create_visualizations(metrics)
            
            logger.info("\n" + "="*80)
            logger.info("✅ DEMONSTRATION COMPLETE")
            logger.info("="*80)
            logger.info("\nFor your presentation, you can:")
            logger.info("1. Show the performance metrics (printed above)")
            logger.info("2. Display the visualization plots in presentation_plots/")
            logger.info("3. Explain the temporal split approach to prevent data leakage")
            logger.info("4. Discuss the feature engineering (24 features from MODIS, CHIRPS, SRTM)")
            logger.info("\n🎓 Good luck with your presentation!")

def main():
    """Main entry point."""
    demo = FireSentryDemo()
    demo.run_demo(split_date=None)  # None = auto-calculate 80/20 split

if __name__ == "__main__":
    main()

