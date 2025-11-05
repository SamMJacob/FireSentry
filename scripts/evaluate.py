"""
Model Evaluation and Visualization Script

Generates evaluation metrics, ROC curves, confusion matrices, and feature importance plots
for the trained fire prediction model.

Author: FireSentry Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
from pathlib import Path
import logging
import json
import sys
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from model.train import FirePredictionTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Model evaluation and visualization class.
    
    Generates comprehensive evaluation metrics and plots for the trained model.
    """
    
    def __init__(self, model_dir: str = "model/artifacts"):
        """
        Initialize evaluator.
        
        Args:
            model_dir: Directory containing model artifacts
        """
        self.model_dir = Path(model_dir)
        self.trainer = None
        self.results = {}
        
        logger.info(f"ModelEvaluator initialized with model directory: {self.model_dir}")
    
    def load_model(self):
        """Load trained model and artifacts."""
        try:
            self.trainer = FirePredictionTrainer()
            self.trainer.load_model()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def load_test_data(self, feature_file: str = "data/features.parquet", test_size: float = 0.2):
        """
        Load and prepare test data.
        
        Args:
            feature_file: Path to feature Parquet file
            test_size: Test set size fraction
            
        Returns:
            Tuple of (X_test, y_test)
        """
        logger.info("Loading test data")
        
        # Load data
        df = pd.read_parquet(feature_file)
        
        # Separate features and target
        feature_columns = [col for col in df.columns if col not in ['lat', 'lon', 'date', 'target', 'is_fire']]
        X = df[feature_columns].copy()
        y = df['target'].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Split data (same random state as training)
        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logger.info(f"Test data loaded: {len(X_test)} samples")
        return X_test, y_test
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating model performance")
        
        # Make predictions
        y_pred, y_pred_proba = self.trainer.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Store results
        self.results = {
            'metrics': metrics,
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'predictions': {
                'y_true': y_test.tolist(),
                'y_pred': y_pred.tolist(),
                'y_pred_proba': y_pred_proba.tolist()
            }
        }
        
        logger.info("Model evaluation complete")
        logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Test Precision: {metrics['precision']:.4f}")
        logger.info(f"Test Recall: {metrics['recall']:.4f}")
        logger.info(f"Test F1: {metrics['f1']:.4f}")
        logger.info(f"Test AUC: {metrics['auc']:.4f}")
        
        return self.results
    
    def plot_roc_curve(self, output_dir: str = "docs"):
        """Generate and save ROC curve plot."""
        logger.info("Generating ROC curve")
        
        y_true = self.results['predictions']['y_true']
        y_pred_proba = self.results['predictions']['y_pred_proba']
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = self.results['metrics']['auc']
        
        # Create plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Save plot
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        plt.savefig(output_path / "fig_roc.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curve saved to {output_path / 'fig_roc.png'}")
    
    def plot_confusion_matrix(self, output_dir: str = "docs"):
        """Generate and save confusion matrix plot."""
        logger.info("Generating confusion matrix")
        
        cm = np.array(self.results['confusion_matrix'])
        
        # Create plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Fire', 'Fire'],
                   yticklabels=['No Fire', 'Fire'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Save plot
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        plt.savefig(output_path / "fig_cm.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {output_path / 'fig_cm.png'}")
    
    def plot_feature_importance(self, output_dir: str = "docs"):
        """Generate and save feature importance plot."""
        logger.info("Generating feature importance plot")
        
        # Get feature importance
        if hasattr(self.trainer.msfs, 'get_feature_importance'):
            importance = self.trainer.msfs.get_feature_importance()
        else:
            # Fallback: use model feature importance if available
            if hasattr(self.trainer.model, 'feature_importances_'):
                importance = dict(zip(self.trainer.feature_names, self.trainer.model.feature_importances_))
            else:
                logger.warning("No feature importance available")
                return
        
        # Sort features by importance
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        features, importances = zip(*sorted_features)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        bars = plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance (Top Features)')
        plt.gca().invert_yaxis()
        
        # Color bars by importance
        colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        plt.savefig(output_path / "fig_feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature importance plot saved to {output_path / 'fig_feature_importance.png'}")
    
    def plot_precision_recall_curve(self, output_dir: str = "docs"):
        """Generate and save precision-recall curve plot."""
        logger.info("Generating precision-recall curve")
        
        y_true = self.results['predictions']['y_true']
        y_pred_proba = self.results['predictions']['y_pred_proba']
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        
        # Create plot
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        plt.savefig(output_path / "fig_pr_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Precision-recall curve saved to {output_path / 'fig_pr_curve.png'}")
    
    def generate_results_summary(self, output_dir: str = "docs"):
        """Generate results summary JSON file."""
        logger.info("Generating results summary")
        
        # Create comprehensive results summary
        summary = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'model_info': {
                'model_type': self.trainer.model_artifacts.get('model_type', 'unknown'),
                'features_used': len(self.trainer.feature_names),
                'selected_features': self.trainer.feature_names
            },
            'performance_metrics': self.results['metrics'],
            'classification_report': self.results['classification_report'],
            'confusion_matrix': self.results['confusion_matrix'],
            'training_results': self.trainer.training_results
        }
        
        # Save summary
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        with open(output_path / "results.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results summary saved to {output_path / 'results.json'}")
        
        return summary
    
    def run_complete_evaluation(self, feature_file: str = "data/features.parquet"):
        """Run complete evaluation pipeline."""
        logger.info("Starting complete evaluation pipeline")
        
        # Load model
        self.load_model()
        
        # Load test data
        X_test, y_test = self.load_test_data(feature_file)
        
        # Evaluate model
        self.evaluate_model(X_test, y_test)
        
        # Generate plots
        self.plot_roc_curve()
        self.plot_confusion_matrix()
        self.plot_feature_importance()
        self.plot_precision_recall_curve()
        
        # Generate summary
        summary = self.generate_results_summary()
        
        logger.info("Complete evaluation pipeline finished")
        
        return summary

def main():
    """Main evaluation function."""
    evaluator = ModelEvaluator()
    
    try:
        summary = evaluator.run_complete_evaluation()
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS SUMMARY")
        print("="*50)
        print(f"Model Type: {summary['model_info']['model_type']}")
        print(f"Features Used: {summary['model_info']['features_used']}")
        print(f"\nPerformance Metrics:")
        for metric, value in summary['performance_metrics'].items():
            print(f"  {metric.capitalize()}: {value:.4f}")
        
        print(f"\nSelected Features: {summary['model_info']['selected_features']}")
        print(f"\nResults saved to docs/ directory")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()




