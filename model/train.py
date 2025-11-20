"""
Auto-sklearn Model Training

Trains ensemble models using Auto-sklearn for fire prediction.
Implements the training pipeline as specified in the base paper.

Author: FireSentry Team
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import logging
from typing import Tuple, Dict, Optional
import json
from datetime import datetime
import sys
import os
from dotenv import load_dotenv
import psutil

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from msfs.selection import MultiStageFeatureSelection

# Auto-sklearn imports
try:
    import autosklearn.classification
    from autosklearn.metrics import accuracy, precision, recall, f1_macro, roc_auc
    AUTOSKLEARN_AVAILABLE = True
except ImportError:
    AUTOSKLEARN_AVAILABLE = False
    logging.warning("Auto-sklearn not available. Using scikit-learn fallback.")

# Fallback imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class FirePredictionTrainer:
    """
    Fire prediction model trainer using Auto-sklearn.
    
    Implements the training pipeline with MSFS feature selection and Auto-sklearn
    ensemble training as specified in the base paper.
    """
    
    def __init__(self, 
                 time_limit: Optional[int] = None,
                 per_run_time_limit: Optional[int] = None,
                 ensemble_size: Optional[int] = None,
                 ensemble_nbest: Optional[int] = None,
                 random_state: int = 42,
                 use_autosklearn: bool = True):
        """
        Initialize trainer.
        
        Args:
            time_limit: Total time limit for Auto-sklearn (seconds)
            per_run_time_limit: Time limit per model run (seconds)
            ensemble_size: Maximum ensemble size
            ensemble_nbest: Number of best models to include in ensemble
            random_state: Random seed
            use_autosklearn: Whether to use Auto-sklearn (fallback to sklearn if False)
        """
        # Load environment variables
        load_dotenv()
        
        # Set parameters from environment or defaults
        self.time_limit = time_limit or int(os.getenv('AUTOSKLEARN_TIME_LIMIT', 14400))
        self.per_run_time_limit = per_run_time_limit or int(os.getenv('AUTOSKLEARN_PER_RUN_LIMIT', 300))
        self.ensemble_size = ensemble_size or int(os.getenv('AUTOSKLEARN_ENSEMBLE_SIZE', 50))
        self.ensemble_nbest = ensemble_nbest or int(os.getenv('AUTOSKLEARN_ENSEMBLE_NBEST', 10))
        self.random_state = random_state
        self.use_autosklearn = use_autosklearn and AUTOSKLEARN_AVAILABLE
        
        # Model components
        self.msfs = None
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        # Results storage
        self.training_results = {}
        self.model_artifacts = {}
        
        logger.info(f"Trainer initialized: Auto-sklearn={self.use_autosklearn}, "
                   f"time_limit={time_limit}s, ensemble_size={ensemble_size}")
    
    def load_feature_data(self, feature_file: str = "data/features.parquet") -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load feature matrix from Parquet file.
        
        Args:
            feature_file: Path to feature Parquet file
            
        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        logger.info(f"Loading feature data from {feature_file}")
        
        if not Path(feature_file).exists():
            raise FileNotFoundError(f"Feature file not found: {feature_file}")
        
        # Load data
        df = pd.read_parquet(feature_file)
        
        # Separate features and target
        feature_columns = [col for col in df.columns if col not in ['lat', 'lon', 'date', 'target', 'is_fire']]
        X = df[feature_columns].copy()
        
        # Handle different target column names
        if 'target' in df.columns:
            y = df['target'].copy()
        elif 'is_fire' in df.columns:
            y = df['is_fire'].copy()
        else:
            raise ValueError("No target column found. Expected 'target' or 'is_fire'")
        
        # Handle missing values - more robust approach
        logger.info(f"Handling missing values...")
        logger.info(f"NaN counts before cleaning: {X.isnull().sum().sum()}")
        
        # Fill NaN values with median for numeric columns
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].median())
        
        # For any remaining NaN values, fill with 0
        X = X.fillna(0)
        
        # Check for any remaining NaN values
        if X.isnull().any().any():
            logger.warning(f"Still have NaN values after cleaning: {X.isnull().sum().sum()}")
            # Replace any remaining NaN with 0
            X = X.fillna(0)
        
        # Ensure target is boolean/int
        if y.dtype == 'object':
            y = y.astype(bool)
        y = y.astype(int)
        
        # Check for infinite values
        if np.isinf(X.select_dtypes(include=[np.number])).any().any():
            logger.warning("Infinite values detected, replacing with 0")
            X = X.replace([np.inf, -np.inf], 0)
        
        logger.info(f"Loaded {len(X)} samples with {len(feature_columns)} features")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        logger.info(f"NaN counts after cleaning: {X.isnull().sum().sum()}")
        
        return X, y
    
    def apply_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Apply MSFS feature selection.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Selected features
        """
        logger.info("Applying MSFS feature selection")
        
        # Final NaN check before feature selection
        if X.isnull().any().any():
            logger.warning(f"NaN values detected before feature selection: {X.isnull().sum().sum()}")
            X = X.fillna(0)
        
        # Check for infinite values
        if np.isinf(X.select_dtypes(include=[np.number])).any().any():
            logger.warning("Infinite values detected, replacing with 0")
            X = X.replace([np.inf, -np.inf], 0)
        
        # Initialize MSFS with production parameters
        n_repeats = int(os.getenv('MSFS_N_REPEATS', 10))
        self.msfs = MultiStageFeatureSelection(
            k_mi=12,
            n_repeats=n_repeats,
            cv_folds=5,
            random_state=self.random_state
        )
        
        # Fit and transform with error handling
        try:
            X_selected = self.msfs.fit_transform(X, y)
        except Exception as e:
            logger.error(f"MSFS feature selection failed: {e}")
            logger.info("Falling back to all features")
            X_selected = X.copy()
        
        # Store feature names
        self.feature_names = list(X_selected.columns)
        
        logger.info(f"Feature selection complete: {X.shape[1]} → {X_selected.shape[1]} features")
        logger.info(f"Selected features: {self.feature_names}")
        
        return X_selected
    
    def train_autosklearn(self, X: pd.DataFrame, y: pd.Series):
        """
        Train Auto-sklearn ensemble model with robust error handling.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Trained Auto-sklearn model (AutoSklearnClassifier if available, else RandomForestClassifier)
        """
        logger.info("Training Auto-sklearn ensemble model")
        logger.info(f"System specs: {psutil.cpu_count()} cores, {psutil.virtual_memory().total / (1024**3):.1f}GB RAM")
        logger.info(f"Available memory: {psutil.virtual_memory().available / (1024**3):.1f}GB")
        
        try:
            # Initialize Auto-sklearn with more conservative parameters for debugging
            automl = autosklearn.classification.AutoSklearnClassifier(
                time_left_for_this_task=self.time_limit,
                per_run_time_limit=60,  # Shorter per-run limit for debugging
                memory_limit=4096,  # Reduced memory limit
                ensemble_size=5,  # Smaller ensemble for debugging
                ensemble_nbest=3,  # Fewer models in ensemble
                initial_configurations_via_metalearning=5,  # Fewer initial configs
                resampling_strategy='holdout',  # Use holdout instead of CV for debugging
                resampling_strategy_arguments={'train_size': 0.8},
                metric=roc_auc,  # Use AUC for better performance on imbalanced data
                seed=self.random_state,
                n_jobs=2,  # Reduced parallel jobs for debugging
                delete_tmp_folder_after_terminate=True,
                tmp_folder=None,  # Use default temp folder
                # Note: include_estimators and include_preprocessors are deprecated in newer Auto-sklearn versions
                # Auto-sklearn will automatically select appropriate algorithms
            )
            
            # Train model with timeout protection
            logger.info("Starting Auto-sklearn training...")
            logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
            logger.info(f"Data quality check: NaN in X={X.isnull().any().any()}, NaN in y={y.isnull().any()}")
            
            # Quick test with basic sklearn model to verify data is trainable
            logger.info("Testing data with basic RandomForest...")
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            
            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                test_rf = RandomForestClassifier(n_estimators=10, random_state=42)
                test_rf.fit(X_train, y_train)
                test_score = test_rf.score(X_test, y_test)
                logger.info(f"✅ Basic RandomForest test successful: {test_score:.3f} accuracy")
            except Exception as e:
                logger.error(f"❌ Basic RandomForest test failed: {e}")
                logger.error("Data is not trainable - this explains Auto-sklearn failures")
                raise e
            
            # Fit with detailed error handling
            try:
                logger.info("Calling automl.fit()...")
                automl.fit(X, y)
                logger.info("automl.fit() completed successfully")
            except Exception as e:
                logger.error(f"Auto-sklearn fit failed with error: {e}")
                logger.error(f"Error type: {type(e).__name__}")
                
                # Check for specific error types
                if "Input contains NaN" in str(e) or "Input contains infinity" in str(e):
                    logger.error(f"Data quality error during fit: {e}")
                    logger.info("Attempting to clean data again...")
                    X = X.fillna(0).replace([np.inf, -np.inf], 0)
                    y = y.fillna(0)
                    logger.info("Retrying fit with cleaned data...")
                    automl.fit(X, y)
                elif "No runs were available" in str(e):
                    logger.error("No successful model runs - this suggests individual models are failing")
                    logger.info("This could be due to:")
                    logger.info("1. Data quality issues (even after cleaning)")
                    logger.info("2. Memory constraints")
                    logger.info("3. Algorithm compatibility issues")
                    logger.info("4. Resource contention")
                    raise e
                else:
                    logger.error(f"Unexpected error: {e}")
                    raise e
            
            # Try to get ensemble info with fallback
            try:
                models_with_weights = automl.get_models_with_weights()
                logger.info(f"Auto-sklearn training complete")
                logger.info(f"Final ensemble size: {len(models_with_weights)}")
                
                # Check if ensemble is empty (common issue)
                if len(models_with_weights) == 0:
                    logger.warning("Empty ensemble detected - this may cause prediction issues")
                    logger.info("Auto-sklearn will use the best single model for predictions")
                else:
                    # Log detailed ensemble composition
                    logger.info("🔍 ENSEMBLE COMPOSITION:")
                    logger.info("-" * 40)
                    for i, (model_id, weight) in enumerate(models_with_weights, 1):
                        logger.info(f"{i:2d}. Model ID: {model_id} (weight: {weight:.4f})")
                        try:
                            individual_model = automl.get_models()[model_id]
                            logger.info(f"    Type: {type(individual_model).__name__}")
                        except Exception as e:
                            logger.warning(f"    Could not get model details: {e}")
                    
                    # Try to get leaderboard
                    try:
                        leaderboard = automl.leaderboard()
                        if len(leaderboard) > 0:
                            logger.info("\n📈 TOP 5 MODELS BY PERFORMANCE:")
                            logger.info(leaderboard.head().to_string())
                    except Exception as e:
                        logger.warning(f"Could not get leaderboard: {e}")
                    
            except Exception as e:
                logger.warning(f"Could not get ensemble info: {e}")
                logger.info("Auto-sklearn training completed (ensemble info unavailable)")
                logger.info("This is normal - Auto-sklearn will handle predictions internally")
            
            return automl
            
        except Exception as e:
            logger.error(f"Auto-sklearn training failed: {e}")
            logger.info("Falling back to Random Forest...")
            return self.train_sklearn_fallback(X, y)
    
    def train_sklearn_fallback(self, X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
        """
        Train sklearn fallback model (Random Forest).
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Trained Random Forest model
        """
        logger.info("Training sklearn fallback model (Random Forest)")
        
        # Initialize Random Forest
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Train model
        rf.fit(X, y)
        
        logger.info("Random Forest training complete")
        
        return rf
    
    def evaluate_model(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                      X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate trained model.
        
        Args:
            model: Trained model
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating model performance")
        logger.info(f"Input shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")
        logger.info(f"Input columns - X_train: {list(X_train.columns)}")
        
        # Training predictions
        logger.info("Making training predictions...")
        y_train_pred = model.predict(X_train)
        y_train_pred_proba = model.predict_proba(X_train)[:, 1]
        
        # Test predictions
        logger.info("Making test predictions...")
        y_test_pred = model.predict(X_test)
        y_test_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'train_precision': precision_score(y_train, y_train_pred),
            'train_recall': recall_score(y_train, y_train_pred),
            'train_f1': f1_score(y_train, y_train_pred),
            'train_auc': roc_auc_score(y_train, y_train_pred_proba),
            
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'test_precision': precision_score(y_test, y_test_pred),
            'test_recall': recall_score(y_test, y_test_pred),
            'test_f1': f1_score(y_test, y_test_pred),
            'test_auc': roc_auc_score(y_test, y_test_pred_proba)
        }
        
        # Cross-validation on training set
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            scoring='accuracy',
            n_jobs=-1
        )
        
        metrics.update({
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std()
        })
        
        logger.info(f"Model evaluation complete:")
        logger.info(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
        logger.info(f"Test Precision: {metrics['test_precision']:.4f}")
        logger.info(f"Test Recall: {metrics['test_recall']:.4f}")
        logger.info(f"Test F1: {metrics['test_f1']:.4f}")
        logger.info(f"Test AUC: {metrics['test_auc']:.4f}")
        
        return metrics
    
    def train(self, feature_file: str = "data/features.parquet", 
              test_size: float = 0.2,
              use_temporal_split: bool = True,
              temporal_split_date: Optional[str] = None) -> Dict:
        """
        Complete training pipeline with temporal split to prevent data leakage.
        
        Args:
            feature_file: Path to feature Parquet file
            test_size: Test set size fraction (only used if temporal_split=False)
            use_temporal_split: If True, split by date (prevents leakage)
            temporal_split_date: Split date for temporal split
            
        Returns:
            Dictionary with training results
        """
        logger.info("Starting complete training pipeline")
        
        # Load data with dates
        df = pd.read_parquet(feature_file)
        
        # Separate features and target
        feature_columns = [col for col in df.columns if col not in ['lat', 'lon', 'date', 'target', 'is_fire', 'dtw_start', 'dtw_end']]
        X = df[feature_columns].copy()
        
        # Handle different target column names
        if 'target' in df.columns:
            y = df['target'].copy()
        elif 'is_fire' in df.columns:
            y = df['is_fire'].copy()
        else:
            raise ValueError("No target column found. Expected 'target' or 'is_fire'")
        
        # Handle missing values
        logger.info(f"Handling missing values...")
        logger.info(f"NaN counts before cleaning: {X.isnull().sum().sum()}")
        
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].median())
        X = X.fillna(0)
        
        # Ensure target is int
        if y.dtype == 'object':
            y = y.astype(bool)
        y = y.astype(int)
        
        # Check for infinite values
        if np.isinf(X.select_dtypes(include=[np.number])).any().any():
            logger.warning("Infinite values detected, replacing with 0")
            X = X.replace([np.inf, -np.inf], 0)
        
        logger.info(f"Loaded {len(X)} samples with {len(feature_columns)} features")
        logger.info(f"Target distribution: {dict(y.value_counts())}")
        logger.info(f"NaN counts after cleaning: {X.isnull().sum().sum()}")
        
        # Split data - TEMPORAL SPLIT TO PREVENT DATA LEAKAGE
        if use_temporal_split and 'date' in df.columns:
            logger.info(f"🔥 Using TEMPORAL SPLIT to prevent data leakage")
            df['date'] = pd.to_datetime(df['date'])
            
            # Calculate split date: if temporal_split_date provided, use it; otherwise calculate from percentile
            if temporal_split_date is not None:
                split_date = pd.to_datetime(temporal_split_date)
                logger.info(f"📊 Using provided split date: {split_date.date()}")
            else:
                # Calculate split date as (1 - test_size) percentile for 80/20 split
                split_date = df['date'].quantile(1 - test_size)
                logger.info(f"📊 Auto-calculated split date from {test_size*100:.0f}% test size: {split_date.date()}")
            
            train_mask = df['date'] < split_date
            test_mask = df['date'] >= split_date
            
            X_train = X[train_mask]
            X_test = X[test_mask]
            y_train = y[train_mask]
            y_test = y[test_mask]
            
            train_pct = len(X_train) / len(df) * 100
            test_pct = len(X_test) / len(df) * 100
            
            logger.info(f"📅 Train period: {df[train_mask]['date'].min()} to {df[train_mask]['date'].max()} ({train_pct:.1f}%)")
            logger.info(f"📅 Test period: {df[test_mask]['date'].min()} to {df[test_mask]['date'].max()} ({test_pct:.1f}%)")
            logger.info(f"📊 Split date: {split_date.date()}")
        else:
            logger.warning("⚠️  Using RANDOM SPLIT - this may cause data leakage!")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )
        
        logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test")
        
        # Apply feature selection
        X_train_selected = self.apply_feature_selection(X_train, y_train)
        logger.info(f"After training feature selection: {X_train_selected.shape}")
        logger.info(f"Training selected features: {list(X_train_selected.columns)}")
        
        # Apply same feature selection to test data
        X_test_selected = X_test[self.feature_names]  # Use the same selected features
        logger.info(f"After test feature selection: {X_test_selected.shape}")
        logger.info(f"Test selected features: {list(X_test_selected.columns)}")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train_selected),
            columns=X_train_selected.columns,
            index=X_train_selected.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test_selected),
            columns=X_test_selected.columns,
            index=X_test_selected.index
        )
        
        # Final data quality check after scaling
        logger.info("Performing final data quality check...")
        
        # Check for NaN after scaling
        if X_train_scaled.isnull().any().any():
            nan_counts = X_train_scaled.isnull().sum()
            logger.warning(f"NaN values detected after scaling: {nan_counts[nan_counts > 0]}")
            X_train_scaled = X_train_scaled.fillna(0)
            logger.info("Filled NaN values with 0")
        
        # Check for infinite values after scaling
        if np.isinf(X_train_scaled.select_dtypes(include=[np.number])).any().any():
            logger.warning("Infinite values detected after scaling")
            X_train_scaled = X_train_scaled.replace([np.inf, -np.inf], 0)
            logger.info("Replaced infinite values with 0")
        
        # Clean test set as well
        if X_test_scaled.isnull().any().any():
            X_test_scaled = X_test_scaled.fillna(0)
        if np.isinf(X_test_scaled.select_dtypes(include=[np.number])).any().any():
            X_test_scaled = X_test_scaled.replace([np.inf, -np.inf], 0)
        
        # Verify data quality
        logger.info(f"Final data check: NaN={X_train_scaled.isnull().sum().sum()}, Inf={np.isinf(X_train_scaled.select_dtypes(include=[np.number])).sum().sum()}")
        logger.info(f"Data types: {X_train_scaled.dtypes.value_counts().to_dict()}")
        
        # Check memory before training
        memory_percent = psutil.virtual_memory().percent
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        logger.info(f"Memory usage before training: {memory_percent:.1f}% ({available_memory_gb:.1f}GB available)")
        
        if memory_percent > 95:
            logger.warning(f"Very high memory usage ({memory_percent:.1f}%), switching to Random Forest")
            self.use_autosklearn = False
        elif memory_percent > 85:
            logger.warning(f"High memory usage ({memory_percent:.1f}%), using conservative Auto-sklearn settings")
        elif available_memory_gb < 8:
            logger.warning(f"Low available memory ({available_memory_gb:.1f}GB), using conservative settings")
        
        # Train model with error handling
        try:
            if self.use_autosklearn:
                logger.info("Training with Auto-sklearn...")
                self.model = self.train_autosklearn(X_train_scaled, y_train)
            else:
                logger.info("Training with Random Forest...")
                self.model = self.train_sklearn_fallback(X_train_scaled, y_train)
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            logger.info("Falling back to Random Forest...")
            self.model = self.train_sklearn_fallback(X_train_scaled, y_train)
            self.use_autosklearn = False  # Mark as fallback
        
        # Check memory after training
        memory_percent = psutil.virtual_memory().percent
        logger.info(f"Memory usage after training: {memory_percent:.1f}%")
        
        # Evaluate model with error handling
        try:
            logger.info("Evaluating model performance...")
            logger.info(f"Model input shapes - X_train_scaled: {X_train_scaled.shape}, X_test_scaled: {X_test_scaled.shape}")
            logger.info(f"Model input columns - X_train_scaled: {list(X_train_scaled.columns)}")
            self.training_results = self.evaluate_model(
                self.model, X_train_scaled, y_train, X_test_scaled, y_test
            )
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            logger.error(f"Error details: {type(e).__name__}")
            # Create minimal results to prevent complete failure
            self.training_results = {
                'model_type': 'Fallback',
                'error': str(e),
                'test_accuracy': 0.0,
                'test_precision': 0.0,
                'test_recall': 0.0,
                'test_f1': 0.0,
                'test_auc': 0.0
            }
        
        # Store model artifacts
        self.model_artifacts = {
            'model_type': 'autosklearn' if self.use_autosklearn else 'random_forest',
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'msfs': self.msfs,
            'training_results': self.training_results,
            'created_at': datetime.now().isoformat()
        }
        
        logger.info("Training pipeline complete")
        
        return self.training_results
    
    def save_model(self, output_dir: str = "model/artifacts"):
        """
        Save trained model and artifacts.
        
        Args:
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_file = output_path / "model.joblib"
        joblib.dump(self.model, model_file)
        logger.info(f"Model saved to {model_file}")
        
        # Save artifacts
        artifacts_file = output_path / "model_artifacts.joblib"
        joblib.dump(self.model_artifacts, artifacts_file)
        logger.info(f"Model artifacts saved to {artifacts_file}")
        
        # Save feature specification
        feature_spec = {
            'selected_features': self.feature_names,
            'num_features': len(self.feature_names),
            'feature_selection_method': 'MSFS',
            'model_type': self.model_artifacts['model_type'],
            'training_results': self.training_results,
            'created_at': self.model_artifacts['created_at']
        }
        
        spec_file = output_path / "feature_spec.json"
        with open(spec_file, 'w') as f:
            json.dump(feature_spec, f, indent=2)
        logger.info(f"Feature specification saved to {spec_file}")
        
        # Save MSFS results
        if self.msfs is not None:
            self.msfs.save_results(output_dir)
    
    def load_model(self, output_dir: str = "model/artifacts"):
        """
        Load trained model and artifacts.
        
        Args:
            output_dir: Input directory
        """
        output_path = Path(output_dir)
        
        # Load model
        model_file = output_path / "model.joblib"
        self.model = joblib.load(model_file)
        logger.info(f"Model loaded from {model_file}")
        
        # Load artifacts
        artifacts_file = output_path / "model_artifacts.joblib"
        self.model_artifacts = joblib.load(artifacts_file)
        logger.info(f"Model artifacts loaded from {artifacts_file}")
        
        # Restore components
        self.feature_names = self.model_artifacts['feature_names']
        self.scaler = self.model_artifacts['scaler']
        self.msfs = self.model_artifacts['msfs']
        self.training_results = self.model_artifacts['training_results']
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Apply feature selection
        X_selected = self.msfs.transform(X)
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_selected),
            columns=X_selected.columns,
            index=X_selected.index
        )
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        return predictions, probabilities

def main():
    """Example usage of trainer."""
    # Initialize trainer with production parameters
    trainer = FirePredictionTrainer(
        use_autosklearn=AUTOSKLEARN_AVAILABLE
    )
    
    # Train model
    results = trainer.train()
    
    # Save model
    trainer.save_model()
    
    print("Training Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()

