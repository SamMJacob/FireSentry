"""
Multi-Stage Feature Selection (MSFS) Implementation

Implements the three-stage feature selection approach from the base paper:
1. Mutual Information Gain (MIG) filtering
2. Recursive Feature Elimination with Cross-Validation (RFECV)
3. Voting aggregation across multiple runs

Author: FireSentry Team
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging
from typing import List, Tuple, Dict, Optional
import json
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)

class MultiStageFeatureSelection:
    """
    Multi-Stage Feature Selection (MSFS) implementation.
    
    Combines filter methods (MIG) for computational efficiency with wrapper methods (RFECV)
    for accuracy, then aggregates results via voting for stability.
    """
    
    def __init__(self, 
                 k_mi: int = 12, 
                 n_repeats: int = 3, 
                 cv_folds: int = 5,
                 random_state: int = 42):
        """
        Initialize MSFS selector.
        
        Args:
            k_mi: Number of top features to select in MIG stage
            n_repeats: Number of voting iterations
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
        """
        self.k_mi = k_mi
        self.n_repeats = n_repeats
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # Results storage
        self.mi_scores_ = None
        self.selected_features_ = None
        self.feature_importance_ = None
        self.voting_results_ = None
        
        logger.info(f"MSFS initialized: k_mi={k_mi}, n_repeats={n_repeats}, cv_folds={cv_folds}")
    
    def stage1_mutual_information_gain(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Stage 1: Mutual Information Gain filtering.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            List of selected feature names
        """
        logger.info("Stage 1: Computing Mutual Information Gain scores")
        
        # Compute MI scores
        mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
        
        # Create feature-score pairs
        feature_scores = list(zip(X.columns, mi_scores))
        
        # Sort by MI score (descending)
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top k_mi features
        selected_features = [feat for feat, score in feature_scores[:self.k_mi]]
        
        # Store results
        self.mi_scores_ = dict(feature_scores)
        
        logger.info(f"Stage 1 complete: Selected {len(selected_features)} features")
        logger.info(f"Top features: {selected_features[:5]}")
        
        return selected_features
    
    def stage2_rfecv(self, X: pd.DataFrame, y: pd.Series, 
                    feature_names: List[str]) -> Tuple[List[str], float]:
        """
        Stage 2: Recursive Feature Elimination with Cross-Validation.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: Features to consider (from Stage 1)
            
        Returns:
            Tuple of (selected_features, best_score)
        """
        logger.info(f"Stage 2: RFECV with {len(feature_names)} features")
        
        # Subset features
        X_subset = X[feature_names]
        
        # Initialize base estimator
        rf_estimator = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Initialize RFECV
        rfecv = RFECV(
            estimator=rf_estimator,
            step=1,
            cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
            scoring='accuracy',
            n_jobs=-1
        )
        
        # Fit RFECV
        rfecv.fit(X_subset, y)
        
        # Get selected features
        selected_mask = rfecv.support_
        selected_features = [feat for feat, selected in zip(feature_names, selected_mask) if selected]
        
        # Get best score
        best_score = rfecv.cv_results_['mean_test_score'][rfecv.n_features_ - 1]
        
        # Store feature importance
        self.feature_importance_ = dict(zip(feature_names, rfecv.estimator_.feature_importances_))
        
        logger.info(f"Stage 2 complete: Selected {len(selected_features)} features")
        logger.info(f"Best CV score: {best_score:.4f}")
        logger.info(f"Selected features: {selected_features}")
        
        return selected_features, best_score
    
    def stage3_voting_aggregation(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Stage 3: Voting aggregation across multiple runs.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Final selected features
        """
        logger.info(f"Stage 3: Voting aggregation over {self.n_repeats} runs")
        
        all_selections = []
        voting_scores = {}
        
        for run in range(self.n_repeats):
            logger.info(f"Voting run {run + 1}/{self.n_repeats}")
            
            # Set different random seed for each run
            run_seed = self.random_state + run
            
            # Stage 1: MIG with run-specific seed
            mi_scores = mutual_info_classif(X, y, random_state=run_seed)
            feature_scores = list(zip(X.columns, mi_scores))
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            mi_features = [feat for feat, score in feature_scores[:self.k_mi]]
            
            # Stage 2: RFECV with run-specific seed
            X_subset = X[mi_features]
            rf_estimator = RandomForestClassifier(
                n_estimators=100,
                random_state=run_seed,
                n_jobs=-1
            )
            
            rfecv = RFECV(
                estimator=rf_estimator,
                step=1,
                cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=run_seed),
                scoring='accuracy',
                n_jobs=-1
            )
            
            rfecv.fit(X_subset, y)
            selected_mask = rfecv.support_
            selected_features = [feat for feat, selected in zip(mi_features, selected_mask) if selected]
            
            all_selections.append(selected_features)
            
            # Track voting scores
            for feat in selected_features:
                voting_scores[feat] = voting_scores.get(feat, 0) + 1
        
        # Apply majority voting (features appearing in >50% of runs)
        majority_threshold = self.n_repeats / 2
        final_features = [
            feat for feat, count in voting_scores.items() 
            if count > majority_threshold
        ]
        
        # Sort by voting frequency
        final_features.sort(key=lambda x: voting_scores[x], reverse=True)
        
        # Store voting results
        self.voting_results_ = {
            'all_selections': all_selections,
            'voting_scores': voting_scores,
            'final_features': final_features,
            'majority_threshold': majority_threshold
        }
        
        logger.info(f"Stage 3 complete: {len(final_features)} features selected by majority vote")
        logger.info(f"Final features: {final_features}")
        
        return final_features
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'MultiStageFeatureSelection':
        """
        Fit MSFS to training data.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Self
        """
        logger.info(f"Starting MSFS with {X.shape[1]} features and {len(y)} samples")
        
        # Run all three stages
        final_features = self.stage3_voting_aggregation(X, y)
        
        # Store final selection
        self.selected_features_ = final_features
        
        logger.info(f"MSFS complete: Selected {len(final_features)} features")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform feature matrix to selected features only.
        
        Args:
            X: Feature matrix
            
        Returns:
            Transformed feature matrix
        """
        if self.selected_features_ is None:
            raise ValueError("MSFS must be fitted before transform")
        
        return X[self.selected_features_]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit MSFS and transform feature matrix.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Transformed feature matrix
        """
        return self.fit(X, y).transform(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.feature_importance_ is None:
            logger.warning("Feature importance not available - run fit() first")
            return {}
        
        return self.feature_importance_.copy()
    
    def get_mi_scores(self) -> Dict[str, float]:
        """
        Get mutual information scores.
        
        Returns:
            Dictionary mapping feature names to MI scores
        """
        if self.mi_scores_ is None:
            logger.warning("MI scores not available - run fit() first")
            return {}
        
        return self.mi_scores_.copy()
    
    def get_voting_results(self) -> Dict:
        """
        Get voting aggregation results.
        
        Returns:
            Dictionary with voting results
        """
        if self.voting_results_ is None:
            logger.warning("Voting results not available - run fit() first")
            return {}
        
        return self.voting_results_.copy()
    
    def evaluate_selection(self, X: pd.DataFrame, y: pd.Series, 
                          X_test: pd.DataFrame = None, y_test: pd.Series = None) -> Dict:
        """
        Evaluate the selected features using cross-validation.
        
        Args:
            X: Training feature matrix
            y: Training target variable
            X_test: Test feature matrix (optional)
            y_test: Test target variable (optional)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.selected_features_ is None:
            raise ValueError("MSFS must be fitted before evaluation")
        
        # Transform features
        X_selected = self.transform(X)
        
        # Initialize classifier
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Cross-validation evaluation
        cv_scores = cross_val_score(
            rf, X_selected, y,
            cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
            scoring='accuracy',
            n_jobs=-1
        )
        
        results = {
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'cv_accuracy_scores': cv_scores.tolist(),
            'num_features': len(self.selected_features_),
            'selected_features': self.selected_features_
        }
        
        # Test set evaluation if provided
        if X_test is not None and y_test is not None:
            X_test_selected = self.transform(X_test)
            rf.fit(X_selected, y)
            
            y_pred = rf.predict(X_test_selected)
            y_pred_proba = rf.predict_proba(X_test_selected)[:, 1]
            
            results.update({
                'test_accuracy': accuracy_score(y_test, y_pred),
                'test_precision': precision_score(y_test, y_pred),
                'test_recall': recall_score(y_test, y_pred),
                'test_f1': f1_score(y_test, y_pred),
                'test_auc': roc_auc_score(y_test, y_pred_proba)
            })
        
        return results
    
    def save_results(self, output_dir: str = "model/artifacts"):
        """
        Save MSFS results to files.
        
        Args:
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save selected features
        if self.selected_features_ is not None:
            feature_spec = {
                'selected_features': self.selected_features_,
                'num_features': len(self.selected_features_),
                'parameters': {
                    'k_mi': self.k_mi,
                    'n_repeats': self.n_repeats,
                    'cv_folds': self.cv_folds,
                    'random_state': self.random_state
                }
            }
            
            with open(output_path / "msfs_feature_spec.json", 'w') as f:
                json.dump(feature_spec, f, indent=2)
            
            logger.info(f"Feature specification saved to {output_path / 'msfs_feature_spec.json'}")
        
        # Save MSFS object
        joblib.dump(self, output_path / "msfs_selector.joblib")
        logger.info(f"MSFS selector saved to {output_path / 'msfs_selector.joblib'}")
    
    @classmethod
    def load_results(cls, output_dir: str = "model/artifacts") -> 'MultiStageFeatureSelection':
        """
        Load MSFS results from files.
        
        Args:
            output_dir: Input directory
            
        Returns:
            Loaded MSFS selector
        """
        output_path = Path(output_dir)
        selector = joblib.load(output_path / "msfs_selector.joblib")
        logger.info(f"MSFS selector loaded from {output_path / 'msfs_selector.joblib'}")
        return selector

def main():
    """Example usage of MSFS."""
    # Create sample data
    np.random.seed(42)
    n_samples, n_features = 1000, 24
    
    # Generate features with different relevance levels
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Create target with some features being relevant
    y = (X['feature_0'] + X['feature_5'] + X['feature_10'] + 
         np.random.randn(n_samples) * 0.1 > 0).astype(int)
    
    # Initialize and fit MSFS
    msfs = MultiStageFeatureSelection(k_mi=12, n_repeats=3)
    X_selected = msfs.fit_transform(X, y)
    
    print(f"Original features: {X.shape[1]}")
    print(f"Selected features: {X_selected.shape[1]}")
    print(f"Selected features: {msfs.selected_features_}")
    
    # Evaluate selection
    results = msfs.evaluate_selection(X, y)
    print(f"CV Accuracy: {results['cv_accuracy_mean']:.4f} ± {results['cv_accuracy_std']:.4f}")
    
    # Save results
    msfs.save_results()

if __name__ == "__main__":
    main()



