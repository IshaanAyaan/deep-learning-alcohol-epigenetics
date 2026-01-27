"""
=============================================================================
BASELINE MACHINE LEARNING MODELS
=============================================================================
AlcoholMethylationML Project
Author: Ishaan Ranjan
Date: January 2026

This module implements baseline machine learning models for predicting
alcohol use from DNA methylation data. These serve as comparison points
for the novel deep learning architecture.

IMPLEMENTED MODELS:

1. ELASTIC NET LOGISTIC REGRESSION
   - L1 + L2 regularization
   - Ideal for high-dimensional sparse data like methylation
   - Provides interpretable coefficients
   - Standard in EWAS prediction studies

2. RIDGE LOGISTIC REGRESSION
   - L2 regularization only
   - Good when all features contribute
   - Comparison baseline

3. XGBOOST CLASSIFIER
   - Gradient boosting decision trees
   - Captures nonlinear interactions
   - State-of-the-art for tabular data
   - Can reveal complex methylation patterns

4. RANDOM FOREST
   - Ensemble of decision trees
   - Robust to overfitting
   - Provides feature importance

These baselines establish performance benchmarks and help validate
that the deep learning model provides meaningful improvements.

REFERENCES:
- Zou & Hastie (2005): Elastic Net
- Chen & Guestrin (2016): XGBoost
- Breiman (2001): Random Forests
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
import warnings

# XGBoost is optional
XGBOOST_AVAILABLE = False
xgb = None
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except (ImportError, Exception):
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Skipping XGBoost model.")


@dataclass
class ModelResult:
    """
    Container for model training results.
    
    Stores the trained model along with performance metrics
    and feature importance (if available).
    """
    model: Any
    model_name: str
    cv_scores: np.ndarray
    mean_cv_auc: float
    std_cv_auc: float
    feature_importance: Optional[np.ndarray]
    feature_names: Optional[List[str]]


class ElasticNetClassifier:
    """
    Elastic Net Logistic Regression for methylation classification.
    
    Elastic Net combines L1 (LASSO) and L2 (Ridge) regularization:
    - L1 promotes sparsity (selects important CpGs)
    - L2 handles correlated features (common in methylation)
    
    The regularization path (alpha) is selected via cross-validation.
    
    This is the GOLD STANDARD for high-dimensional EWAS prediction.
    """
    
    def __init__(
        self,
        l1_ratio: float = 0.5,
        cv: int = 5,
        max_iter: int = 1000,
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Initialize Elastic Net classifier.
        
        Parameters:
        -----------
        l1_ratio : float
            Balance between L1 and L2 (0.5 = equal)
            l1_ratio=1 is LASSO, l1_ratio=0 is Ridge
        cv : int
            Cross-validation folds for alpha selection
        max_iter : int
            Maximum iterations for convergence
        random_state : int
            Random seed
        verbose : bool
            Print progress
        """
        self.l1_ratio = l1_ratio
        self.cv = cv
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        
        # Model will be fitted
        self.model_ = None
        self.scaler_ = None
        self.best_alpha_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ElasticNetClassifier':
        """
        Fit Elastic Net classifier with cross-validation.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
            
        Returns:
        --------
        self
        """
        if self.verbose:
            print("\n" + "=" * 50)
            print("TRAINING ELASTIC NET CLASSIFIER")
            print("=" * 50)
        
        # Standardize features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        if self.verbose:
            print(f"Features: {X.shape[1]}")
            print(f"Samples: {X.shape[0]}")
            print(f"L1 ratio: {self.l1_ratio}")
        
        # Fit with cross-validation
        self.model_ = LogisticRegressionCV(
            penalty='elasticnet',
            solver='saga',
            l1_ratios=[self.l1_ratio],
            cv=self.cv,
            scoring='roc_auc',
            max_iter=self.max_iter,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.model_.fit(X_scaled, y)
        self.best_alpha_ = self.model_.C_[0]
        
        if self.verbose:
            print(f"Best regularization (C): {self.best_alpha_:.4f}")
            n_nonzero = np.sum(self.model_.coef_ != 0)
            print(f"Non-zero coefficients: {n_nonzero}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        X_scaled = self.scaler_.transform(X)
        return self.model_.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        X_scaled = self.scaler_.transform(X)
        return self.model_.predict_proba(X_scaled)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get absolute coefficient values as importance."""
        return np.abs(self.model_.coef_[0])


class XGBoostClassifier:
    """
    XGBoost classifier for methylation-based prediction.
    
    XGBoost is a gradient boosting algorithm that:
    - Captures nonlinear relationships between CpGs
    - Handles feature interactions automatically
    - Is robust to different scales
    - Provides feature importance
    
    Often outperforms linear models when there are complex patterns.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Initialize XGBoost classifier.
        
        Parameters:
        -----------
        n_estimators : int
            Number of boosting rounds
        max_depth : int
            Maximum tree depth
        learning_rate : float
            Step size shrinkage
        subsample : float
            Fraction of samples per tree
        colsample_bytree : float
            Fraction of features per tree
        random_state : int
            Random seed
        verbose : bool
            Print progress
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.verbose = verbose
        
        self.model_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'XGBoostClassifier':
        """Fit XGBoost classifier."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available")
        
        if self.verbose:
            print("\n" + "=" * 50)
            print("TRAINING XGBOOST CLASSIFIER")
            print("=" * 50)
            print(f"Features: {X.shape[1]}")
            print(f"Samples: {X.shape[0]}")
        
        self.model_ = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            eval_metric='auc',
            use_label_encoder=False
        )
        
        self.model_.fit(X, y, verbose=False)
        
        if self.verbose:
            print(f"Trees: {self.n_estimators}")
            print(f"Max depth: {self.max_depth}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model_.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model_.predict_proba(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance from trained model."""
        return self.model_.feature_importances_


class RandomForestModel:
    """
    Random Forest classifier for methylation prediction.
    
    Random Forest:
    - Ensemble of decision trees with bootstrap sampling
    - Robust to overfitting
    - Handles high-dimensional data well
    - Provides feature importance via impurity decrease
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 5,
        max_features: str = 'sqrt',
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Initialize Random Forest classifier.
        
        Parameters:
        -----------
        n_estimators : int
            Number of trees
        max_depth : int, optional
            Maximum tree depth (None = unlimited)
        min_samples_split : int
            Minimum samples to split node
        max_features : str
            Features to consider at each split
        random_state : int
            Random seed
        verbose : bool
            Print progress
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.verbose = verbose
        
        self.model_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestModel':
        """Fit Random Forest classifier."""
        if self.verbose:
            print("\n" + "=" * 50)
            print("TRAINING RANDOM FOREST CLASSIFIER")
            print("=" * 50)
            print(f"Features: {X.shape[1]}")
            print(f"Samples: {X.shape[0]}")
        
        self.model_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.model_.fit(X, y)
        
        if self.verbose:
            print(f"Trees: {self.n_estimators}")
            print(f"Max features: {self.max_features}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model_.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model_.predict_proba(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance from trained model."""
        return self.model_.feature_importances_


def train_baseline_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    cv_folds: int = 5,
    verbose: bool = True
) -> Dict[str, ModelResult]:
    """
    Train all baseline models and evaluate performance.
    
    This function trains multiple baseline models and returns
    comprehensive results for comparison with the deep learning model.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    feature_names : List[str]
        Feature names for importance analysis
    cv_folds : int
        Cross-validation folds
    verbose : bool
        Print progress
        
    Returns:
    --------
    Dict[str, ModelResult]
        Results for each model
    """
    from sklearn.metrics import roc_auc_score
    
    if verbose:
        print("\n" + "=" * 60)
        print("TRAINING BASELINE MODELS")
        print("=" * 60)
    
    results = {}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # 1. Elastic Net
    if verbose:
        print("\n" + "-" * 40)
        print("Model 1: Elastic Net")
    
    elastic_net = ElasticNetClassifier(l1_ratio=0.5, verbose=verbose)
    elastic_net.fit(X_train, y_train)
    
    # Cross-validation scores
    cv_scores_en = cross_val_score(
        elastic_net.model_, 
        elastic_net.scaler_.transform(X_train), 
        y_train,
        cv=cv, scoring='roc_auc'
    )
    
    # Test prediction
    y_pred_en = elastic_net.predict_proba(X_test)[:, 1]
    test_auc_en = roc_auc_score(y_test, y_pred_en)
    
    results['elastic_net'] = ModelResult(
        model=elastic_net,
        model_name='Elastic Net',
        cv_scores=cv_scores_en,
        mean_cv_auc=cv_scores_en.mean(),
        std_cv_auc=cv_scores_en.std(),
        feature_importance=elastic_net.get_feature_importance(),
        feature_names=feature_names
    )
    
    if verbose:
        print(f"CV AUC: {cv_scores_en.mean():.3f} ± {cv_scores_en.std():.3f}")
        print(f"Test AUC: {test_auc_en:.3f}")
    
    # 2. Random Forest
    if verbose:
        print("\n" + "-" * 40)
        print("Model 2: Random Forest")
    
    rf = RandomForestModel(n_estimators=100, verbose=verbose)
    rf.fit(X_train, y_train)
    
    cv_scores_rf = cross_val_score(
        rf.model_, X_train, y_train,
        cv=cv, scoring='roc_auc'
    )
    
    y_pred_rf = rf.predict_proba(X_test)[:, 1]
    test_auc_rf = roc_auc_score(y_test, y_pred_rf)
    
    results['random_forest'] = ModelResult(
        model=rf,
        model_name='Random Forest',
        cv_scores=cv_scores_rf,
        mean_cv_auc=cv_scores_rf.mean(),
        std_cv_auc=cv_scores_rf.std(),
        feature_importance=rf.get_feature_importance(),
        feature_names=feature_names
    )
    
    if verbose:
        print(f"CV AUC: {cv_scores_rf.mean():.3f} ± {cv_scores_rf.std():.3f}")
        print(f"Test AUC: {test_auc_rf:.3f}")
    
    # 3. XGBoost (if available)
    if XGBOOST_AVAILABLE:
        if verbose:
            print("\n" + "-" * 40)
            print("Model 3: XGBoost")
        
        xgb_model = XGBoostClassifier(verbose=verbose)
        xgb_model.fit(X_train, y_train)
        
        cv_scores_xgb = cross_val_score(
            xgb_model.model_, X_train, y_train,
            cv=cv, scoring='roc_auc'
        )
        
        y_pred_xgb = xgb_model.predict_proba(X_test)[:, 1]
        test_auc_xgb = roc_auc_score(y_test, y_pred_xgb)
        
        results['xgboost'] = ModelResult(
            model=xgb_model,
            model_name='XGBoost',
            cv_scores=cv_scores_xgb,
            mean_cv_auc=cv_scores_xgb.mean(),
            std_cv_auc=cv_scores_xgb.std(),
            feature_importance=xgb_model.get_feature_importance(),
            feature_names=feature_names
        )
        
        if verbose:
            print(f"CV AUC: {cv_scores_xgb.mean():.3f} ± {cv_scores_xgb.std():.3f}")
            print(f"Test AUC: {test_auc_xgb:.3f}")
    
    if verbose:
        print("\n" + "=" * 60)
        print("BASELINE MODEL TRAINING COMPLETE")
        print("=" * 60)
        print("\nSummary:")
        for name, result in results.items():
            print(f"  {result.model_name}: AUC = {result.mean_cv_auc:.3f} ± {result.std_cv_auc:.3f}")
    
    return results


def get_top_features(
    result: ModelResult,
    n_top: int = 20
) -> pd.DataFrame:
    """
    Get top features by importance from trained model.
    
    Parameters:
    -----------
    result : ModelResult
        Result object from model training
    n_top : int
        Number of top features to return
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with top features and importance scores
    """
    if result.feature_importance is None:
        return pd.DataFrame()
    
    importance = result.feature_importance
    names = result.feature_names
    
    # Sort by importance
    indices = np.argsort(importance)[-n_top:][::-1]
    
    return pd.DataFrame({
        'feature': [names[i] for i in indices],
        'importance': importance[indices]
    })


if __name__ == "__main__":
    # Demo baseline models
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Baseline Models")
    print("=" * 60)
    
    # Create synthetic data
    np.random.seed(42)
    n_samples, n_features = 400, 500
    
    X = np.random.randn(n_samples, n_features)
    # Add signal to first 20 features
    signal = np.random.randn(n_samples) * 2
    X[:, :20] += signal.reshape(-1, 1) * 0.3
    
    y = (signal > 0).astype(int)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Train models
    results = train_baseline_models(
        X_train, y_train, X_test, y_test,
        feature_names, verbose=True
    )
    
    # Show top features
    print("\n\nTop 10 Features (Elastic Net):")
    print(get_top_features(results['elastic_net'], n_top=10))
