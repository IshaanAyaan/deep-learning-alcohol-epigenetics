"""
=============================================================================
STATISTICAL ANALYSIS MODULE
=============================================================================
AlcoholMethylationML Project
Author: Ishaan Ranjan
Date: January 2026

This module provides comprehensive statistical analysis functions for
evaluating methylation-based prediction models and comparing epigenetic
age acceleration between groups.

ANALYSIS COMPONENTS:

1. MODEL EVALUATION METRICS
   - ROC-AUC with confidence intervals
   - Precision, recall, F1-score
   - Sensitivity, specificity
   - Confusion matrix analysis

2. CROSS-VALIDATION ANALYSIS
   - Stratified k-fold CV
   - Nested CV for hyperparameter tuning
   - Monte Carlo cross-validation

3. STATISTICAL TESTS
   - t-tests and Mann-Whitney U tests
   - Multiple testing correction (FDR)
   - Effect size calculations

4. BOOTSTRAP INFERENCE
   - Confidence intervals for AUC
   - Feature importance stability

REFERENCES:
- DeLong et al. (1988): Comparing AUCs
- Benjamini & Hochberg (1995): FDR correction
- Efron & Tibshirani (1993): Bootstrap methods
=============================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass


@dataclass
class EvaluationResults:
    """Container for model evaluation results."""
    model_name: str
    auc: float
    auc_ci_low: float
    auc_ci_high: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    sensitivity: float
    specificity: float
    confusion_matrix: np.ndarray
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = 'Model',
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    verbose: bool = True
) -> EvaluationResults:
    """
    Calculate comprehensive evaluation metrics.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_proba : np.ndarray
        Predicted probabilities for positive class
    model_name : str
        Name of model for reporting
    n_bootstrap : int
        Number of bootstrap samples for CI
    ci_level : float
        Confidence interval level
    verbose : bool
        Print results
        
    Returns:
    --------
    EvaluationResults
        Comprehensive evaluation metrics
    """
    # Basic metrics
    auc = roc_auc_score(y_true, y_proba)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    
    # Bootstrap CI for AUC
    auc_bootstrap = []
    n_samples = len(y_true)
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n_samples, n_samples, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            auc_boot = roc_auc_score(y_true[idx], y_proba[idx])
            auc_bootstrap.append(auc_boot)
        except:
            continue
    
    alpha = 1 - ci_level
    auc_ci_low = np.percentile(auc_bootstrap, alpha/2 * 100) if auc_bootstrap else auc
    auc_ci_high = np.percentile(auc_bootstrap, (1-alpha/2) * 100) if auc_bootstrap else auc
    
    results = EvaluationResults(
        model_name=model_name,
        auc=auc,
        auc_ci_low=auc_ci_low,
        auc_ci_high=auc_ci_high,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        sensitivity=sensitivity,
        specificity=specificity,
        confusion_matrix=cm,
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds
    )
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"EVALUATION RESULTS: {model_name}")
        print('='*50)
        print(f"AUC: {auc:.4f} (95% CI: {auc_ci_low:.4f} - {auc_ci_high:.4f})")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Sensitivity: {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TN={tn}, FP={fp}")
        print(f"  FN={fn}, TP={tp}")
    
    return results


def compare_models(
    results_list: List[EvaluationResults],
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compare multiple model evaluation results.
    
    Parameters:
    -----------
    results_list : List[EvaluationResults]
        List of evaluation results
    verbose : bool
        Print comparison table
        
    Returns:
    --------
    pd.DataFrame
        Comparison table
    """
    comparison = []
    
    for r in results_list:
        comparison.append({
            'Model': r.model_name,
            'AUC': r.auc,
            'AUC_CI_Low': r.auc_ci_low,
            'AUC_CI_High': r.auc_ci_high,
            'Accuracy': r.accuracy,
            'Precision': r.precision,
            'Recall': r.recall,
            'F1': r.f1,
            'Sensitivity': r.sensitivity,
            'Specificity': r.specificity
        })
    
    df = pd.DataFrame(comparison)
    
    if verbose:
        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)
        print(df.to_string(index=False))
    
    return df


def delong_test(
    y_true: np.ndarray,
    y_proba_1: np.ndarray,
    y_proba_2: np.ndarray
) -> Tuple[float, float]:
    """
    DeLong test for comparing two AUCs.
    
    Tests whether two AUC values are significantly different.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_proba_1 : np.ndarray
        Probabilities from model 1
    y_proba_2 : np.ndarray
        Probabilities from model 2
        
    Returns:
    --------
    Tuple[float, float]
        Z-statistic and p-value
    """
    # Simplified DeLong test using bootstrap
    n_bootstrap = 1000
    n_samples = len(y_true)
    
    auc_diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n_samples, n_samples, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            auc1 = roc_auc_score(y_true[idx], y_proba_1[idx])
            auc2 = roc_auc_score(y_true[idx], y_proba_2[idx])
            auc_diffs.append(auc1 - auc2)
        except:
            continue
    
    if not auc_diffs:
        return 0.0, 1.0
    
    auc_diffs = np.array(auc_diffs)
    z_stat = np.mean(auc_diffs) / (np.std(auc_diffs) + 1e-10)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    return z_stat, p_value


def cross_validation_analysis(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    verbose: bool = True
) -> Dict:
    """
    Perform cross-validation analysis.
    
    Parameters:
    -----------
    model : sklearn-compatible model
        Model to evaluate
    X : np.ndarray
        Features
    y : np.ndarray
        Labels
    cv_folds : int
        Number of folds
    verbose : bool
        Print results
        
    Returns:
    --------
    Dict
        Cross-validation results
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Collect metrics per fold
    fold_results = {
        'auc': [], 'accuracy': [], 'precision': [], 
        'recall': [], 'f1': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Fit and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        fold_results['auc'].append(roc_auc_score(y_val, y_proba))
        fold_results['accuracy'].append(accuracy_score(y_val, y_pred))
        fold_results['precision'].append(precision_score(y_val, y_pred, zero_division=0))
        fold_results['recall'].append(recall_score(y_val, y_pred, zero_division=0))
        fold_results['f1'].append(f1_score(y_val, y_pred, zero_division=0))
    
    # Summary statistics
    results = {}
    for metric, values in fold_results.items():
        results[f'{metric}_mean'] = np.mean(values)
        results[f'{metric}_std'] = np.std(values)
        results[f'{metric}_values'] = values
    
    if verbose:
        print("\n" + "=" * 50)
        print(f"CROSS-VALIDATION RESULTS ({cv_folds}-fold)")
        print("=" * 50)
        for metric in ['auc', 'accuracy', 'precision', 'recall', 'f1']:
            mean = results[f'{metric}_mean']
            std = results[f'{metric}_std']
            print(f"{metric.upper():12s}: {mean:.4f} ± {std:.4f}")
    
    return results


def group_statistical_comparison(
    values_group1: np.ndarray,
    values_group2: np.ndarray,
    group1_name: str = 'Control',
    group2_name: str = 'Case',
    variable_name: str = 'Variable',
    verbose: bool = True
) -> Dict:
    """
    Statistical comparison between two groups.
    
    Performs t-test and Mann-Whitney U test, calculates effect sizes.
    
    Parameters:
    -----------
    values_group1 : np.ndarray
        Values for group 1
    values_group2 : np.ndarray
        Values for group 2
    group1_name : str
        Name of group 1
    group2_name : str
        Name of group 2
    variable_name : str
        Name of variable being compared
    verbose : bool
        Print results
        
    Returns:
    --------
    Dict
        Statistical comparison results
    """
    # Descriptive statistics
    mean1, std1 = np.mean(values_group1), np.std(values_group1)
    mean2, std2 = np.mean(values_group2), np.std(values_group2)
    n1, n2 = len(values_group1), len(values_group2)
    
    # t-test (independent samples)
    t_stat, t_pval = stats.ttest_ind(values_group1, values_group2)
    
    # Mann-Whitney U test (non-parametric)
    u_stat, u_pval = stats.mannwhitneyu(
        values_group1, values_group2, alternative='two-sided'
    )
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(
        ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
    )
    cohens_d = (mean2 - mean1) / pooled_std if pooled_std > 0 else 0
    
    # Hedge's g (bias-corrected)
    correction = 1 - 3 / (4 * (n1 + n2) - 9)
    hedges_g = cohens_d * correction
    
    results = {
        'variable': variable_name,
        'group1_name': group1_name,
        'group2_name': group2_name,
        'mean1': mean1,
        'std1': std1,
        'n1': n1,
        'mean2': mean2,
        'std2': std2,
        'n2': n2,
        'mean_difference': mean2 - mean1,
        't_statistic': t_stat,
        't_pvalue': t_pval,
        'u_statistic': u_stat,
        'u_pvalue': u_pval,
        'cohens_d': cohens_d,
        'hedges_g': hedges_g
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"STATISTICAL COMPARISON: {variable_name}")
        print("=" * 60)
        print(f"\n{group1_name} (n={n1}):")
        print(f"  Mean: {mean1:.4f} ± {std1:.4f}")
        print(f"\n{group2_name} (n={n2}):")
        print(f"  Mean: {mean2:.4f} ± {std2:.4f}")
        print(f"\nDifference: {mean2 - mean1:.4f}")
        print(f"\nStatistical Tests:")
        print(f"  t-test: t={t_stat:.3f}, p={t_pval:.2e}")
        print(f"  Mann-Whitney U: U={u_stat:.1f}, p={u_pval:.2e}")
        print(f"\nEffect Size:")
        print(f"  Cohen's d: {cohens_d:.3f}")
        print(f"  Hedge's g: {hedges_g:.3f}")
        
        # Interpret effect size
        if abs(cohens_d) < 0.2:
            effect_interp = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_interp = "small"
        elif abs(cohens_d) < 0.8:
            effect_interp = "medium"
        else:
            effect_interp = "large"
        print(f"  Interpretation: {effect_interp}")
    
    return results


def multiple_testing_correction(
    p_values: np.ndarray,
    method: str = 'fdr_bh',
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply multiple testing correction.
    
    Parameters:
    -----------
    p_values : np.ndarray
        Uncorrected p-values
    method : str
        Correction method: 'bonferroni' or 'fdr_bh'
    alpha : float
        Significance threshold
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Corrected p-values and significance mask
    """
    n_tests = len(p_values)
    
    if method == 'bonferroni':
        p_corrected = np.minimum(p_values * n_tests, 1.0)
        significant = p_corrected < alpha
        
    elif method == 'fdr_bh':
        # Benjamini-Hochberg FDR
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]
        
        # Calculate critical values
        ranks = np.arange(1, n_tests + 1)
        critical = ranks * alpha / n_tests
        
        # Find threshold
        significant_sorted = sorted_p <= critical
        
        # Determine adjusted p-values
        p_corrected = np.zeros(n_tests)
        for i in range(n_tests - 1, -1, -1):
            if i == n_tests - 1:
                p_corrected[sorted_idx[i]] = sorted_p[i]
            else:
                p_corrected[sorted_idx[i]] = min(
                    p_corrected[sorted_idx[i + 1]],
                    sorted_p[i] * n_tests / (i + 1)
                )
        p_corrected = np.minimum(p_corrected, 1.0)
        significant = p_corrected < alpha
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return p_corrected, significant


def feature_importance_analysis(
    importance_scores: np.ndarray,
    feature_names: List[str],
    n_top: int = 20,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Analyze and rank feature importance.
    
    Parameters:
    -----------
    importance_scores : np.ndarray
        Feature importance values
    feature_names : List[str]
        Feature names
    n_top : int
        Number of top features to show
    verbose : bool
        Print results
        
    Returns:
    --------
    pd.DataFrame
        Ranked feature importance
    """
    # Create DataFrame
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    })
    
    # Sort by importance
    df = df.sort_values('importance', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    
    # Normalize importance
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df['importance_cumulative'] = df['importance_normalized'].cumsum()
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"TOP {n_top} IMPORTANT FEATURES")
        print("=" * 60)
        for i in range(min(n_top, len(df))):
            row = df.iloc[i]
            print(f"{row['rank']:3d}. {row['feature']:30s} "
                  f"Importance: {row['importance']:.4f} "
                  f"({row['importance_normalized']:.2%})")
    
    return df


if __name__ == "__main__":
    # Demo statistical analysis
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Statistical Analysis")
    print("=" * 60)
    
    # Create sample predictions
    np.random.seed(42)
    n = 100
    
    y_true = np.random.randint(0, 2, n)
    y_proba = np.clip(y_true * 0.6 + np.random.randn(n) * 0.2 + 0.2, 0, 1)
    y_pred = (y_proba > 0.5).astype(int)
    
    # Calculate metrics
    results = calculate_metrics(
        y_true, y_pred, y_proba,
        model_name='Demo Model',
        verbose=True
    )
    
    # Group comparison
    group1_values = np.random.normal(0, 1, 50)  # Control
    group2_values = np.random.normal(0.5, 1.2, 50)  # Case (higher)
    
    comparison = group_statistical_comparison(
        group1_values, group2_values,
        group1_name='Control',
        group2_name='Alcohol',
        variable_name='Age Acceleration',
        verbose=True
    )
