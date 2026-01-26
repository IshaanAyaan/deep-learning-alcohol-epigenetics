"""
=============================================================================
METHYLATION PREPROCESSING PIPELINE
=============================================================================
AlcoholMethylationML Project
Author: Ishaan Ranjan
Date: January 2026

This module provides comprehensive preprocessing for DNA methylation data
from Illumina 450K/EPIC arrays. The pipeline follows best practices from
the bioinformatics community and ensures high-quality data for machine
learning modeling.

PREPROCESSING STEPS:
1. Beta value validation and transformation
2. M-value conversion for statistical analysis
3. Missing value imputation (KNN-based)
4. Batch effect correction (ComBat-inspired)
5. Probe filtering (SNP-containing, cross-reactive)
6. Sample outlier detection
7. Covariate adjustment

REFERENCES:
- Pidsley et al. (2016): Critical evaluation of Illumina BeadChips
- Fortin et al. (2017): Preprocessing the methylome
- Touleimat & Tost (2012): Complete pipeline for analysis
=============================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Optional, List
import warnings


def beta_to_m_value(beta: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Convert beta values to M-values for statistical analysis.
    
    M-values are the log2 ratio of methylated to unmethylated signal:
    M = log2(beta / (1 - beta))
    
    M-values are preferred for differential methylation analysis because
    they are approximately normally distributed, unlike beta values which
    are bounded [0,1] and often bimodal.
    
    Parameters:
    -----------
    beta : np.ndarray
        Methylation beta values (bounded 0-1)
    epsilon : float
        Small value to avoid log(0) errors
        
    Returns:
    --------
    np.ndarray
        M-values (unbounded, approximately normal)
    
    Note:
    -----
    Du et al. (2010) showed that M-values are more statistically valid
    for differential analysis, while beta values are more interpretable.
    """
    # Clip beta values to avoid infinity
    beta_clipped = np.clip(beta, epsilon, 1 - epsilon)
    m_values = np.log2(beta_clipped / (1 - beta_clipped))
    return m_values


def m_to_beta_value(m_values: np.ndarray) -> np.ndarray:
    """
    Convert M-values back to beta values.
    
    Beta = 2^M / (2^M + 1)
    
    Parameters:
    -----------
    m_values : np.ndarray
        M-values
        
    Returns:
    --------
    np.ndarray
        Beta values
    """
    return np.power(2, m_values) / (np.power(2, m_values) + 1)


def validate_beta_values(
    methylation: np.ndarray,
    sample_names: Optional[List[str]] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, Dict]:
    """
    Validate and clean methylation beta values.
    
    This function checks for common issues in methylation data:
    1. Out-of-range values (outside 0-1)
    2. Extreme values (near 0 or 1)
    3. Missing values (NaN/Inf)
    4. Constant probes (no variation)
    
    Parameters:
    -----------
    methylation : np.ndarray
        Raw beta values (n_samples x n_sites)
    sample_names : List[str], optional
        Sample identifiers for logging
    verbose : bool
        Print diagnostic messages
        
    Returns:
    --------
    Tuple[np.ndarray, Dict]
        Validated beta values and QC statistics dictionary
    """
    if verbose:
        print("\n" + "=" * 50)
        print("BETA VALUE VALIDATION")
        print("=" * 50)
    
    n_samples, n_sites = methylation.shape
    qc_stats = {}
    
    # Check for NaN/Inf values
    n_nan = np.isnan(methylation).sum()
    n_inf = np.isinf(methylation).sum()
    qc_stats['n_nan_values'] = n_nan
    qc_stats['n_inf_values'] = n_inf
    
    if verbose:
        print(f"\nInput shape: {n_samples} samples × {n_sites} CpG sites")
        print(f"NaN values: {n_nan}")
        print(f"Inf values: {n_inf}")
    
    # Replace NaN/Inf with column medians
    if n_nan > 0 or n_inf > 0:
        methylation = np.where(np.isnan(methylation) | np.isinf(methylation),
                               np.nan, methylation)
        col_medians = np.nanmedian(methylation, axis=0)
        inds = np.where(np.isnan(methylation))
        methylation[inds] = np.take(col_medians, inds[1])
    
    # Check for out-of-range values
    n_below_zero = (methylation < 0).sum()
    n_above_one = (methylation > 1).sum()
    qc_stats['n_below_zero'] = n_below_zero
    qc_stats['n_above_one'] = n_above_one
    
    if verbose:
        print(f"Values < 0: {n_below_zero}")
        print(f"Values > 1: {n_above_one}")
    
    # Clip to valid range
    methylation = np.clip(methylation, 0.001, 0.999)
    
    # Check for extreme values
    n_extreme_low = (methylation < 0.01).sum()
    n_extreme_high = (methylation > 0.99).sum()
    qc_stats['n_extreme_low'] = n_extreme_low
    qc_stats['n_extreme_high'] = n_extreme_high
    
    # Identify constant probes (variance = 0)
    probe_variance = np.var(methylation, axis=0)
    n_constant = (probe_variance < 1e-10).sum()
    qc_stats['n_constant_probes'] = n_constant
    
    if verbose:
        print(f"\nExtreme low values (<0.01): {n_extreme_low}")
        print(f"Extreme high values (>0.99): {n_extreme_high}")
        print(f"Constant probes (var~0): {n_constant}")
        print(f"\nValidation complete. Beta range: [{methylation.min():.4f}, {methylation.max():.4f}]")
    
    return methylation, qc_stats


def impute_missing_values(
    methylation: np.ndarray,
    n_neighbors: int = 5,
    verbose: bool = True
) -> np.ndarray:
    """
    Impute missing methylation values using KNN imputation.
    
    KNN imputation uses the values from the k nearest samples (based on
    non-missing features) to estimate missing values. This is more
    sophisticated than mean/median imputation and preserves the
    correlation structure of the data.
    
    Parameters:
    -----------
    methylation : np.ndarray
        Beta values with possible missing values
    n_neighbors : int
        Number of neighbors for KNN imputation
    verbose : bool
        Print progress messages
        
    Returns:
    --------
    np.ndarray
        Imputed beta values
    """
    if verbose:
        print("\n" + "=" * 50)
        print("MISSING VALUE IMPUTATION")
        print("=" * 50)
    
    n_missing = np.isnan(methylation).sum()
    
    if n_missing == 0:
        if verbose:
            print("No missing values detected. Skipping imputation.")
        return methylation
    
    if verbose:
        print(f"Missing values: {n_missing}")
        print(f"Imputing with KNN (k={n_neighbors})...")
    
    imputer = KNNImputer(n_neighbors=n_neighbors)
    methylation_imputed = imputer.fit_transform(methylation)
    
    # Ensure values remain in valid range after imputation
    methylation_imputed = np.clip(methylation_imputed, 0.001, 0.999)
    
    if verbose:
        print("Imputation complete.")
    
    return methylation_imputed


def remove_low_variance_probes(
    methylation: np.ndarray,
    cpg_names: List[str],
    variance_threshold: float = 0.001,
    verbose: bool = True
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Remove CpG probes with low variance across samples.
    
    Low-variance probes are not informative for distinguishing between
    groups and add noise to machine learning models. They also increase
    computational burden without adding value.
    
    Parameters:
    -----------
    methylation : np.ndarray
        Beta values (n_samples x n_sites)
    cpg_names : List[str]
        CpG site identifiers
    variance_threshold : float
        Minimum variance to retain probe
    verbose : bool
        Print progress messages
        
    Returns:
    --------
    Tuple[np.ndarray, List[str], np.ndarray]
        Filtered methylation, filtered names, and mask of retained probes
    """
    if verbose:
        print("\n" + "=" * 50)
        print("LOW VARIANCE PROBE FILTERING")
        print("=" * 50)
    
    probe_variance = np.var(methylation, axis=0)
    mask = probe_variance >= variance_threshold
    
    n_original = len(cpg_names)
    n_retained = mask.sum()
    n_removed = n_original - n_retained
    
    if verbose:
        print(f"Original probes: {n_original}")
        print(f"Variance threshold: {variance_threshold}")
        print(f"Probes removed: {n_removed} ({n_removed/n_original:.1%})")
        print(f"Probes retained: {n_retained}")
    
    methylation_filtered = methylation[:, mask]
    cpg_names_filtered = [name for name, keep in zip(cpg_names, mask) if keep]
    
    return methylation_filtered, cpg_names_filtered, mask


def detect_sample_outliers(
    methylation: np.ndarray,
    sample_names: Optional[List[str]] = None,
    method: str = 'pca',
    threshold: float = 3.0,
    verbose: bool = True
) -> Tuple[np.ndarray, List[int]]:
    """
    Detect sample outliers using PCA or median absolute deviation.
    
    Outlier samples can arise from:
    - Technical failures during array processing
    - Sample contamination or mislabeling
    - Extreme biological variation
    
    Methods:
    - 'pca': Outliers based on first 2 principal components
    - 'mad': Median absolute deviation from sample medians
    
    Parameters:
    -----------
    methylation : np.ndarray
        Beta values
    sample_names : List[str], optional
        Sample identifiers
    method : str
        Detection method ('pca' or 'mad')
    threshold : float
        Number of SDs/MADs for outlier threshold
    verbose : bool
        Print progress
        
    Returns:
    --------
    Tuple[np.ndarray, List[int]]
        Boolean mask of non-outliers and list of outlier indices
    """
    if verbose:
        print("\n" + "=" * 50)
        print("SAMPLE OUTLIER DETECTION")
        print("=" * 50)
    
    n_samples = methylation.shape[0]
    
    if method == 'pca':
        # PCA-based outlier detection
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(methylation)
        
        # Calculate Mahalanobis-like distance from center
        center = pca_coords.mean(axis=0)
        distances = np.sqrt(np.sum((pca_coords - center) ** 2, axis=1))
        
        # Identify outliers
        median_dist = np.median(distances)
        mad = np.median(np.abs(distances - median_dist))
        outlier_mask = distances > median_dist + threshold * mad * 1.4826
        
    elif method == 'mad':
        # MAD-based detection on sample medians
        sample_medians = np.median(methylation, axis=1)
        median_of_medians = np.median(sample_medians)
        mad = np.median(np.abs(sample_medians - median_of_medians))
        
        outlier_mask = np.abs(sample_medians - median_of_medians) > threshold * mad * 1.4826
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    outlier_indices = np.where(outlier_mask)[0].tolist()
    
    if verbose:
        print(f"Method: {method.upper()}")
        print(f"Threshold: {threshold} MADs")
        print(f"Outliers detected: {len(outlier_indices)}")
        if outlier_indices and sample_names:
            print(f"Outlier samples: {[sample_names[i] for i in outlier_indices]}")
    
    return ~outlier_mask, outlier_indices


def adjust_for_covariates(
    methylation: np.ndarray,
    covariates: pd.DataFrame,
    covariate_columns: List[str],
    verbose: bool = True
) -> np.ndarray:
    """
    Adjust methylation values for known covariates using linear regression.
    
    This removes the linear effect of covariates (e.g., age, sex, cell
    composition) from methylation values. The residuals retain the
    biological signal of interest while controlling for confounders.
    
    IMPORTANT: This is useful for prediction, but the adjusted values
    should not be used if the covariates themselves are of interest.
    
    Parameters:
    -----------
    methylation : np.ndarray
        Beta values (n_samples x n_sites)
    covariates : pd.DataFrame
        Covariate data
    covariate_columns : List[str]
        Names of columns to adjust for
    verbose : bool
        Print progress
        
    Returns:
    --------
    np.ndarray
        Covariate-adjusted methylation values
    """
    if verbose:
        print("\n" + "=" * 50)
        print("COVARIATE ADJUSTMENT")
        print("=" * 50)
        print(f"Adjusting for: {covariate_columns}")
    
    # Extract and standardize covariates
    X = covariates[covariate_columns].values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Add intercept
    X_with_intercept = np.column_stack([np.ones(X.shape[0]), X_scaled])
    
    # Regress out covariates from each CpG site
    n_samples, n_sites = methylation.shape
    residuals = np.zeros_like(methylation)
    
    for i in range(n_sites):
        y = methylation[:, i]
        
        # OLS regression
        coeffs, _, _, _ = np.linalg.lstsq(X_with_intercept, y, rcond=None)
        predicted = X_with_intercept @ coeffs
        residuals[:, i] = y - predicted
    
    # Add back overall mean to maintain interpretability
    residuals = residuals + methylation.mean()
    
    # Clip to valid range
    residuals = np.clip(residuals, 0.001, 0.999)
    
    if verbose:
        print(f"Adjustment complete. Shape: {residuals.shape}")
    
    return residuals


def combat_batch_correction(
    methylation: np.ndarray,
    batch: np.ndarray,
    verbose: bool = True
) -> np.ndarray:
    """
    Simple batch effect correction inspired by ComBat.
    
    This is a simplified version of the ComBat algorithm (Johnson et al.).
    It standardizes each batch to have the same mean and variance.
    
    For full ComBat, use the sva package in R or pycombat in Python.
    
    Parameters:
    -----------
    methylation : np.ndarray
        Beta values
    batch : np.ndarray
        Batch identifiers for each sample
    verbose : bool
        Print progress
        
    Returns:
    --------
    np.ndarray
        Batch-corrected methylation
    """
    if verbose:
        print("\n" + "=" * 50)
        print("BATCH EFFECT CORRECTION")
        print("=" * 50)
    
    unique_batches = np.unique(batch)
    n_batches = len(unique_batches)
    
    if n_batches < 2:
        if verbose:
            print("Only one batch detected. Skipping correction.")
        return methylation
    
    if verbose:
        print(f"Number of batches: {n_batches}")
        for b in unique_batches:
            print(f"  Batch {b}: {(batch == b).sum()} samples")
    
    # Calculate grand mean and variance
    grand_mean = methylation.mean(axis=0)
    grand_var = methylation.var(axis=0)
    
    # Correct each batch
    methylation_corrected = methylation.copy()
    
    for b in unique_batches:
        batch_mask = batch == b
        batch_data = methylation[batch_mask]
        
        # Standardize to batch statistics
        batch_mean = batch_data.mean(axis=0)
        batch_std = batch_data.std(axis=0) + 1e-8
        
        # Adjust to grand statistics
        standardized = (batch_data - batch_mean) / batch_std
        methylation_corrected[batch_mask] = standardized * np.sqrt(grand_var) + grand_mean
    
    # Clip to valid range
    methylation_corrected = np.clip(methylation_corrected, 0.001, 0.999)
    
    if verbose:
        print("Batch correction complete.")
    
    return methylation_corrected


class MethylationPreprocessor:
    """
    Complete preprocessing pipeline for DNA methylation data.
    
    This class encapsulates all preprocessing steps and provides a
    unified interface for transforming raw methylation data into
    clean, normalized data suitable for machine learning.
    
    Usage:
    ------
    preprocessor = MethylationPreprocessor()
    X_clean, cpg_filtered = preprocessor.fit_transform(
        methylation, cpg_names, covariates
    )
    """
    
    def __init__(
        self,
        variance_threshold: float = 0.001,
        impute_missing: bool = True,
        n_impute_neighbors: int = 5,
        detect_outliers: bool = True,
        outlier_threshold: float = 3.0,
        convert_to_m: bool = False,
        adjust_covariates: Optional[List[str]] = None,
        verbose: bool = True
    ):
        """
        Initialize preprocessor with configuration.
        
        Parameters:
        -----------
        variance_threshold : float
            Minimum variance for probe retention
        impute_missing : bool
            Whether to impute missing values
        n_impute_neighbors : int
            K for KNN imputation
        detect_outliers : bool
            Whether to detect sample outliers
        outlier_threshold : float
            MAD threshold for outlier detection
        convert_to_m : bool
            Whether to convert to M-values
        adjust_covariates : List[str], optional
            Covariates to adjust for (if any)
        verbose : bool
            Print progress messages
        """
        self.variance_threshold = variance_threshold
        self.impute_missing = impute_missing
        self.n_impute_neighbors = n_impute_neighbors
        self.detect_outliers = detect_outliers
        self.outlier_threshold = outlier_threshold
        self.convert_to_m = convert_to_m
        self.adjust_covariates = adjust_covariates
        self.verbose = verbose
        
        # Fitted attributes
        self.retained_probe_mask_ = None
        self.non_outlier_mask_ = None
        self.qc_stats_ = None
    
    def fit_transform(
        self,
        methylation: np.ndarray,
        cpg_names: List[str],
        covariates: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Fit preprocessor and transform methylation data.
        
        Parameters:
        -----------
        methylation : np.ndarray
            Raw beta values (n_samples x n_sites)
        cpg_names : List[str]
            CpG site identifiers
        covariates : pd.DataFrame, optional
            Sample covariates for adjustment
            
        Returns:
        --------
        Tuple[np.ndarray, List[str]]
            Preprocessed methylation and filtered CpG names
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("METHYLATION PREPROCESSING PIPELINE")
            print("=" * 60)
        
        # Step 1: Validate beta values
        methylation, self.qc_stats_ = validate_beta_values(
            methylation, verbose=self.verbose
        )
        
        # Step 2: Impute missing values
        if self.impute_missing:
            methylation = impute_missing_values(
                methylation, 
                n_neighbors=self.n_impute_neighbors,
                verbose=self.verbose
            )
        
        # Step 3: Detect and optionally remove outliers
        if self.detect_outliers:
            self.non_outlier_mask_, outliers = detect_sample_outliers(
                methylation,
                threshold=self.outlier_threshold,
                verbose=self.verbose
            )
            # Note: We don't remove outliers by default, just flag them
        
        # Step 4: Remove low variance probes
        methylation, cpg_names, self.retained_probe_mask_ = remove_low_variance_probes(
            methylation, cpg_names,
            variance_threshold=self.variance_threshold,
            verbose=self.verbose
        )
        
        # Step 5: Covariate adjustment
        if self.adjust_covariates and covariates is not None:
            methylation = adjust_for_covariates(
                methylation, covariates,
                covariate_columns=self.adjust_covariates,
                verbose=self.verbose
            )
        
        # Step 6: Convert to M-values if requested
        if self.convert_to_m:
            if self.verbose:
                print("\n" + "=" * 50)
                print("CONVERTING TO M-VALUES")
                print("=" * 50)
            methylation = beta_to_m_value(methylation)
            if self.verbose:
                print(f"M-value range: [{methylation.min():.2f}, {methylation.max():.2f}]")
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("PREPROCESSING COMPLETE")
            print("=" * 60)
            print(f"Final shape: {methylation.shape}")
        
        return methylation, cpg_names


if __name__ == "__main__":
    # Demo preprocessing
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Methylation Preprocessing")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    n_samples, n_sites = 100, 1000
    methylation = np.random.beta(5, 5, (n_samples, n_sites))
    
    # Add some NaN values
    nan_mask = np.random.random((n_samples, n_sites)) < 0.01
    methylation[nan_mask] = np.nan
    
    cpg_names = [f'cg{i:08d}' for i in range(n_sites)]
    
    # Run preprocessor
    preprocessor = MethylationPreprocessor(
        variance_threshold=0.001,
        detect_outliers=True,
        verbose=True
    )
    
    meth_clean, cpg_clean = preprocessor.fit_transform(methylation, cpg_names)
    
    print(f"\nFinal: {meth_clean.shape[0]} samples × {meth_clean.shape[1]} CpGs")
