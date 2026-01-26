"""
=============================================================================
FEATURE ENGINEERING FOR DNA METHYLATION DATA
=============================================================================
AlcoholMethylationML Project
Author: Ishaan Ranjan
Date: January 2026

This module provides feature engineering utilities for transforming raw
methylation data into informative features for machine learning models.

FEATURE ENGINEERING STRATEGIES:

1. VARIANCE-BASED SELECTION
   - Select CpGs with highest variance across samples
   - High variance sites are most informative for discrimination
   - Reduces dimensionality while preserving signal

2. PRINCIPAL COMPONENT ANALYSIS (PCA)
   - Reduce 10,000+ CpGs to manageable number of PCs
   - Captures major axes of methylation variation
   - Often first few PCs correlate with batch/cell composition

3. PATHWAY-LEVEL AGGREGATION
   - Average methylation within gene pathways
   - Reduces noise through aggregation
   - Creates biologically interpretable features

4. EPIGENETIC CLOCK FEATURES
   - Age acceleration as additional features
   - Captures biological aging signal

5. GENETIC RISK SCORE INTEGRATION
   - Combine PRS with methylation features
   - Gene-environment interaction modeling

REFERENCES:
- Liang et al. (2014): Best practices for feature selection in EWAS
- Heiss & Just (2018): Feature selection for methylation prediction
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (
    VarianceThreshold, 
    SelectKBest, 
    f_classif,
    mutual_info_classif
)
from typing import Dict, Tuple, Optional, List, Union
from dataclasses import dataclass


@dataclass
class FeatureSet:
    """
    Container for engineered features.
    
    Stores the transformed features along with metadata about
    how they were created for reproducibility.
    """
    features: np.ndarray
    feature_names: List[str]
    method: str
    parameters: Dict
    

def select_by_variance(
    methylation: np.ndarray,
    cpg_names: List[str],
    n_features: int = 1000,
    verbose: bool = True
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Select top CpG sites by variance.
    
    Variance-based selection is simple but effective. Sites with higher
    variance across samples are more likely to differ between groups
    and contribute to prediction.
    
    Parameters:
    -----------
    methylation : np.ndarray
        Beta values (n_samples x n_sites)
    cpg_names : List[str]
        CpG site names
    n_features : int
        Number of features to select
    verbose : bool
        Print progress
        
    Returns:
    --------
    Tuple[np.ndarray, List[str], np.ndarray]
        Selected features, names, and variance values
    """
    if verbose:
        print("\n" + "=" * 50)
        print("VARIANCE-BASED FEATURE SELECTION")
        print("=" * 50)
    
    # Calculate variance for each CpG
    variances = np.var(methylation, axis=0)
    
    # Select top N by variance
    n_select = min(n_features, len(cpg_names))
    top_indices = np.argsort(variances)[-n_select:][::-1]
    
    selected_meth = methylation[:, top_indices]
    selected_names = [cpg_names[i] for i in top_indices]
    selected_vars = variances[top_indices]
    
    if verbose:
        print(f"Original features: {len(cpg_names)}")
        print(f"Selected features: {n_select}")
        print(f"Variance range: [{selected_vars.min():.4f}, {selected_vars.max():.4f}]")
    
    return selected_meth, selected_names, selected_vars


def select_by_association(
    methylation: np.ndarray,
    cpg_names: List[str],
    y: np.ndarray,
    n_features: int = 500,
    method: str = 'f_classif',
    verbose: bool = True
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Select CpG sites by association with outcome.
    
    This uses univariate statistical tests to identify CpGs most
    associated with the target variable. Methods:
    - f_classif: ANOVA F-statistic for classification
    - mutual_info: Mutual information (captures nonlinear relationships)
    
    Parameters:
    -----------
    methylation : np.ndarray
        Beta values (n_samples x n_sites)
    cpg_names : List[str]
        CpG site names
    y : np.ndarray
        Target variable (binary for classification)
    n_features : int
        Number of features to select
    method : str
        'f_classif' or 'mutual_info'
    verbose : bool
        Print progress
        
    Returns:
    --------
    Tuple[np.ndarray, List[str], np.ndarray]
        Selected features, names, and scores
    """
    if verbose:
        print("\n" + "=" * 50)
        print(f"ASSOCIATION-BASED SELECTION ({method.upper()})")
        print("=" * 50)
    
    # Choose scoring function
    if method == 'f_classif':
        score_func = f_classif
    elif method == 'mutual_info':
        score_func = mutual_info_classif
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Fit selector
    n_select = min(n_features, methylation.shape[1])
    selector = SelectKBest(score_func=score_func, k=n_select)
    selected_meth = selector.fit_transform(methylation, y)
    
    # Get selected feature info
    mask = selector.get_support()
    selected_names = [name for name, keep in zip(cpg_names, mask) if keep]
    scores = selector.scores_[mask]
    
    if verbose:
        print(f"Original features: {len(cpg_names)}")
        print(f"Selected features: {n_select}")
        print(f"Top 5 scores: {np.sort(scores)[-5:][::-1].round(2)}")
    
    return selected_meth, selected_names, scores


def extract_pca_features(
    methylation: np.ndarray,
    n_components: int = 50,
    standardize: bool = True,
    verbose: bool = True
) -> Tuple[np.ndarray, PCA, float]:
    """
    Extract principal components from methylation data.
    
    PCA is essential for high-dimensional methylation data:
    - Reduces 10,000+ features to manageable number
    - First PCs often capture technical variation (batch, cells)
    - Later PCs may capture biological signal
    
    Parameters:
    -----------
    methylation : np.ndarray
        Beta values (n_samples x n_sites)
    n_components : int
        Number of PCs to extract
    standardize : bool
        Whether to standardize features first
    verbose : bool
        Print progress
        
    Returns:
    --------
    Tuple[np.ndarray, PCA, float]
        PC features, fitted PCA object, and explained variance
    """
    if verbose:
        print("\n" + "=" * 50)
        print("PRINCIPAL COMPONENT ANALYSIS")
        print("=" * 50)
    
    # Standardize if requested
    if standardize:
        scaler = StandardScaler()
        meth_scaled = scaler.fit_transform(methylation)
    else:
        meth_scaled = methylation
    
    # Fit PCA
    n_comp = min(n_components, min(methylation.shape) - 1)
    pca = PCA(n_components=n_comp)
    pc_features = pca.fit_transform(meth_scaled)
    
    explained_var = pca.explained_variance_ratio_.sum()
    
    if verbose:
        print(f"Input shape: {methylation.shape}")
        print(f"Components extracted: {n_comp}")
        print(f"Variance explained: {explained_var:.1%}")
        print(f"PC1 variance: {pca.explained_variance_ratio_[0]:.1%}")
        print(f"PC2 variance: {pca.explained_variance_ratio_[1]:.1%}")
    
    return pc_features, pca, explained_var


def aggregate_by_pathway(
    methylation: np.ndarray,
    annotations: pd.DataFrame,
    pathways: Dict[str, List[str]],
    agg_func: str = 'mean',
    verbose: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """
    Aggregate methylation by gene pathways.
    
    Pathway-level aggregation:
    - Reduces dimensionality
    - Creates interpretable features
    - Reduces noise through averaging
    
    Parameters:
    -----------
    methylation : np.ndarray
        Beta values
    annotations : pd.DataFrame
        CpG annotations with 'cpg_id' and 'gene' columns
    pathways : Dict[str, List[str]]
        Dictionary mapping pathway names to gene lists
    agg_func : str
        Aggregation function ('mean', 'median', 'max')
    verbose : bool
        Print progress
        
    Returns:
    --------
    Tuple[np.ndarray, List[str]]
        Pathway-level features and pathway names
    """
    if verbose:
        print("\n" + "=" * 50)
        print("PATHWAY-LEVEL AGGREGATION")
        print("=" * 50)
    
    pathway_features = []
    pathway_names = []
    
    for pathway_name, genes in pathways.items():
        # Find CpGs in pathway genes
        gene_mask = annotations['gene'].isin(genes)
        cpg_indices = annotations.index[gene_mask].tolist()
        
        if len(cpg_indices) == 0:
            continue
        
        # Subset methylation
        pathway_meth = methylation[:, cpg_indices]
        
        # Aggregate
        if agg_func == 'mean':
            agg_value = np.mean(pathway_meth, axis=1)
        elif agg_func == 'median':
            agg_value = np.median(pathway_meth, axis=1)
        elif agg_func == 'max':
            agg_value = np.max(pathway_meth, axis=1)
        else:
            raise ValueError(f"Unknown aggregation: {agg_func}")
        
        pathway_features.append(agg_value)
        pathway_names.append(pathway_name)
    
    features = np.column_stack(pathway_features) if pathway_features else np.empty((methylation.shape[0], 0))
    
    if verbose:
        print(f"Pathways with CpGs: {len(pathway_names)}")
        print(f"Feature dimension: {features.shape}")
    
    return features, pathway_names


class MethylationFeatureEngineer:
    """
    Comprehensive feature engineering pipeline for methylation data.
    
    This class combines multiple feature engineering strategies to
    create a rich feature set for machine learning models.
    
    Features created:
    1. Top variance CpGs (direct methylation values)
    2. Principal components (global patterns)
    3. Pathway aggregates (biological features)
    4. Epigenetic ages and accelerations
    5. Genetic risk score
    6. Covariates (demographic/clinical)
    
    Usage:
    ------
    engineer = MethylationFeatureEngineer(
        n_top_variance=500,
        n_pca_components=20,
        include_pathways=True
    )
    features = engineer.fit_transform(data)
    """
    
    def __init__(
        self,
        n_top_variance: int = 500,
        n_pca_components: int = 20,
        n_association_features: int = 200,
        include_pathways: bool = True,
        include_clocks: bool = True,
        include_covariates: bool = True,
        covariate_columns: Optional[List[str]] = None,
        verbose: bool = True
    ):
        """
        Initialize feature engineer.
        
        Parameters:
        -----------
        n_top_variance : int
            Number of top variance CpGs to include
        n_pca_components : int
            Number of PCA components
        n_association_features : int
            Number of association-selected features (requires y)
        include_pathways : bool
            Whether to include pathway-level features
        include_clocks : bool
            Whether to include epigenetic age features
        include_covariates : bool
            Whether to include covariate features
        covariate_columns : List[str], optional
            Which covariates to include
        verbose : bool
            Print progress
        """
        self.n_top_variance = n_top_variance
        self.n_pca_components = n_pca_components
        self.n_association_features = n_association_features
        self.include_pathways = include_pathways
        self.include_clocks = include_clocks
        self.include_covariates = include_covariates
        self.covariate_columns = covariate_columns or [
            'age', 'sex', 'smoking_status', 'bmi', 
            'genetic_risk_score'
        ]
        self.verbose = verbose
        
        # Fitted attributes
        self.pca_ = None
        self.scaler_ = None
        self.selected_cpgs_ = None
        self.feature_names_ = None
        
        # Default pathways (can be overridden)
        self.pathways = {
            'alcohol_metabolism': ['ADH1A', 'ADH1B', 'ADH1C', 'ADH4', 'ADH5', 
                                   'ALDH1A1', 'ALDH2', 'CYP2E1'],
            'immune_response': ['IL6', 'IL1B', 'TNF', 'NFKB1', 'TLR4', 'CXCL8'],
            'liver_function': ['CYP1A2', 'CYP2D6', 'CYP3A4', 'UGT1A1'],
            'oxidative_stress': ['SOD1', 'SOD2', 'CAT', 'GPX1', 'NRF2'],
            'dna_repair': ['PARP1', 'XRCC1', 'OGG1', 'MGMT'],
        }
    
    def fit_transform(
        self,
        data: Dict,
        y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Engineer features from methylation data.
        
        Parameters:
        -----------
        data : Dict
            Data dictionary from synthetic_generator containing:
            - 'methylation': np.ndarray
            - 'cpg_names': List[str]
            - 'annotations': pd.DataFrame
            - 'covariates': pd.DataFrame
            - 'clock_sites': Dict (for epigenetic ages)
        y : np.ndarray, optional
            Target variable for association-based selection
            
        Returns:
        --------
        Tuple[np.ndarray, List[str]]
            Feature matrix and feature names
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("COMPREHENSIVE FEATURE ENGINEERING")
            print("=" * 60)
        
        all_features = []
        all_names = []
        
        methylation = data['methylation']
        cpg_names = data['cpg_names']
        
        # 1. Top variance CpGs
        if self.verbose:
            print("\n[1/5] Extracting top variance CpGs...")
        var_features, var_names, _ = select_by_variance(
            methylation, cpg_names, 
            n_features=self.n_top_variance,
            verbose=self.verbose
        )
        all_features.append(var_features)
        all_names.extend([f"var_{name}" for name in var_names])
        
        # 2. PCA components
        if self.n_pca_components > 0:
            if self.verbose:
                print("\n[2/5] Extracting PCA components...")
            pc_features, self.pca_, _ = extract_pca_features(
                methylation,
                n_components=self.n_pca_components,
                verbose=self.verbose
            )
            all_features.append(pc_features)
            all_names.extend([f"PC{i+1}" for i in range(pc_features.shape[1])])
        
        # 3. Association-based features (if y provided)
        if y is not None and self.n_association_features > 0:
            if self.verbose:
                print("\n[3/5] Extracting association-based features...")
            assoc_features, assoc_names, _ = select_by_association(
                methylation, cpg_names, y,
                n_features=self.n_association_features,
                verbose=self.verbose
            )
            all_features.append(assoc_features)
            all_names.extend([f"assoc_{name}" for name in assoc_names])
        
        # 4. Pathway aggregates
        if self.include_pathways:
            if self.verbose:
                print("\n[4/5] Extracting pathway features...")
            
            # Create index mapping for annotations
            annotations = data['annotations'].copy()
            annotations = annotations.reset_index(drop=True)
            
            # Only proceed if we have gene annotations
            if 'gene' in annotations.columns:
                pathway_features, pathway_names = aggregate_by_pathway(
                    methylation, annotations, self.pathways,
                    verbose=self.verbose
                )
                if pathway_features.size > 0:
                    all_features.append(pathway_features)
                    all_names.extend([f"pathway_{name}" for name in pathway_names])
        
        # 5. Epigenetic ages and covariates
        if self.include_clocks or self.include_covariates:
            if self.verbose:
                print("\n[5/5] Adding epigenetic ages and covariates...")
            
            covariates = data['covariates']
            
            if self.include_clocks:
                # Calculate epigenetic ages
                from features.epigenetic_clocks import EpigeneticClockCalculator
                
                calculator = EpigeneticClockCalculator(verbose=False)
                age_results = calculator.calculate_all_clocks(
                    data['clock_sites'],
                    covariates['age'].values
                )
                
                # Add age accelerations
                for clock in ['horvath', 'phenoage', 'grimage']:
                    aa_col = f'{clock}_aa_residual'
                    if aa_col in age_results.columns:
                        all_features.append(age_results[aa_col].values.reshape(-1, 1))
                        all_names.append(f'age_accel_{clock}')
            
            if self.include_covariates:
                # Add covariate features
                available_covs = [c for c in self.covariate_columns 
                                  if c in covariates.columns]
                cov_features = covariates[available_covs].values.astype(float)
                all_features.append(cov_features)
                all_names.extend(available_covs)
        
        # Combine all features
        X = np.hstack(all_features)
        
        # Store feature names
        self.feature_names_ = all_names
        
        # Handle any NaN/Inf values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("FEATURE ENGINEERING COMPLETE")
            print("=" * 60)
            print(f"Total features: {X.shape[1]}")
            print(f"Sample size: {X.shape[0]}")
            print(f"\nFeature breakdown:")
            print(f"  - Top variance CpGs: {self.n_top_variance}")
            print(f"  - PCA components: {self.n_pca_components}")
            if y is not None:
                print(f"  - Association features: {self.n_association_features}")
            if self.include_pathways:
                print(f"  - Pathway features: {len([n for n in all_names if n.startswith('pathway_')])}")
            if self.include_clocks:
                print(f"  - Age acceleration: 3")
            if self.include_covariates:
                print(f"  - Covariates: {len(available_covs)}")
        
        return X, self.feature_names_


def create_train_test_features(
    data: Dict,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Create train/test split with engineered features.
    
    This is a convenience function that:
    1. Splits data into train/test
    2. Engineers features on training data
    3. Applies same transformations to test data
    
    Parameters:
    -----------
    data : Dict
        Data dictionary from synthetic_generator
    test_size : float
        Fraction for test set
    random_state : int
        Random seed
    verbose : bool
        Print progress
        
    Returns:
    --------
    Dict
        Dictionary with X_train, X_test, y_train, y_test, feature_names
    """
    from sklearn.model_selection import train_test_split
    
    if verbose:
        print("\n" + "=" * 60)
        print("CREATING TRAIN/TEST FEATURES")
        print("=" * 60)
    
    # Get target variable
    y = data['covariates']['alcohol_status'].values
    n_samples = len(y)
    
    # Create indices for splitting
    indices = np.arange(n_samples)
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, 
        random_state=random_state, stratify=y
    )
    
    if verbose:
        print(f"Training samples: {len(train_idx)}")
        print(f"Test samples: {len(test_idx)}")
    
    # Split all data components
    def split_data(data_dict, train_idx, test_idx):
        """Split data dictionary into train/test."""
        train_data = {
            'methylation': data_dict['methylation'][train_idx],
            'cpg_names': data_dict['cpg_names'],
            'annotations': data_dict['annotations'],
            'covariates': data_dict['covariates'].iloc[train_idx].reset_index(drop=True),
            'clock_sites': {
                k: v[train_idx] for k, v in data_dict['clock_sites'].items()
                if isinstance(v, np.ndarray)
            },
        }
        test_data = {
            'methylation': data_dict['methylation'][test_idx],
            'cpg_names': data_dict['cpg_names'],
            'annotations': data_dict['annotations'],
            'covariates': data_dict['covariates'].iloc[test_idx].reset_index(drop=True),
            'clock_sites': {
                k: v[test_idx] for k, v in data_dict['clock_sites'].items()
                if isinstance(v, np.ndarray)
            },
        }
        # Add clock CpG names
        for name in ['horvath_cpg_names', 'phenoage_cpg_names', 'grimage_cpg_names']:
            if name in data_dict['clock_sites']:
                train_data['clock_sites'][name] = data_dict['clock_sites'][name]
                test_data['clock_sites'][name] = data_dict['clock_sites'][name]
        
        return train_data, test_data
    
    train_data, test_data = split_data(data, train_idx, test_idx)
    
    # Engineer features
    engineer = MethylationFeatureEngineer(
        n_top_variance=500,
        n_pca_components=20,
        n_association_features=200,
        verbose=verbose
    )
    
    # Fit on training data
    y_train = train_data['covariates']['alcohol_status'].values
    X_train, feature_names = engineer.fit_transform(train_data, y=y_train)
    
    # Transform test data (using same selected features)
    y_test = test_data['covariates']['alcohol_status'].values
    X_test, _ = engineer.fit_transform(test_data, y=y_test)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'train_idx': train_idx,
        'test_idx': test_idx
    }


if __name__ == "__main__":
    # Demo feature engineering
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Feature Engineering Pipeline")
    print("=" * 60)
    
    # Generate sample data
    import sys
    sys.path.insert(0, '..')
    from data.synthetic_generator import generate_methylation_data
    
    data = generate_methylation_data(n_samples=400, random_state=42)
    
    # Create features
    y = data['covariates']['alcohol_status'].values
    
    engineer = MethylationFeatureEngineer(
        n_top_variance=200,
        n_pca_components=10,
        n_association_features=100,
        verbose=True
    )
    
    X, feature_names = engineer.fit_transform(data, y=y)
    
    print(f"\nFinal feature matrix: {X.shape}")
    print(f"Number of features: {len(feature_names)}")
