"""
=============================================================================
EPIGENETIC CLOCK CALCULATORS
=============================================================================
AlcoholMethylationML Project
Author: Ishaan Ranjan
Date: January 2026

This module implements simplified versions of the major epigenetic clocks
used to estimate biological age from DNA methylation data. These clocks
are central to understanding how alcohol affects the aging process.

IMPLEMENTED CLOCKS:

1. HORVATH CLOCK (2013)
   - Multi-tissue clock using 353 CpG sites
   - Trained on multiple tissue types
   - Estimates chronological age with MAE ~3.6 years
   - Less sensitive to lifestyle factors

2. PHENOAGE (Levine et al., 2018)
   - Uses 513 CpG sites
   - Incorporates clinical biomarkers in training
   - Better predictor of healthspan and mortality
   - More sensitive to alcohol, smoking, obesity

3. GRIMAGE (Lu et al., 2019)
   - Uses 1030 CpG sites
   - Includes surrogates for plasma proteins and smoking
   - Best predictor of mortality and morbidity
   - Most sensitive to alcohol use

AGE ACCELERATION:
Age acceleration = Epigenetic age - Chronological age
Positive values indicate accelerated aging (older than expected)
Negative values indicate decelerated aging (younger than expected)

ALCOHOL AND AGING:
- Alcohol Use Disorder associated with 2-5 years acceleration
- PhenoAge and GrimAge show strongest effects
- Age acceleration may partially mediate alcohol's health effects

REFERENCES:
- Horvath (2013) Genome Biology
- Levine et al. (2018) Aging
- Lu et al. (2019) Aging
- Rosen et al. (2018) Translational Psychiatry
=============================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from sklearn.linear_model import Ridge


@dataclass  
class ClockCoefficients:
    """
    Storage for epigenetic clock model coefficients.
    
    In real applications, these would be the published coefficients
    from the original clock papers. Here we simulate them to demonstrate
    the methodology.
    """
    cpg_names: List[str]
    coefficients: np.ndarray
    intercept: float
    clock_name: str


def simulate_clock_coefficients(
    n_cpgs: int,
    clock_name: str,
    prefix: str = 'cg',
    random_state: int = 42
) -> ClockCoefficients:
    """
    Simulate realistic clock coefficients for demonstration.
    
    Real clock coefficients are proprietary or published. This function
    creates plausible coefficients that mimic the structure of real clocks.
    
    Clock characteristics:
    - Horvath: Mix of positive/negative, many near-zero
    - PhenoAge: More health/mortality-focused sites
    - GrimAge: Includes large effects (smoking surrogates)
    
    Parameters:
    -----------
    n_cpgs : int
        Number of CpG sites in clock
    clock_name : str
        Name of clock (horvath, phenoage, grimage)
    prefix : str
        CpG name prefix
    random_state : int
        Random seed
        
    Returns:
    --------
    ClockCoefficients
        Object containing clock parameters
    """
    np.random.seed(random_state)
    
    # Generate CpG names
    cpg_names = [f'{prefix}_{clock_name}_{i:05d}' for i in range(n_cpgs)]
    
    # Generate coefficients based on clock type
    if clock_name == 'horvath':
        # Horvath: Sparse, symmetric around zero
        coefficients = np.random.laplace(0, 0.5, n_cpgs)
        # Many sites have small effects
        coefficients[np.abs(coefficients) < 0.3] *= 0.1
        intercept = 35.0  # Approximate baseline age
        
    elif clock_name == 'phenoage':
        # PhenoAge: Mortality/health focused
        coefficients = np.random.normal(0, 1, n_cpgs)
        # Some sites have larger effects
        n_strong = int(n_cpgs * 0.1)
        strong_idx = np.random.choice(n_cpgs, n_strong, replace=False)
        coefficients[strong_idx] *= 3
        intercept = 40.0
        
    elif clock_name == 'grimage':
        # GrimAge: Includes smoking and protein surrogates
        coefficients = np.random.normal(0, 0.8, n_cpgs)
        # DNAm PAI-1, GDF15, etc. surrogates have large effects
        n_surrogates = int(n_cpgs * 0.05)
        surrogate_idx = np.random.choice(n_cpgs, n_surrogates, replace=False)
        coefficients[surrogate_idx] = np.random.uniform(2, 5, n_surrogates)
        intercept = 38.0
        
    else:
        raise ValueError(f"Unknown clock: {clock_name}")
    
    return ClockCoefficients(
        cpg_names=cpg_names,
        coefficients=coefficients,
        intercept=intercept,
        clock_name=clock_name
    )


class EpigeneticClockCalculator:
    """
    Calculator for epigenetic age using multiple clocks.
    
    This class encapsulates the calculation of:
    1. Epigenetic age estimates (Horvath, PhenoAge, GrimAge)
    2. Age acceleration (epigenetic age - chronological age)
    3. Residual age acceleration (adjusted for covariates)
    
    Usage:
    ------
    calculator = EpigeneticClockCalculator()
    results = calculator.calculate_all_clocks(
        clock_sites, chronological_age
    )
    """
    
    def __init__(
        self,
        use_residual_acceleration: bool = True,
        verbose: bool = True
    ):
        """
        Initialize clock calculator.
        
        Parameters:
        -----------
        use_residual_acceleration : bool
            If True, calculate residual acceleration (regressed on chrono age)
        verbose : bool
            Print progress messages
        """
        self.use_residual_acceleration = use_residual_acceleration
        self.verbose = verbose
        
        # Initialize clock coefficients (simulated)
        self.clocks = {
            'horvath': simulate_clock_coefficients(353, 'horvath', 'hv_cg', 42),
            'phenoage': simulate_clock_coefficients(513, 'phenoage', 'pa_cg', 43),
            'grimage': simulate_clock_coefficients(1030, 'grimage', 'gr_cg', 44),
        }
    
    def calculate_epigenetic_age(
        self,
        methylation: np.ndarray,
        clock_name: str
    ) -> np.ndarray:
        """
        Calculate epigenetic age using a specific clock.
        
        The epigenetic age is a weighted sum of methylation values:
        EpiAge = intercept + sum(coef_i * methylation_i)
        
        Parameters:
        -----------
        methylation : np.ndarray
            Methylation values at clock CpG sites (n_samples x n_clock_cpgs)
        clock_name : str
            Name of clock to use
            
        Returns:
        --------
        np.ndarray
            Epigenetic age estimates
        """
        clock = self.clocks[clock_name]
        
        # Ensure correct number of CpGs
        if methylation.shape[1] != len(clock.coefficients):
            raise ValueError(
                f"Expected {len(clock.coefficients)} CpGs for {clock_name}, "
                f"got {methylation.shape[1]}"
            )
        
        # Calculate epigenetic age
        epi_age = clock.intercept + methylation @ clock.coefficients
        
        # Apply anti-log transformation for Horvath clock
        # (Real Horvath clock uses an age transformation)
        if clock_name == 'horvath':
            # Simplified transformation
            epi_age = np.where(
                epi_age > 0,
                epi_age + 1,
                np.exp(epi_age) - 1
            )
        
        return epi_age
    
    def calculate_age_acceleration(
        self,
        epigenetic_age: np.ndarray,
        chronological_age: np.ndarray,
        method: str = 'residual'
    ) -> np.ndarray:
        """
        Calculate age acceleration.
        
        Methods:
        - 'simple': AA = EpiAge - ChronoAge
        - 'residual': Residual from regressing EpiAge on ChronoAge
        
        Residual acceleration is preferred because it's independent of
        chronological age (removes correlation).
        
        Parameters:
        -----------
        epigenetic_age : np.ndarray
            Epigenetic age estimates
        chronological_age : np.ndarray
            Actual chronological ages
        method : str
            'simple' or 'residual'
            
        Returns:
        --------
        np.ndarray
            Age acceleration values
        """
        if method == 'simple':
            return epigenetic_age - chronological_age
        
        elif method == 'residual':
            # Regress epigenetic age on chronological age
            # Residuals = age acceleration
            X = chronological_age.reshape(-1, 1)
            y = epigenetic_age
            
            model = Ridge(alpha=1.0)
            model.fit(X, y)
            predicted = model.predict(X)
            
            residual = y - predicted
            return residual
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def calculate_all_clocks(
        self,
        clock_sites: Dict[str, np.ndarray],
        chronological_age: np.ndarray,
        covariates: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Calculate ages and accelerations for all implemented clocks.
        
        This is the main method for obtaining epigenetic age estimates.
        It calculates:
        - Epigenetic ages (Horvath, PhenoAge, GrimAge)
        - Age accelerations (simple and residual)
        
        Parameters:
        -----------
        clock_sites : Dict[str, np.ndarray]
            Dictionary with methylation at clock CpGs
            Keys: 'horvath', 'phenoage', 'grimage'
        chronological_age : np.ndarray
            Actual ages of samples
        covariates : pd.DataFrame, optional
            Covariates for adjusted acceleration (not yet implemented)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with all age estimates and accelerations
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("EPIGENETIC AGE CALCULATION")
            print("=" * 60)
        
        n_samples = len(chronological_age)
        results = {'chronological_age': chronological_age}
        
        for clock_name in ['horvath', 'phenoage', 'grimage']:
            if self.verbose:
                print(f"\nCalculating {clock_name.upper()} clock...")
            
            # Get methylation data for this clock
            meth = clock_sites[clock_name]
            
            # Calculate epigenetic age
            epi_age = self.calculate_epigenetic_age(meth, clock_name)
            results[f'{clock_name}_age'] = epi_age
            
            # Calculate accelerations
            simple_aa = self.calculate_age_acceleration(
                epi_age, chronological_age, method='simple'
            )
            results[f'{clock_name}_aa_simple'] = simple_aa
            
            if self.use_residual_acceleration:
                residual_aa = self.calculate_age_acceleration(
                    epi_age, chronological_age, method='residual'
                )
                results[f'{clock_name}_aa_residual'] = residual_aa
            
            if self.verbose:
                print(f"  Mean epigenetic age: {epi_age.mean():.1f} ± {epi_age.std():.1f}")
                print(f"  Mean acceleration: {simple_aa.mean():.2f} ± {simple_aa.std():.2f}")
        
        results_df = pd.DataFrame(results)
        
        if self.verbose:
            print("\n" + "-" * 40)
            print("Clock calculation complete.")
        
        return results_df


def compare_age_acceleration_by_group(
    age_results: pd.DataFrame,
    group_labels: np.ndarray,
    clock_names: List[str] = ['horvath', 'phenoage', 'grimage'],
    use_residual: bool = True
) -> Dict:
    """
    Compare age acceleration between groups (e.g., alcohol vs control).
    
    This function performs statistical tests to determine if there are
    significant differences in age acceleration between groups.
    
    Parameters:
    -----------
    age_results : pd.DataFrame
        Results from calculate_all_clocks
    group_labels : np.ndarray
        Binary group labels (0 = control, 1 = case)
    clock_names : List[str]
        Which clocks to compare
    use_residual : bool
        Use residual acceleration (recommended)
        
    Returns:
    --------
    Dict
        Statistical comparison results for each clock
    """
    from scipy import stats
    
    results = {}
    
    for clock in clock_names:
        col = f'{clock}_aa_residual' if use_residual else f'{clock}_aa_simple'
        
        if col not in age_results.columns:
            continue
        
        aa = age_results[col].values
        
        # Split by group
        aa_control = aa[group_labels == 0]
        aa_case = aa[group_labels == 1]
        
        # Statistical tests
        t_stat, t_pval = stats.ttest_ind(aa_case, aa_control)
        u_stat, u_pval = stats.mannwhitneyu(aa_case, aa_control, alternative='two-sided')
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(aa_control) - 1) * aa_control.std()**2 + 
             (len(aa_case) - 1) * aa_case.std()**2) / 
            (len(aa_control) + len(aa_case) - 2)
        )
        cohens_d = (aa_case.mean() - aa_control.mean()) / pooled_std
        
        results[clock] = {
            'mean_control': aa_control.mean(),
            'mean_case': aa_case.mean(),
            'mean_difference': aa_case.mean() - aa_control.mean(),
            'std_control': aa_control.std(),
            'std_case': aa_case.std(),
            't_statistic': t_stat,
            't_pvalue': t_pval,
            'mann_whitney_u': u_stat,
            'mann_whitney_p': u_pval,
            'cohens_d': cohens_d
        }
    
    return results


def print_age_acceleration_comparison(results: Dict) -> None:
    """
    Print formatted comparison of age acceleration by group.
    
    Parameters:
    -----------
    results : Dict
        Results from compare_age_acceleration_by_group
    """
    print("\n" + "=" * 70)
    print("AGE ACCELERATION COMPARISON: ALCOHOL vs CONTROL")
    print("=" * 70)
    
    for clock, stats in results.items():
        print(f"\n{clock.upper()} CLOCK:")
        print("-" * 40)
        print(f"  Control group: {stats['mean_control']:+.2f} ± {stats['std_control']:.2f} years")
        print(f"  Alcohol group: {stats['mean_case']:+.2f} ± {stats['std_case']:.2f} years")
        print(f"  Difference: {stats['mean_difference']:+.2f} years")
        print(f"  Effect size (Cohen's d): {stats['cohens_d']:.3f}")
        print(f"  t-test p-value: {stats['t_pvalue']:.2e}")
        print(f"  Mann-Whitney p-value: {stats['mann_whitney_p']:.2e}")
        
        # Significance indicator
        if stats['t_pvalue'] < 0.001:
            sig = "***"
        elif stats['t_pvalue'] < 0.01:
            sig = "**"
        elif stats['t_pvalue'] < 0.05:
            sig = "*"
        else:
            sig = "ns"
        print(f"  Significance: {sig}")


if __name__ == "__main__":
    # Demo: Calculate epigenetic ages
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Epigenetic Clock Calculation")
    print("=" * 60)
    
    # Simulate data
    np.random.seed(42)
    n_samples = 200
    
    # Create clock sites (simulated methylation)
    clock_sites = {
        'horvath': np.random.beta(5, 5, (n_samples, 353)),
        'phenoage': np.random.beta(5, 5, (n_samples, 513)),
        'grimage': np.random.beta(5, 5, (n_samples, 1030)),
    }
    
    # Chronological age
    chrono_age = np.random.uniform(25, 70, n_samples)
    
    # Add age-related methylation changes
    age_norm = (chrono_age - 25) / 45  # Normalize to 0-1
    for clock_name, sites in clock_sites.items():
        # Some sites increase with age
        n_age_sites = sites.shape[1] // 3
        for i in range(n_age_sites):
            sites[:, i] += age_norm * np.random.uniform(0.1, 0.3)
    
    # Create group labels (simulate alcohol effect)
    alcohol_status = np.random.binomial(1, 0.5, n_samples)
    
    # Alcohol accelerates aging
    for clock_name, sites in clock_sites.items():
        n_effect_sites = 50
        for i in range(n_effect_sites):
            sites[:, i] += alcohol_status * np.random.uniform(0.02, 0.05)
    
    # Clip to valid range
    for clock_name in clock_sites:
        clock_sites[clock_name] = np.clip(clock_sites[clock_name], 0.001, 0.999)
    
    # Calculate ages
    calculator = EpigeneticClockCalculator(verbose=True)
    results = calculator.calculate_all_clocks(clock_sites, chrono_age)
    
    # Compare groups
    comparison = compare_age_acceleration_by_group(results, alcohol_status)
    print_age_acceleration_comparison(comparison)
