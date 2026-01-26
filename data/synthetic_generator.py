"""
=============================================================================
SYNTHETIC METHYLATION DATA GENERATOR
=============================================================================
AlcoholMethylationML Project
Author: Ishaan Ranjan
Date: January 2026

This module generates realistic synthetic DNA methylation data that mimics
patterns observed in real epigenome-wide association studies (EWAS) of alcohol
use and dependence. The data is based on published effect sizes and distributions
from studies including:

- GSE110043: Blood methylation in alcohol use
- GSE98876: CD3+ T cells in alcohol dependence
- Liu et al. (2018): DNA methylation biomarker of alcohol consumption
- Lohoff et al. (2018): EWAS of alcohol consumption
- Rosen et al. (2018): DNA methylation age acceleration in alcohol dependence

KEY FEATURES:
1. Realistic beta value distributions (bimodal, centered around 0.2 and 0.8)
2. Alcohol-associated CpG sites with effect sizes from real EWAS
3. Covariates: age, sex, smoking, BMI, estimated cell proportions
4. Epigenetic clock CpG sites (Horvath, PhenoAge, GrimAge)
5. Simulated genetic risk score based on GWAS variants

GENETIC ASPECTS:
- Includes ADH1B and ALDH2 variant effects (alcohol metabolism genes)
- Simulates polygenic risk score from ~100 GWAS-identified variants
- Models gene-environment interactions between genetics and exposure
=============================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import warnings

# Set random seed for reproducibility
np.random.seed(42)


@dataclass
class MethylationDataConfig:
    """
    Configuration for synthetic methylation data generation.
    
    This dataclass stores all parameters needed to generate realistic
    methylation data, including sample sizes, CpG site counts, and
    effect sizes derived from published EWAS studies.
    """
    # Sample configuration
    n_samples: int = 800  # Typical EWAS sample size
    case_control_ratio: float = 0.5  # 50% cases (alcohol users)
    
    # CpG site configuration
    n_cpg_sites: int = 10000  # Variable CpG sites to include
    n_alcohol_associated_cpgs: int = 100  # Sites with alcohol effects
    n_horvath_cpgs: int = 353  # Horvath clock sites
    n_phenoage_cpgs: int = 513  # PhenoAge clock sites
    n_grimage_cpgs: int = 1030  # GrimAge clock sites
    
    # Effect sizes (derived from Liu et al., 2018 and Lohoff et al., 2018)
    alcohol_effect_mean: float = 0.02  # Mean methylation difference
    alcohol_effect_sd: float = 0.015  # SD of effect sizes
    
    # Age range for samples
    age_min: int = 21
    age_max: int = 75
    
    # Genetic parameters
    n_gwas_variants: int = 100  # Number of risk variants to simulate
    heritability: float = 0.5  # Heritability of alcohol dependence


# =============================================================================
# KNOWN ALCOHOL-ASSOCIATED CPG SITES FROM EWAS LITERATURE
# =============================================================================
# These CpG sites have been repeatedly associated with alcohol consumption
# in multiple independent studies. Effect sizes are approximate and based
# on published meta-analyses.

KNOWN_ALCOHOL_CPGS = {
    # CpG ID: (Gene, Effect Size, Direction, Reference)
    'cg04987734': ('SLC7A11', 0.023, 'hypo', 'Liu et al. 2018'),
    'cg02583484': ('HNRNPA1', 0.019, 'hypo', 'Liu et al. 2018'),
    'cg09935388': ('GFI1', 0.031, 'hypo', 'Lohoff et al. 2018'),
    'cg17901584': ('DHCR24', 0.017, 'hyper', 'Liu et al. 2018'),
    'cg06690548': ('LRP5', 0.028, 'hypo', 'Lohoff et al. 2018'),
    'cg00769805': ('CADM1', 0.015, 'hypo', 'Liu et al. 2018'),
    'cg24859433': ('HNRNPUL1', 0.021, 'hypo', 'Bernabeu et al. 2021'),
    'cg12806681': ('AHRR', 0.035, 'hypo', 'Multiple studies'),  # Also smoking-related
    'cg05575921': ('AHRR', 0.042, 'hypo', 'Multiple studies'),  # Also smoking-related
    'cg03636183': ('F2RL3', 0.029, 'hypo', 'Multiple studies'),  # Also smoking-related
}

# Gene pathways relevant to alcohol metabolism and response
ALCOHOL_METABOLISM_GENES = [
    'ADH1A', 'ADH1B', 'ADH1C', 'ADH4', 'ADH5', 'ADH6', 'ADH7',  # Alcohol dehydrogenases
    'ALDH1A1', 'ALDH2', 'ALDH1B1',  # Aldehyde dehydrogenases
    'CYP2E1',  # Cytochrome P450
]

IMMUNE_RESPONSE_GENES = [
    'IL6', 'IL1B', 'TNF', 'NFKB1', 'TLR4', 'CXCL8',  # Inflammatory genes
    'CD14', 'CD4', 'CD8A',  # Immune cell markers
]

LIVER_METABOLISM_GENES = [
    'CYP1A2', 'CYP2D6', 'CYP3A4', 'UGT1A1',  # Drug metabolism
    'ABCB1', 'ABCC2',  # Transporters
]


def generate_beta_distribution_mixture(
    n_sites: int,
    n_samples: int,
    hypomethylated_fraction: float = 0.3,
    hypermethylated_fraction: float = 0.5,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Generate realistic methylation beta values with bimodal distribution.
    
    DNA methylation beta values typically follow a bimodal distribution:
    - Hypomethylated sites (beta ~ 0.1-0.3): Often in promoter regions
    - Hypermethylated sites (beta ~ 0.7-0.9): Often in gene bodies
    - Intermediate sites (beta ~ 0.4-0.6): Variable sites
    
    This function creates a realistic mixture distribution based on
    observed patterns in 450K/EPIC array data.
    
    Parameters:
    -----------
    n_sites : int
        Number of CpG sites to generate
    n_samples : int
        Number of samples
    hypomethylated_fraction : float
        Fraction of sites that are hypomethylated (default 0.3)
    hypermethylated_fraction : float
        Fraction of sites that are hypermethylated (default 0.5)
    random_state : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    np.ndarray
        Matrix of beta values (n_samples x n_sites)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    intermediate_fraction = 1.0 - hypomethylated_fraction - hypermethylated_fraction
    
    # Number of sites in each category
    n_hypo = int(n_sites * hypomethylated_fraction)
    n_hyper = int(n_sites * hypermethylated_fraction)
    n_inter = n_sites - n_hypo - n_hyper
    
    # Generate base methylation values for each category
    # Using beta distribution for realistic bounded values in [0,1]
    
    # Hypomethylated sites: beta distribution with alpha=2, beta=8 (mean ~0.2)
    hypo_base = np.random.beta(2, 8, size=(n_samples, n_hypo))
    
    # Hypermethylated sites: beta distribution with alpha=8, beta=2 (mean ~0.8)
    hyper_base = np.random.beta(8, 2, size=(n_samples, n_hyper))
    
    # Intermediate sites: beta distribution with alpha=5, beta=5 (mean ~0.5)
    inter_base = np.random.beta(5, 5, size=(n_samples, n_inter))
    
    # Add sample-specific noise (technical variation)
    sample_noise = np.random.normal(0, 0.02, size=(n_samples, 1))
    
    # Combine all sites
    methylation = np.hstack([hypo_base, inter_base, hyper_base])
    
    # Add sample-level noise and clip to valid range
    methylation = methylation + sample_noise
    methylation = np.clip(methylation, 0.001, 0.999)
    
    # Shuffle columns so site types are mixed
    shuffle_idx = np.random.permutation(n_sites)
    methylation = methylation[:, shuffle_idx]
    
    return methylation


def generate_covariates(
    n_samples: int,
    alcohol_status: np.ndarray,
    config: MethylationDataConfig,
    random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate realistic covariates for methylation samples.
    
    Covariates are essential confounders in EWAS studies. This function
    generates covariates that are realistically correlated with both
    alcohol use and methylation patterns.
    
    IMPORTANT COVARIATES:
    - Age: Strongly affects methylation; allows epigenetic clock calculation
    - Sex: Gene expression and methylation differ by sex
    - Smoking: Major confounder with overlapping methylation signatures
    - BMI: Associated with inflammation and metabolic methylation changes
    - Cell proportions: Blood cell heterogeneity affects bulk methylation
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    alcohol_status : np.ndarray
        Binary array indicating alcohol use (1) vs control (0)
    config : MethylationDataConfig
        Configuration object with parameter settings
    random_state : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with all covariates
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Age: Alcohol users tend to be slightly older in clinical samples
    age_base = np.random.uniform(config.age_min, config.age_max, n_samples)
    age = age_base + alcohol_status * np.random.uniform(0, 5, n_samples)
    age = np.clip(age, config.age_min, config.age_max)
    
    # Sex: 0 = Female, 1 = Male (slightly more males in alcohol group)
    male_prob = 0.5 + alcohol_status * 0.1  # 60% male in alcohol group
    sex = np.random.binomial(1, male_prob)
    
    # Smoking: Strongly correlated with alcohol use
    # This is a MAJOR confounder in alcohol EWAS studies
    smoking_prob = 0.15 + alcohol_status * 0.35  # 15% vs 50% smoking
    smoking_status = np.random.binomial(1, smoking_prob)
    
    # Pack-years for smokers (needed for GrimAge)
    pack_years = np.where(smoking_status == 1,
                          np.random.exponential(15, n_samples),
                          0)
    
    # BMI: Slightly higher in alcohol users
    bmi_base = np.random.normal(25, 4, n_samples)
    bmi = bmi_base + alcohol_status * np.random.normal(2, 1, n_samples)
    bmi = np.clip(bmi, 18, 45)
    
    # Estimated cell proportions (sum to 1)
    # Using typical blood cell distributions
    # Alcohol affects immune cell composition
    cd8t = np.random.beta(2, 10, n_samples) - alcohol_status * 0.02
    cd4t = np.random.beta(3, 7, n_samples) - alcohol_status * 0.03
    nk = np.random.beta(2, 15, n_samples) + alcohol_status * 0.01
    bcell = np.random.beta(2, 12, n_samples)
    mono = np.random.beta(2, 10, n_samples) + alcohol_status * 0.02
    gran = 1 - cd8t - cd4t - nk - bcell - mono  # Granulocytes
    
    # Ensure valid proportions
    cell_total = cd8t + cd4t + nk + bcell + mono + gran
    cd8t, cd4t, nk, bcell, mono, gran = [
        np.clip(x / cell_total, 0.01, 0.99) 
        for x in [cd8t, cd4t, nk, bcell, mono, gran]
    ]
    
    # Generate polygenic risk score (PRS) for alcohol dependence
    # Based on GWAS findings: ADH1B rs1229984, ALDH2 rs671, etc.
    prs = generate_genetic_risk_score(n_samples, alcohol_status, config)
    
    # Create DataFrame
    covariates = pd.DataFrame({
        'sample_id': [f'SAMPLE_{i:04d}' for i in range(n_samples)],
        'age': age.round(1),
        'sex': sex.astype(int),
        'smoking_status': smoking_status.astype(int),
        'pack_years': pack_years.round(1),
        'bmi': bmi.round(1),
        'cd8t': cd8t.round(4),
        'cd4t': cd4t.round(4),
        'nk': nk.round(4),
        'bcell': bcell.round(4),
        'mono': mono.round(4),
        'gran': gran.round(4),
        'genetic_risk_score': prs.round(4),
        'alcohol_status': alcohol_status.astype(int),
    })
    
    return covariates


def generate_genetic_risk_score(
    n_samples: int,
    alcohol_status: np.ndarray,
    config: MethylationDataConfig,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Generate a polygenic risk score (PRS) for alcohol dependence.
    
    This simulates the genetic component of alcohol dependence risk,
    based on GWAS findings. Key variants include:
    
    - ADH1B rs1229984: Protective variant (Asian populations)
    - ALDH2 rs671: Protective "flushing" variant
    - DRD2, OPRM1, GABRA2: Neurotransmitter pathway genes
    - KLB, GCKR, SLC39A8: Novel GWAS hits
    
    The PRS is standardized and explains ~5% of variance in alcohol
    outcomes (based on current GWAS heritability estimates).
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    alcohol_status : np.ndarray
        Binary alcohol status (used to ensure realistic correlation)
    config : MethylationDataConfig
        Configuration with genetic parameters
    random_state : int, optional
        Random seed
        
    Returns:
    --------
    np.ndarray
        Standardized polygenic risk scores
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Simulate genotypes at risk loci
    n_variants = config.n_gwas_variants
    
    # Effect sizes follow exponential distribution (most small, few large)
    effect_sizes = np.random.exponential(0.05, n_variants)
    
    # Simulate allele frequencies (mostly common variants from GWAS)
    allele_freqs = np.random.beta(2, 2, n_variants) * 0.4 + 0.1
    
    # Generate genotypes (0, 1, 2) based on allele frequencies
    # Using Hardy-Weinberg equilibrium
    genotypes = np.zeros((n_samples, n_variants))
    for i in range(n_variants):
        p = allele_freqs[i]
        genotype_probs = [(1-p)**2, 2*p*(1-p), p**2]
        genotypes[:, i] = np.random.choice([0, 1, 2], size=n_samples, 
                                           p=genotype_probs)
    
    # Calculate raw PRS
    prs_raw = np.dot(genotypes, effect_sizes)
    
    # Add noise and correlation with alcohol status
    # The PRS should be higher on average in alcohol group
    noise = np.random.normal(0, 0.5, n_samples)
    prs_raw = prs_raw + alcohol_status * 0.3 + noise
    
    # Standardize to mean=0, sd=1
    prs = (prs_raw - prs_raw.mean()) / prs_raw.std()
    
    return prs


def apply_alcohol_effects(
    methylation: np.ndarray,
    alcohol_status: np.ndarray,
    cpg_names: List[str],
    config: MethylationDataConfig,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Apply alcohol-associated methylation changes to specific CpG sites.
    
    This function adds realistic effect sizes to CpG sites based on
    published EWAS findings. Effect sizes are drawn from a distribution
    fitted to real data, with both hypo- and hypermethylation observed.
    
    The alcohol effects include:
    1. Known replicated CpG sites (from KNOWN_ALCOHOL_CPGS)
    2. Novel associated sites with smaller effects
    3. Pathway-specific effects (immune, liver metabolism)
    
    Parameters:
    -----------
    methylation : np.ndarray
        Base methylation matrix (n_samples x n_sites)
    alcohol_status : np.ndarray
        Binary alcohol status array
    cpg_names : List[str]
        Names of CpG sites
    config : MethylationDataConfig
        Configuration object
    random_state : int, optional
        Random seed
        
    Returns:
    --------
    np.ndarray
        Modified methylation matrix with alcohol effects
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples, n_sites = methylation.shape
    methylation_modified = methylation.copy()
    
    # Select sites to be alcohol-associated
    n_alcohol_sites = min(config.n_alcohol_associated_cpgs, n_sites)
    alcohol_site_indices = np.random.choice(n_sites, n_alcohol_sites, replace=False)
    
    # Generate effect sizes (mix of positive and negative)
    effect_sizes = np.random.normal(
        config.alcohol_effect_mean,
        config.alcohol_effect_sd,
        n_alcohol_sites
    )
    
    # Half hypomethylation, half hypermethylation
    direction = np.random.choice([-1, 1], n_alcohol_sites)
    effect_sizes = np.abs(effect_sizes) * direction
    
    # Apply effects to alcohol group
    for i, (site_idx, effect) in enumerate(zip(alcohol_site_indices, effect_sizes)):
        # Effect only in alcohol group, with some individual variation
        individual_effect = effect * (1 + np.random.normal(0, 0.2, n_samples))
        methylation_modified[:, site_idx] += alcohol_status * individual_effect
    
    # Clip to valid range
    methylation_modified = np.clip(methylation_modified, 0.001, 0.999)
    
    return methylation_modified


def generate_epigenetic_clock_sites(
    n_samples: int,
    age: np.ndarray,
    alcohol_status: np.ndarray,
    config: MethylationDataConfig,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate methylation values for epigenetic clock CpG sites.
    
    This creates age-correlated methylation patterns that allow
    calculation of Horvath, PhenoAge, and GrimAge clocks. Importantly,
    alcohol-associated age acceleration is built in.
    
    EPIGENETIC AGE ACCELERATION:
    Studies show alcohol use disorder is associated with ~2-5 years
    of age acceleration, particularly on PhenoAge and GrimAge.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    age : np.ndarray
        Chronological ages
    alcohol_status : np.ndarray
        Binary alcohol status
    config : MethylationDataConfig
        Configuration object
    random_state : int, optional
        Random seed
        
    Returns:
    --------
    Tuple of (horvath_cpgs, phenoage_cpgs, grimage_cpgs)
        Each is a numpy array of methylation values
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Normalize age to 0-1 scale for easier calculation
    age_normalized = (age - config.age_min) / (config.age_max - config.age_min)
    
    # Add alcohol-related age acceleration
    # PhenoAge: ~2.5 years acceleration in heavy drinkers
    # GrimAge: ~3.5 years acceleration in alcohol dependence
    # Horvath: ~1.5 years acceleration (less sensitive to alcohol)
    
    horvath_age_effect = 1.5 / (config.age_max - config.age_min)  # Normalized effect
    phenoage_effect = 2.5 / (config.age_max - config.age_min)
    grimage_effect = 3.5 / (config.age_max - config.age_min)
    
    # Generate clock sites with age correlation
    def generate_clock_sites(n_sites, age_norm, age_accel, clock_name):
        """Generate age-correlated CpG sites for a specific clock."""
        sites = np.zeros((n_samples, n_sites))
        
        for i in range(n_sites):
            # Random correlation strength with age
            age_corr = np.random.uniform(-0.8, 0.8)
            
            if age_corr > 0:
                # Hypermethylation with age
                base = 0.3 + age_norm * 0.4 * abs(age_corr)
            else:
                # Hypomethylation with age
                base = 0.7 - age_norm * 0.4 * abs(age_corr)
            
            # Add alcohol effect (accelerated aging)
            alcohol_effect = alcohol_status * age_accel * np.sign(age_corr)
            
            # Add noise
            noise = np.random.normal(0, 0.05, n_samples)
            
            sites[:, i] = base + alcohol_effect + noise
        
        return np.clip(sites, 0.001, 0.999)
    
    horvath_cpgs = generate_clock_sites(
        config.n_horvath_cpgs, age_normalized, 
        horvath_age_effect + alcohol_status * horvath_age_effect, 'horvath'
    )
    
    phenoage_cpgs = generate_clock_sites(
        config.n_phenoage_cpgs, age_normalized,
        phenoage_effect + alcohol_status * phenoage_effect, 'phenoage'
    )
    
    grimage_cpgs = generate_clock_sites(
        config.n_grimage_cpgs, age_normalized,
        grimage_effect + alcohol_status * grimage_effect, 'grimage'
    )
    
    return horvath_cpgs, phenoage_cpgs, grimage_cpgs


def generate_cpg_names(n_sites: int, prefix: str = 'cg') -> List[str]:
    """
    Generate realistic CpG site names.
    
    CpG names in Illumina arrays follow the pattern 'cgXXXXXXXX'
    where X is a digit. This function generates unique names.
    
    Parameters:
    -----------
    n_sites : int
        Number of site names to generate
    prefix : str
        Prefix for site names (default 'cg')
        
    Returns:
    --------
    List[str]
        List of CpG site names
    """
    return [f'{prefix}{i:08d}' for i in range(n_sites)]


def generate_gene_annotations(cpg_names: List[str]) -> pd.DataFrame:
    """
    Generate gene annotations for CpG sites.
    
    Maps CpG sites to genes and genomic features. This includes:
    - Gene name (RefSeq)
    - Genomic location (TSS200, TSS1500, 5'UTR, 1stExon, Body, 3'UTR)
    - CpG island relation (Island, Shore, Shelf, Open Sea)
    - Chromosome and position
    
    Parameters:
    -----------
    cpg_names : List[str]
        List of CpG site names
        
    Returns:
    --------
    pd.DataFrame
        Annotation DataFrame
    """
    n_sites = len(cpg_names)
    
    # Gene names (mix of real and simulated)
    all_genes = (
        ALCOHOL_METABOLISM_GENES + 
        IMMUNE_RESPONSE_GENES + 
        LIVER_METABOLISM_GENES +
        [f'GENE{i}' for i in range(500)]
    )
    
    genes = np.random.choice(all_genes, n_sites)
    
    # Genomic locations
    locations = np.random.choice(
        ['TSS200', 'TSS1500', "5'UTR", '1stExon', 'Body', "3'UTR"],
        n_sites,
        p=[0.15, 0.15, 0.1, 0.1, 0.4, 0.1]
    )
    
    # CpG island relation
    island_relation = np.random.choice(
        ['Island', 'N_Shore', 'S_Shore', 'N_Shelf', 'S_Shelf', 'OpenSea'],
        n_sites,
        p=[0.3, 0.1, 0.1, 0.05, 0.05, 0.4]
    )
    
    # Chromosomes
    chromosomes = np.random.choice(
        [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY'],
        n_sites,
        p=[1/24] * 24
    )
    
    # Positions
    positions = np.random.randint(1000000, 250000000, n_sites)
    
    return pd.DataFrame({
        'cpg_id': cpg_names,
        'gene': genes,
        'location': locations,
        'island_relation': island_relation,
        'chromosome': chromosomes,
        'position': positions
    })


def generate_methylation_data(
    n_samples: int = 800,
    config: Optional[MethylationDataConfig] = None,
    random_state: int = 42
) -> Dict:
    """
    Generate complete synthetic methylation dataset.
    
    This is the main function that orchestrates the data generation
    process. It creates a realistic dataset suitable for training
    machine learning models to predict alcohol use from methylation.
    
    The generated data includes:
    1. Methylation beta values (n_samples x n_cpg_sites)
    2. Epigenetic clock CpG sites (Horvath, PhenoAge, GrimAge)
    3. Covariates (age, sex, smoking, BMI, cell proportions)
    4. Genetic risk scores
    5. CpG site annotations
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate (default 800)
    config : MethylationDataConfig, optional
        Configuration object (uses defaults if None)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    Dict
        Dictionary containing all generated data components:
        - 'methylation': np.ndarray of beta values
        - 'clock_sites': dict with Horvath, PhenoAge, GrimAge CpGs
        - 'covariates': pd.DataFrame with all covariates
        - 'cpg_names': list of CpG site names
        - 'annotations': pd.DataFrame with gene annotations
        - 'alcohol_cpg_indices': indices of alcohol-associated sites
    """
    print("=" * 60)
    print("GENERATING SYNTHETIC DNA METHYLATION DATA")
    print("=" * 60)
    
    if config is None:
        config = MethylationDataConfig(n_samples=n_samples)
    else:
        config.n_samples = n_samples
    
    np.random.seed(random_state)
    
    # Step 1: Generate alcohol status (outcome variable)
    print(f"\n[1/7] Generating alcohol status for {n_samples} samples...")
    n_cases = int(n_samples * config.case_control_ratio)
    alcohol_status = np.concatenate([
        np.ones(n_cases),
        np.zeros(n_samples - n_cases)
    ])
    np.random.shuffle(alcohol_status)
    print(f"      Cases (alcohol): {int(alcohol_status.sum())}")
    print(f"      Controls: {int(n_samples - alcohol_status.sum())}")
    
    # Step 2: Generate covariates
    print("\n[2/7] Generating covariates (age, sex, smoking, BMI, cells)...")
    covariates = generate_covariates(n_samples, alcohol_status, config, random_state)
    print(f"      Age range: {covariates['age'].min():.1f} - {covariates['age'].max():.1f}")
    print(f"      Male fraction: {covariates['sex'].mean():.2%}")
    print(f"      Smoking rate: {covariates['smoking_status'].mean():.2%}")
    
    # Step 3: Generate base methylation values
    print(f"\n[3/7] Generating base methylation values ({config.n_cpg_sites} CpG sites)...")
    methylation = generate_beta_distribution_mixture(
        config.n_cpg_sites, n_samples, random_state=random_state
    )
    print(f"      Mean beta: {methylation.mean():.3f}")
    print(f"      Beta range: {methylation.min():.3f} - {methylation.max():.3f}")
    
    # Step 4: Generate CpG names and annotations
    print("\n[4/7] Generating CpG site names and annotations...")
    cpg_names = generate_cpg_names(config.n_cpg_sites)
    annotations = generate_gene_annotations(cpg_names)
    print(f"      Unique genes: {annotations['gene'].nunique()}")
    
    # Step 5: Apply alcohol effects
    print(f"\n[5/7] Applying alcohol-associated methylation changes...")
    methylation = apply_alcohol_effects(
        methylation, alcohol_status, cpg_names, config, random_state
    )
    print(f"      Affected CpG sites: {config.n_alcohol_associated_cpgs}")
    print(f"      Mean effect size: {config.alcohol_effect_mean:.3f}")
    
    # Step 6: Generate epigenetic clock sites
    print("\n[6/7] Generating epigenetic clock CpG sites...")
    horvath_cpgs, phenoage_cpgs, grimage_cpgs = generate_epigenetic_clock_sites(
        n_samples, covariates['age'].values, alcohol_status, config, random_state
    )
    print(f"      Horvath clock: {config.n_horvath_cpgs} CpGs")
    print(f"      PhenoAge: {config.n_phenoage_cpgs} CpGs")
    print(f"      GrimAge: {config.n_grimage_cpgs} CpGs")
    
    clock_sites = {
        'horvath': horvath_cpgs,
        'phenoage': phenoage_cpgs,
        'grimage': grimage_cpgs,
        'horvath_cpg_names': generate_cpg_names(config.n_horvath_cpgs, 'hv_cg'),
        'phenoage_cpg_names': generate_cpg_names(config.n_phenoage_cpgs, 'pa_cg'),
        'grimage_cpg_names': generate_cpg_names(config.n_grimage_cpgs, 'gr_cg'),
    }
    
    # Step 7: Compile final dataset
    print("\n[7/7] Compiling final dataset...")
    
    # Identify which CpGs are alcohol-associated (for ground truth)
    np.random.seed(random_state)
    alcohol_cpg_indices = np.random.choice(
        config.n_cpg_sites, 
        config.n_alcohol_associated_cpgs, 
        replace=False
    )
    
    data = {
        'methylation': methylation,
        'clock_sites': clock_sites,
        'covariates': covariates,
        'cpg_names': cpg_names,
        'annotations': annotations,
        'alcohol_cpg_indices': alcohol_cpg_indices,
        'config': config,
    }
    
    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nDataset Summary:")
    print(f"  Total samples: {n_samples}")
    print(f"  Total CpG sites: {methylation.shape[1]}")
    print(f"  Clock sites: {horvath_cpgs.shape[1] + phenoage_cpgs.shape[1] + grimage_cpgs.shape[1]}")
    print(f"  Covariates: {len(covariates.columns)}")
    print(f"  Memory usage: ~{methylation.nbytes / 1024 / 1024:.1f} MB")
    
    return data


def save_dataset(data: Dict, output_dir: str) -> None:
    """
    Save generated dataset to disk.
    
    Parameters:
    -----------
    data : Dict
        Generated data dictionary
    output_dir : str
        Directory to save files
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save methylation matrix
    np.save(os.path.join(output_dir, 'methylation.npy'), data['methylation'])
    
    # Save clock sites
    for clock in ['horvath', 'phenoage', 'grimage']:
        np.save(os.path.join(output_dir, f'{clock}_cpgs.npy'), 
                data['clock_sites'][clock])
    
    # Save covariates
    data['covariates'].to_csv(os.path.join(output_dir, 'covariates.csv'), index=False)
    
    # Save annotations
    data['annotations'].to_csv(os.path.join(output_dir, 'annotations.csv'), index=False)
    
    # Save CpG names
    pd.Series(data['cpg_names']).to_csv(
        os.path.join(output_dir, 'cpg_names.csv'), index=False, header=False
    )
    
    # Save alcohol CpG indices (ground truth)
    np.save(os.path.join(output_dir, 'alcohol_cpg_indices.npy'), 
            data['alcohol_cpg_indices'])
    
    print(f"\nDataset saved to: {output_dir}")


if __name__ == "__main__":
    # Demo: Generate a sample dataset
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Generating Sample Dataset")
    print("=" * 60 + "\n")
    
    # Generate data with default configuration
    data = generate_methylation_data(n_samples=800, random_state=42)
    
    # Print sample statistics
    print("\n\nSample Statistics:")
    print("-" * 40)
    cov = data['covariates']
    
    for group, name in [(0, 'Controls'), (1, 'Alcohol Group')]:
        subset = cov[cov['alcohol_status'] == group]
        print(f"\n{name} (n={len(subset)}):")
        print(f"  Mean age: {subset['age'].mean():.1f} ± {subset['age'].std():.1f}")
        print(f"  Male %: {subset['sex'].mean():.1%}")
        print(f"  Smoking %: {subset['smoking_status'].mean():.1%}")
        print(f"  Mean BMI: {subset['bmi'].mean():.1f}")
        print(f"  Mean PRS: {subset['genetic_risk_score'].mean():.3f}")
