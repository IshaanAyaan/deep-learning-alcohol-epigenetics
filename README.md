# AlcoholMethylationML

## Machine Learning on DNA Methylation to Predict Alcohol Use and Epigenetic Aging

**Author:** Ishaan Ranjan  
**Date:** January 2026  
**Course:** Genetics - Mrs. Hagerman

---

## Overview

This project develops machine learning models to predict alcohol use disorder (AUD) from DNA methylation patterns using **real biological data** from the Gene Expression Omnibus (GEO). The analysis demonstrates that methylation patterns in brain tissue robustly distinguish individuals with AUD from controls, achieving an **AUC of 0.96** with Elastic Net regression.

### Key Findings

- **Elastic Net achieved 0.96 AUC** on real brain methylation data (GSE49393)
- Random Forest achieved 0.88 AUC
- EpiAlcNet deep learning model achieved 0.84 AUC
- PhenoAge showed +0.57 years acceleration in AUD cases (not significant with n=48)

### Data Source

**GSE49393** (Zhang et al., 2013)
- 48 postmortem prefrontal cortex samples
- 23 individuals with Alcohol Use Disorder (AUD)
- 25 matched controls
- Illumina HumanMethylation450 BeadChip (450K array)
- ~50,000 CpG sites retained after quality control

---

## Project Structure

```
AlcoholMethylationML/
├── data/
│   ├── __init__.py
│   ├── synthetic_generator.py      # Generates synthetic methylation data
│   ├── geo_loader.py               # Downloads/processes real GEO data
│   └── geo_cache/                  # Cached downloaded datasets
├── preprocessing/
│   ├── __init__.py
│   └── methylation_pipeline.py     # Data cleaning and QC
├── features/
│   ├── __init__.py
│   ├── epigenetic_clocks.py        # Age clock calculations
│   └── feature_engineering.py      # Feature extraction pipeline
├── models/
│   ├── __init__.py
│   ├── baseline_models.py          # Elastic Net, Random Forest
│   └── deep_methylation_net.py     # Novel EpiAlcNet architecture
├── analysis/
│   ├── __init__.py
│   └── statistical_tests.py        # Metrics and statistical analysis
├── visualization/
│   ├── __init__.py
│   └── plotting.py                 # Publication-quality figures
├── outputs/                        # Synthetic data results
├── outputs_real/                   # Real GEO data results
│   ├── figures/                    # Generated plots
│   └── results/                    # CSV results
├── main.py                         # Main execution script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── paper.md                        # Full research paper
└── submit_paper.tex                # LaTeX submission version
```

---

## Installation

### Requirements

- Python 3.8+
- pip package manager

### Setup

```bash
# Navigate to project directory
cd AlcoholMethylationML

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- **Core**: numpy, pandas, scipy
- **ML**: scikit-learn, torch
- **Visualization**: matplotlib, seaborn
- **Progress**: tqdm

---

## Usage

### Running with Real GEO Data (Recommended)

```bash
# Run with real brain methylation data from GEO
python3 main.py --data-source geo --geo-id GSE49393 --output-dir outputs_real
```

### Running with Synthetic Data

```bash
# Run with synthetic data (for testing/development)
python3 main.py --data-source synthetic --n-samples 800 --output-dir outputs

# Test mode (smaller dataset)
python3 main.py --test-mode --data-source synthetic
```

### Command Line Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--data-source` | Use `geo` for real data or `synthetic` | synthetic |
| `--geo-id` | GEO accession number | GSE49393 |
| `--test-mode` | Use smaller dataset for testing | False |
| `--n-samples` | Number of synthetic samples | 800 |
| `--output-dir` | Directory for results | outputs |
| `--random-seed` | Random seed for reproducibility | 42 |

---

## Results on Real Data (GSE49393)

### Model Performance

| Model | AUC | 95% CI | Accuracy | Precision | Recall |
|-------|-----|--------|----------|-----------|--------|
| **Elastic Net** | **0.96** | 0.81–1.00 | 90% | 100% | 80% |
| Random Forest | 0.88 | 0.56–1.00 | 90% | 100% | 80% |
| EpiAlcNet | 0.84 | 0.43–1.00 | 70% | 100% | 40% |

### Epigenetic Age Acceleration

| Clock | Controls | AUD Cases | Difference | P-value |
|-------|----------|-----------|------------|---------|
| Horvath | +0.08 years | -0.09 years | -0.17 years | 0.60 |
| PhenoAge | -0.27 years | +0.29 years | **+0.57 years** | 0.42 |
| GrimAge | +0.09 years | -0.10 years | -0.20 years | 0.82 |

> **Note**: Age acceleration differences were not statistically significant with this sample size (n=48). Larger samples (n>200) would be needed to detect the 2-3 year effects reported in previous literature.

### Top Predictive Features

1. PC18 (Principal Component)
2. cg20034712 (association-based CpG)
3. cg10526376 (variance-based CpG)
4. cg05029148 (association-based CpG)
5. cg19149522 (association-based CpG)

---

## Methodology

### 1. Data Acquisition

The GEO loader (`data/geo_loader.py`) automatically:
- Downloads the GSE49393 series matrix from NCBI FTP
- Parses methylation values and phenotype data
- Extracts alcohol use disorder status, age, and sex

### 2. Preprocessing Pipeline

- Removes probes with >5% missing values
- Imputes remaining missing values using probe medians
- Filters low-variance probes (variance < 0.0005)
- Selects top 50,000 most variable CpGs
- Detects sample outliers using PCA

### 3. Feature Engineering

- **Top variance CpGs**: 500 most variable sites
- **PCA components**: First 20 principal components
- **Association features**: 200 most predictive sites
- **Epigenetic ages**: Horvath, PhenoAge, GrimAge clocks

### 4. Models

#### Elastic Net Logistic Regression
- L1 + L2 regularization for high-dimensional data
- 5-fold cross-validation for hyperparameter tuning

#### Random Forest
- Ensemble of 100 decision trees
- Handles nonlinear relationships

#### EpiAlcNet (Deep Learning)
- Multi-pathway architecture with:
  - Self-attention for CpG importance learning
  - Multi-scale CNN (kernels 3, 7, 15) for local patterns
  - Bidirectional LSTM for sequential dependencies
- Requires larger sample sizes for optimal performance

---

## Output Files

### Figures (`outputs_real/figures/`)

1. `roc_curves.png` - ROC curves for all models
2. `confusion_matrices.png` - Confusion matrices
3. `model_comparison.png` - Bar chart comparing metrics
4. `feature_importance.png` - Top 25 important features
5. `age_acceleration.png` - Violin plots by group
6. `learning_curves.png` - Training dynamics
7. `pca_scatter.png` - Sample clustering
8. `methylation_heatmap.png` - Top differential CpGs

### Results (`outputs_real/results/`)

- `model_comparison.csv` - Performance metrics
- `feature_importance.csv` - Ranked features
- `epigenetic_ages.csv` - Clock ages and accelerations
- `sample_covariates.csv` - Sample metadata

---

## Scientific Background

### DNA Methylation

DNA methylation is a chemical modification (addition of methyl group to cytosine) that regulates gene expression. Methylation at CpG sites is:
- Heritable but modifiable by environment
- Tissue-specific and age-dependent
- Altered by lifestyle factors including alcohol

### Alcohol and Epigenetics

Chronic alcohol exposure affects methylation through:
1. **Folate metabolism disruption** - Alcohol depletes methyl donors
2. **Oxidative stress** - Reactive oxygen species alter DNA
3. **Inflammation** - Immune activation changes methylation profiles
4. **Neurodegeneration** - Brain tissue shows distinct patterns

### Dataset: GSE49393

Published by Zhang et al. (2013), this dataset examined methylation changes in the prefrontal cortex of individuals with alcohol use disorder. The prefrontal cortex is critical for decision-making and is particularly affected by chronic alcohol exposure.

---

## References

1. Zhang, H., et al. (2014). Differentially co-expressed genes in postmortem prefrontal cortex of individuals with alcohol use disorders. *Human Genetics*.
2. Liu, C., et al. (2018). A DNA methylation biomarker of alcohol consumption. *Molecular Psychiatry*.
3. Horvath, S. (2013). DNA methylation age of human tissues and cell types. *Genome Biology*.
4. Levine, M. E., et al. (2018). An epigenetic biomarker of aging. *Aging*.
5. Lu, A. T., et al. (2019). DNA methylation GrimAge. *Aging*.
6. Rosen, A. D., et al. (2018). DNA methylation age is accelerated in alcohol dependence. *Translational Psychiatry*.

---

## Ethical Considerations

### Privacy
- Methylation data is sensitive biological information
- Real applications require strict de-identification and consent

### Potential Misuse
- Risk predictions should NOT be used for:
  - Employment screening
  - Insurance decisions
  - Criminal justice
  - School discipline

---

## License

This project is for educational purposes as part of a high school genetics course and ISEF competition.

---

## Contact

**Ishaan Ranjan**  
ISEF 2026 Project  
January 2026
