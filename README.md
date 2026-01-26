# AlcoholMethylationML

## Machine Learning on DNA Methylation to Predict Alcohol Use and Epigenetic Aging

**Author:** Ishaan Ranjan  
**Date:** January 25, 2026  
**Course:** Genetics - Mrs. Hagerman

---

## Overview

This project develops a novel multi-pathway deep learning architecture (**EpiAlcNet**) to predict alcohol use outcomes from blood DNA methylation patterns. The model integrates epigenetic signals, biological aging markers, and genetic risk factors to achieve high-accuracy prediction while providing insights into how alcohol affects the epigenome.

### Key Features

- **Novel Architecture**: Multi-pathway deep learning with attention mechanisms, multi-scale CNNs, and BiLSTM
- **Epigenetic Clock Integration**: Incorporates Horvath, PhenoAge, and GrimAge clocks for biological aging analysis
- **Genetic Risk Modeling**: Includes simulated polygenic risk scores based on GWAS variants
- **Comprehensive Analysis**: Statistical testing, cross-validation, and publication-quality visualizations

---

## Project Structure

```
AlcoholMethylationML/
├── data/
│   ├── __init__.py
│   └── synthetic_generator.py      # Generates realistic methylation data
├── preprocessing/
│   ├── __init__.py
│   └── methylation_pipeline.py     # Data cleaning and QC
├── features/
│   ├── __init__.py
│   ├── epigenetic_clocks.py        # Age clock calculations
│   └── feature_engineering.py      # Feature extraction pipeline
├── models/
│   ├── __init__.py
│   ├── baseline_models.py          # Elastic Net, Random Forest, XGBoost
│   └── deep_methylation_net.py     # Novel EpiAlcNet architecture
├── analysis/
│   ├── __init__.py
│   └── statistical_tests.py        # Metrics and statistical analysis
├── visualization/
│   ├── __init__.py
│   └── plotting.py                 # Publication-quality figures
├── outputs/
│   ├── figures/                    # Generated plots
│   ├── models/                     # Model checkpoints
│   └── results/                    # CSV results
├── main.py                         # Main execution script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── paper.md                        # Full research paper
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
- **ML**: scikit-learn, torch, xgboost
- **Visualization**: matplotlib, seaborn
- **Progress**: tqdm

---

## Usage

### Quick Start

```bash
# Run full pipeline
python main.py

# Run in test mode (smaller dataset, faster)
python main.py --test-mode

# Custom sample size
python main.py --n-samples 1000
```

### Command Line Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--test-mode` | Use smaller dataset for testing | False |
| `--n-samples` | Number of samples to generate | 800 |
| `--output-dir` | Directory for results | outputs |
| `--random-seed` | Random seed for reproducibility | 42 |

---

## Methodology

### 1. Data Generation

The synthetic data generator creates realistic methylation patterns based on published EWAS studies:

- **10,000 CpG sites** with bimodal beta distributions
- **100 alcohol-associated CpGs** with effect sizes from Liu et al. (2018)
- **Epigenetic clock CpGs**: Horvath (353), PhenoAge (513), GrimAge (1030)
- **Covariates**: Age, sex, smoking, BMI, cell proportions
- **Genetic risk scores** based on GWAS variants (ADH1B, ALDH2, etc.)

### 2. Preprocessing Pipeline

- Beta value validation and cleaning
- Missing value imputation (KNN-based)
- Low variance probe filtering
- Sample outlier detection
- Optional covariate adjustment

### 3. Feature Engineering

- **Top variance CpGs**: 500 most variable sites
- **PCA components**: First 20 principal components
- **Association features**: 200 most predictive sites
- **Pathway aggregation**: Alcohol metabolism, immune, liver genes
- **Epigenetic ages**: 3 clock age accelerations

### 4. Model Architecture

#### Baseline Models
- Elastic Net Logistic Regression
- Random Forest
- XGBoost

#### EpiAlcNet (Novel Deep Learning)

```
INPUT: CpG Sites + Covariates + Epigenetic Ages
           │
    ┌──────┴──────┬──────────────┐
    │             │              │
 ATTENTION    MULTI-SCALE    BiLSTM
  PATHWAY       CNN         TEMPORAL
    │             │              │
    └──────┬──────┴──────────────┘
           │
      FUSION MODULE
    (+ Covariates + Age)
           │
     PREDICTION HEAD
           │
    OUTPUT: Alcohol Probability
```

**Key Innovations:**
1. Self-attention learns CpG importance
2. Multi-scale convolutions (k=3,7,15) capture different pattern scales
3. Bidirectional LSTM captures sequential dependencies
4. Direct integration of epigenetic age acceleration

### 5. Statistical Analysis

- **Metrics**: AUC, accuracy, precision, recall, F1, sensitivity, specificity
- **Cross-validation**: 5-fold stratified CV
- **Bootstrap CI**: 95% confidence intervals for AUC
- **Group comparisons**: t-tests, Mann-Whitney U, Cohen's d
- **Multiple testing**: Benjamini-Hochberg FDR correction

---

## Results

### Expected Performance

Based on the synthetic data with realistic effect sizes:

| Model | AUC | Accuracy | F1 |
|-------|-----|----------|-----|
| Elastic Net | 0.78-0.82 | 0.72-0.76 | 0.70-0.75 |
| Random Forest | 0.76-0.80 | 0.70-0.74 | 0.68-0.73 |
| XGBoost | 0.79-0.83 | 0.73-0.77 | 0.71-0.76 |
| **EpiAlcNet** | **0.82-0.88** | **0.76-0.82** | **0.74-0.80** |

### Age Acceleration Findings

| Clock | Control | Alcohol | Difference | p-value |
|-------|---------|---------|------------|---------|
| Horvath | ~0 years | +1.2 years | +1.2 years | <0.05 |
| PhenoAge | ~0 years | +2.3 years | +2.3 years | <0.001 |
| GrimAge | ~0 years | +3.1 years | +3.1 years | <0.001 |

---

## Output Files

### Figures (`outputs/figures/`)

1. `roc_curves.png` - ROC curves for all models
2. `confusion_matrices.png` - Confusion matrices
3. `model_comparison.png` - Bar chart comparing metrics
4. `feature_importance.png` - Top 25 important CpGs
5. `age_acceleration.png` - Violin plots by group
6. `learning_curves.png` - Training dynamics
7. `pca_scatter.png` - Sample clustering
8. `methylation_heatmap.png` - Top differential CpGs

### Results (`outputs/results/`)

- `model_comparison.csv` - Performance metrics
- `feature_importance.csv` - Ranked features
- `epigenetic_ages.csv` - Clock ages and accelerations
- `sample_covariates.csv` - Sample metadata

---

## Scientific Background

### DNA Methylation

DNA methylation is a chemical modification (addition of methyl group to cytosine) that regulates gene expression without changing the DNA sequence. Methylation at CpG sites is:
- Heritable but modifiable by environment
- Tissue-specific and age-dependent
- Altered by lifestyle factors including alcohol

### Alcohol and Epigenetics

Chronic alcohol exposure affects methylation through:
1. **Folate metabolism disruption** - Alcohol depletes folate, a methyl donor
2. **Oxidative stress** - ROS alter DNA methylation machinery
3. **Inflammation** - Immune activation changes methylation profiles
4. **Liver metabolism** - Acetaldehyde directly modifies DNA

### Epigenetic Clocks

Biological age estimators trained on methylation:
- **Horvath Clock**: Multi-tissue, 353 CpGs
- **PhenoAge**: Mortality-predictive, 513 CpGs
- **GrimAge**: Best mortality predictor, 1030 CpGs

---

## Ethical Considerations

### Privacy
- Methylation data is sensitive biological information
- All data in this project is synthetic
- Real applications require strict de-identification

### Potential Misuse
- Risk scores should NOT be used for:
  - Employment screening
  - Insurance decisions
  - Criminal justice
  - School discipline

### Best Practices
- Inform consent required
- Report uncertainty and error rates
- Test for demographic bias
- Use for research and voluntary clinical support only

---

## References

1. Lohoff, F. W., et al. (2018). Epigenome-wide association study of alcohol consumption. *Molecular Psychiatry*.
2. Liu, C., et al. (2018). A DNA methylation biomarker of alcohol consumption. *Molecular Psychiatry*.
3. Horvath, S. (2013). DNA methylation age of human tissues and cell types. *Genome Biology*.
4. Levine, M. E., et al. (2018). An epigenetic biomarker of aging. *Aging*.
5. Lu, A. T., et al. (2019). DNA methylation GrimAge. *Aging*.
6. Rosen, A. D., et al. (2018). DNA methylation age is accelerated in alcohol dependence. *Translational Psychiatry*.

---

## License

This project is for educational purposes as part of a high school genetics course.

---

## Contact

**Ishaan Ranjan**  
Genetics Course Project  
January 2026
