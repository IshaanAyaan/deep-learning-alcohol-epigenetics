# Machine Learning on DNA Methylation to Predict Alcohol Use and Epigenetic Aging

**A Multi-Pathway Approach Using Real Brain Methylation Data**

---

**Author:** Ishaan Ranjan  
**Institution:** High School Genetics Course  
**Instructor:** Mrs. Hagerman  
**Date:** January 2026

---

## Abstract

Alcohol use disorder (AUD) affects millions of individuals worldwide and is associated with significant morbidity and mortality. Traditional assessment methods rely on self-reports, which can be incomplete or biased. DNA methylation, an epigenetic modification that reflects environmental exposures, offers a promising biological marker for objective assessment of alcohol exposure. This study analyzes real DNA methylation data from the Gene Expression Omnibus (GSE49393), comprising postmortem prefrontal cortex samples from 48 individuals (23 with AUD, 25 controls). Using multiple machine learning approaches, we demonstrate that methylation patterns robustly distinguish individuals with AUD from controls, with Elastic Net regression achieving an area under the receiver operating characteristic curve (AUC) of **0.96**. Random Forest achieved 0.88 AUC, while our novel EpiAlcNet deep learning architecture achieved 0.84 AUC. Analysis of epigenetic age acceleration showed a trend toward accelerated aging in AUD cases (+0.57 years for PhenoAge), though effects were not statistically significant with this sample size. These findings demonstrate that DNA methylation in brain tissue contains strong, machine-detectable signals associated with alcohol use disorder.

**Keywords:** DNA methylation, epigenetics, alcohol use disorder, machine learning, epigenetic clocks, biological aging, GSE49393

---

## 1. Introduction

### 1.1 Background and Motivation

Alcohol use disorder (AUD) represents a significant global health challenge, affecting an estimated 400 million people worldwide and contributing to 3 million deaths annually according to the World Health Organization. Despite the availability of effective treatments, accurate assessment of alcohol consumption and early identification of at-risk individuals remains challenging. Traditional methods rely primarily on self-reported alcohol intake, which is subject to recall bias, social desirability effects, and underreporting.

The field of epigenetics offers a promising avenue for developing objective biological markers of alcohol exposure. Epigenetics refers to heritable changes in gene expression that occur without alterations to the underlying DNA sequence. Among epigenetic modifications, DNA methylation—the addition of a methyl group to cytosine bases, primarily at CpG dinucleotides—is the most extensively studied and amenable to high-throughput measurement.

### 1.2 DNA Methylation and Alcohol

Chronic alcohol consumption affects DNA methylation through multiple biological mechanisms. First, alcohol metabolism disrupts one-carbon metabolism, which is essential for providing methyl groups for DNA methylation. Specifically, alcohol oxidation consumes nicotinamide adenine dinucleotide (NAD+) and generates reactive oxygen species (ROS), which can damage DNA and alter the function of DNA methyltransferases (DNMTs). Second, alcohol promotes inflammation, which triggers widespread changes in the methylome of immune cells. Third, alcohol-induced liver damage alters hepatic gene expression patterns, with corresponding methylation changes detectable even in peripheral blood.

Multiple epigenome-wide association studies (EWAS) have identified reproducible DNA methylation signatures of alcohol consumption. Liu et al. (2018) identified 144 CpG sites associated with alcohol intake in a meta-analysis of over 9,600 individuals. Lohoff et al. (2018) found 96 differentially methylated CpG sites in individuals with AUD compared to controls.

### 1.3 Epigenetic Clocks and Biological Aging

Beyond individual CpG sites, DNA methylation patterns can be used to estimate "biological age"—a measure of physiological aging that may differ from chronological age. Epigenetic clocks are machine learning models trained to predict age from methylation values at specific CpG sites:

1. **Horvath Clock (2013)**: Uses 353 CpG sites to estimate multi-tissue biological age
2. **PhenoAge (Levine et al., 2018)**: Uses 513 CpG sites trained on clinical biomarkers
3. **GrimAge (Lu et al., 2019)**: Uses 1,030 CpG sites, best predictor of mortality

### 1.4 Research Questions

This study addresses three primary research questions:

1. Can machine learning models trained on real brain DNA methylation accurately predict alcohol use disorder status?
2. Is AUD associated with accelerated epigenetic aging in brain tissue?
3. How do deep learning architectures compare to traditional machine learning for this task?

---

## 2. Methods

### 2.1 Data Source

We analyzed publicly available DNA methylation data from the Gene Expression Omnibus (GEO), specifically dataset **GSE49393** (Zhang et al., 2013).

#### 2.1.1 Dataset Characteristics

| Characteristic | Value |
|----------------|-------|
| Total Samples | 48 |
| AUD Cases | 23 |
| Controls | 25 |
| Tissue | Postmortem prefrontal cortex |
| Platform | Illumina HumanMethylation450 BeadChip |
| Original CpG sites | 485,577 |
| CpG sites after QC | 50,000 |

The prefrontal cortex was chosen as it is directly relevant to the neurobiology of addiction and decision-making.

### 2.2 Data Acquisition and Processing

Data was automatically downloaded and processed using our custom GEO loader module:

```python
from data.geo_loader import load_geo_dataset
data = load_geo_dataset(gse_id='GSE49393')
```

#### 2.2.1 Quality Control Pipeline

1. **Probe filtering**: Removed probes with >5% missing values (55,170 removed)
2. **Value imputation**: Imputed remaining missing values with probe medians
3. **Variance filtering**: Removed probes with variance <0.0005 (235,739 removed)
4. **Feature selection**: Selected top 50,000 most variable CpGs
5. **Outlier detection**: Identified 3 sample outliers via PCA (retained for analysis)

### 2.3 Feature Engineering

#### 2.3.1 Feature Types

| Feature Type | Count | Description |
|-------------|-------|-------------|
| Top variance CpGs | 500 | Most variable sites |
| PCA components | 20 | Global methylation patterns |
| Association features | 200 | Sites associated with AUD status |
| Epigenetic ages | 3 | Clock age accelerations |

#### 2.3.2 Epigenetic Age Calculation

Epigenetic ages were calculated using simplified clock models. Age acceleration was computed as the residual from regressing epigenetic age on chronological age.

### 2.4 Model Development

#### 2.4.1 Elastic Net Logistic Regression

- Regularization: Combined L1 and L2 penalties (l1_ratio=0.5)
- Hyperparameter tuning: 5-fold cross-validation
- Handles high-dimensional data where features > samples

#### 2.4.2 Random Forest

- Ensemble of 100 decision trees
- Maximum features: sqrt(n_features)
- Minimum samples per split: 5

#### 2.4.3 EpiAlcNet Deep Learning Architecture

A novel multi-pathway architecture designed for methylation data:

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
           │
     OUTPUT: AUD Probability
```

**Key Components:**
- Self-attention module learns CpG importance
- Multi-scale CNN (k=3,7,15) captures local patterns
- Bidirectional LSTM captures sequential dependencies

**Training:**
- Optimizer: Adam (lr=0.001)
- Batch size: 32
- Early stopping: Patience of 10 epochs

### 2.5 Evaluation

#### 2.5.1 Data Split
- 80% training, 20% test
- Stratified by AUD status

#### 2.5.2 Metrics
- Primary: Area Under ROC Curve (AUC)
- Secondary: Accuracy, Precision, Recall, F1, Sensitivity, Specificity
- 95% confidence intervals via bootstrap (1000 iterations)

#### 2.5.3 Statistical Analysis for Age Acceleration
- Independent samples t-test
- Mann-Whitney U test
- Cohen's d effect size

---

## 3. Results

### 3.1 Dataset Characteristics

The final dataset included 48 samples after quality control, with relatively balanced groups (23 AUD cases, 25 controls).

### 3.2 Model Performance

All models achieved strong discrimination between AUD cases and controls.

**Table 1: Model Performance on GSE49393 (n=48)**

| Model | AUC | 95% CI | Accuracy | Precision | Recall | F1 |
|-------|-----|--------|----------|-----------|--------|-----|
| **Elastic Net** | **0.96** | 0.81–1.00 | 90% | 100% | 80% | 0.89 |
| Random Forest | 0.88 | 0.56–1.00 | 90% | 100% | 80% | 0.89 |
| EpiAlcNet | 0.84 | 0.43–1.00 | 70% | 100% | 40% | 0.57 |

Elastic Net achieved the highest AUC of 0.96, demonstrating that DNA methylation patterns in the prefrontal cortex robustly distinguish individuals with AUD from controls.

The simpler linear model (Elastic Net) outperformed the deep learning approach, which is expected given the small sample size (n=48). Neural networks typically require larger training sets to effectively learn complex patterns.

### 3.3 Epigenetic Age Acceleration

**Table 2: Age Acceleration by Group**

| Clock | Controls (n=25) | AUD Cases (n=23) | Difference | Cohen's d | p-value |
|-------|-----------------|------------------|------------|-----------|---------|
| Horvath | +0.08 ± 1.01 | -0.09 ± 1.12 | -0.17 years | -0.16 | 0.60 |
| PhenoAge | -0.27 ± 2.47 | +0.29 ± 2.18 | **+0.57 years** | 0.24 | 0.42 |
| GrimAge | +0.09 ± 3.37 | -0.10 ± 2.44 | -0.20 years | -0.07 | 0.82 |

Age acceleration differences were not statistically significant with this sample size. However, PhenoAge showed the expected trend, with AUD cases showing +0.57 years of acceleration relative to controls.

The lack of statistical significance is likely due to:
1. Small sample size (n=48) providing limited statistical power
2. Brain tissue may show different patterns than blood (most published studies used blood)
3. Postmortem changes may introduce additional variability

### 3.4 Feature Importance

**Table 3: Top 10 Predictive Features**

| Rank | Feature | Importance | Type |
|------|---------|------------|------|
| 1 | PC18 | 8.56% | Principal Component |
| 2 | cg20034712 | 6.88% | Association-based |
| 3 | cg10526376 | 5.52% | Variance-based |
| 4 | cg05029148 | 4.88% | Association-based |
| 5 | cg19149522 | 4.27% | Association-based |
| 6 | cg03794530 | 4.05% | Association-based |
| 7 | cg17319011 | 3.91% | Association-based |
| 8 | cg04495110 | 3.63% | Association-based |
| 9 | cg18661413 | 3.55% | Association-based |
| 10 | cg22022881 | 3.17% | Association-based |

Principal components and association-based CpG sites dominated the feature importance rankings, indicating that the predictive signal is distributed across the methylome rather than concentrated in a few sites.

---

## 4. Discussion

### 4.1 Summary of Findings

This study demonstrates that DNA methylation patterns in postmortem prefrontal cortex tissue can effectively distinguish individuals with alcohol use disorder from controls. Using real human data from GSE49393, Elastic Net regression achieved an outstanding AUC of 0.96, confirming the presence of robust, machine-detectable epigenetic signatures of AUD.

### 4.2 Comparison to Previous Work

Our results are stronger than those typically reported in blood-based studies, which generally achieve AUC values of 0.70–0.85. This may reflect:

1. **Tissue relevance**: Brain tissue is directly affected by alcohol and may show stronger methylation differences than peripheral blood
2. **Case definition**: GSE49393 includes individuals with clinically diagnosed AUD, which may show more pronounced epigenetic changes than general population studies of alcohol consumption
3. **Postmortem tissue**: May capture cumulative lifetime exposure better than living tissue samples

### 4.3 Model Selection

The Elastic Net model outperformed Random Forest and deep learning (EpiAlcNet), which is instructive:

- **Small sample sizes favor simpler models**: With n=48, regularized linear models can effectively identify predictive CpGs without overfitting
- **High feature-to-sample ratio**: With 50,000+ features and only 48 samples, L1 regularization provides essential feature selection
- **Deep learning requires more data**: EpiAlcNet would likely show advantages with sample sizes of 200+

### 4.4 Age Acceleration

While we observed the expected trend of accelerated aging in AUD cases (PhenoAge +0.57 years), effects were not statistically significant. This contrasts with previous blood-based studies reporting 2-5 years of acceleration. Possible explanations include:

1. **Power limitations**: Our sample size was insufficient to detect small-to-medium effects
2. **Tissue differences**: Brain tissue may show different aging patterns than blood
3. **Postmortem artifacts**: Postmortem interval and tissue handling may affect epigenetic age estimates

### 4.5 Clinical Implications

If validated in larger cohorts, brain methylation biomarkers could support:

- **Postmortem diagnosis**: Objective evidence of AUD in forensic contexts
- **Neuropathological research**: Understanding how alcohol affects brain epigenetics
- **Drug development**: Identifying therapeutic targets in brain pathways

### 4.6 Limitations

1. **Sample size**: n=48 is small for machine learning, limiting generalization
2. **Postmortem tissue**: Results may not generalize to living individuals
3. **Single dataset**: External validation in independent cohorts needed
4. **Confounding**: Smoking, medication use, and cause of death may influence results
5. **Ancestry**: Dataset composition may affect generalizability

### 4.7 Future Directions

1. Validate in larger brain tissue cohorts
2. Compare brain and blood methylation from same individuals
3. Conduct longitudinal studies with living participants
4. Apply to additional brain regions (hippocampus, striatum)

---

## 5. Conclusion

This study demonstrates that DNA methylation in the prefrontal cortex contains robust, machine-detectable signatures of alcohol use disorder. Analyzing real human data from GSE49393, our Elastic Net model achieved an AUC of 0.96, providing strong evidence that epigenetic patterns can objectively identify individuals with AUD. While epigenetic age acceleration showed expected trends (+0.57 years in cases for PhenoAge), larger samples would be needed to achieve statistical significance.

These findings—validated on real biological data—advance our understanding of alcohol's molecular footprint in the brain and provide a foundation for developing objective biomarkers of alcohol exposure. Future validation in larger, independent cohorts will be essential to establish clinical utility.

---

## 6. References

1. Zhang, H., et al. (2014). Differentially co-expressed genes in postmortem prefrontal cortex of individuals with alcohol use disorders: Influence on alcohol metabolism-related pathways. *Human Genetics*, 133(11), 1383-1394.

2. Liu, C., et al. (2018). A DNA methylation biomarker of alcohol consumption. *Molecular Psychiatry*, 23(2), 422-433.

3. Lohoff, F. W., et al. (2018). Epigenome-wide association study of alcohol consumption in N=6,604 clinically defined bipolar disorder subjects. *Molecular Psychiatry*, 23(11), 2221-2228.

4. Horvath, S. (2013). DNA methylation age of human tissues and cell types. *Genome Biology*, 14(10), R115.

5. Levine, M. E., et al. (2018). An epigenetic biomarker of aging for lifespan and healthspan. *Aging*, 10(4), 573-591.

6. Lu, A. T., et al. (2019). DNA methylation GrimAge strongly predicts lifespan and healthspan. *Aging*, 11(2), 303-327.

7. Rosen, A. D., et al. (2018). DNA methylation age is accelerated in alcohol dependence. *Translational Psychiatry*, 8(1), 182.

8. Bernabeu, E., et al. (2021). Blood-based epigenome-wide association study and prediction of alcohol consumption. *Clinical Epigenetics*, 13(1), 1-14.

---

## 7. Appendix

### Appendix A: Data Availability

The GSE49393 dataset is publicly available from the Gene Expression Omnibus:
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE49393

### Appendix B: Code Availability

All analysis code is available at:
https://github.com/IshaanAyaan/deep-learning-alcohol-epigenetics

### Appendix C: Running the Analysis

```bash
# Install dependencies
pip install -r requirements.txt

# Run with real GEO data
python3 main.py --data-source geo --geo-id GSE49393 --output-dir outputs_real
```

---

## Acknowledgments

This research was conducted as part of a high school genetics course project for ISEF 2026. I thank Mrs. Hagerman for guidance on genetics and epigenetics.

---

**Word Count:** Approximately 3,500 words (excluding references and appendices)
