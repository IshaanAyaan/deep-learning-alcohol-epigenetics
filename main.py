#!/usr/bin/env python3
"""
=============================================================================
MAIN EXECUTION SCRIPT
AlcoholMethylationML: DNA Methylation-Based Prediction of Alcohol Use
=============================================================================
Author: Ishaan Ranjan
Date: January 2026
Course: Genetics - Mrs. Hagerman

This script runs the complete pipeline for predicting alcohol use from
DNA methylation data using a novel multi-pathway deep learning architecture.

PIPELINE OVERVIEW:
==================

1. DATA GENERATION
   - Generate realistic synthetic methylation data
   - Create epigenetic clock CpG sites
   - Generate covariates and genetic risk scores

2. PREPROCESSING
   - Validate and clean methylation values
   - Quality control and filtering

3. FEATURE ENGINEERING
   - Extract top variance CpGs
   - PCA components
   - Epigenetic age calculation

4. MODEL TRAINING
   - Train baseline models (Elastic Net, Random Forest, XGBoost)
   - Train novel EpiAlcNet deep learning model
   - Cross-validation evaluation

5. STATISTICAL ANALYSIS
   - Calculate performance metrics
   - Compare age acceleration between groups
   - Feature importance analysis

6. VISUALIZATION
   - Generate publication-quality figures
   - Save results

USAGE:
======
    python main.py                  # Full pipeline
    python main.py --test-mode      # Quick test with smaller data
    python main.py --skip-training  # Use pre-trained models

=============================================================================
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='DNA Methylation ML Pipeline for Alcohol Use Prediction'
    )
    parser.add_argument(
        '--test-mode', action='store_true',
        help='Run in test mode with smaller dataset'
    )
    parser.add_argument(
        '--n-samples', type=int, default=800,
        help='Number of samples to generate (default: 800)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='outputs',
        help='Output directory (default: outputs)'
    )
    parser.add_argument(
        '--skip-training', action='store_true',
        help='Skip model training'
    )
    parser.add_argument(
        '--random-seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--data-source', type=str, default='synthetic',
        choices=['synthetic', 'geo'],
        help='Use synthetic or real GEO data (default: synthetic)'
    )
    parser.add_argument(
        '--geo-id', type=str, default='GSE49393',
        help='GEO accession number if using real data (default: GSE49393)'
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    
    # Parse arguments
    args = parse_args()
    
    # Adjust for test mode
    if args.test_mode:
        args.n_samples = 200
        print("\n[TEST MODE] Using reduced dataset size")
    
    # Set random seeds for reproducibility
    np.random.seed(args.random_seed)
    
    print("\n" + "=" * 70)
    print("   ALCOHOLMETHYLATION-ML: DNA METHYLATION PREDICTION PIPELINE")
    print("   Predicting Alcohol Use from Epigenetic Signatures")
    print("=" * 70)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Samples: {args.n_samples}")
    print(f"Output: {args.output_dir}")
    
    # Create output directories
    output_path = PROJECT_ROOT / args.output_dir
    figures_path = output_path / 'figures'
    models_path = output_path / 'models'
    results_path = output_path / 'results'
    
    for path in [output_path, figures_path, models_path, results_path]:
        path.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # STEP 1: DATA LOADING
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: DATA LOADING")
    print("=" * 70)
    
    if args.data_source == 'geo':
        # Load real GEO data
        print(f"\n[REAL DATA MODE] Loading {args.geo_id} from GEO...")
        from data.geo_loader import load_geo_dataset
        
        data = load_geo_dataset(
            gse_id=args.geo_id,
            random_state=args.random_seed
        )
        data_source_label = f"GEO {args.geo_id}"
    else:
        # Generate synthetic data
        print(f"\n[SYNTHETIC MODE] Generating {args.n_samples} synthetic samples...")
        from data.synthetic_generator import generate_methylation_data
        
        data = generate_methylation_data(
            n_samples=args.n_samples,
            random_state=args.random_seed
        )
        data_source_label = "Synthetic"
    
    # Extract key components
    methylation = data['methylation']
    covariates = data['covariates']
    clock_sites = data['clock_sites']
    cpg_names = data['cpg_names']
    annotations = data['annotations']
    
    # Get labels
    y = covariates['alcohol_status'].values.astype(int)
    
    print(f"\nDataset loaded ({data_source_label}):")
    print(f"  - Samples: {methylation.shape[0]}")
    print(f"  - CpG sites: {methylation.shape[1]}")
    print(f"  - Cases (alcohol): {y.sum()}")
    print(f"  - Controls: {len(y) - y.sum()}")
    
    # =========================================================================
    # STEP 2: PREPROCESSING
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: PREPROCESSING")
    print("=" * 70)
    
    from preprocessing.methylation_pipeline import MethylationPreprocessor
    
    preprocessor = MethylationPreprocessor(
        variance_threshold=0.001,
        detect_outliers=True,
        verbose=True
    )
    
    meth_clean, cpg_clean = preprocessor.fit_transform(methylation, cpg_names)
    
    print(f"\nAfter preprocessing:")
    print(f"  - CpG sites retained: {len(cpg_clean)}")
    
    # =========================================================================
    # STEP 3: EPIGENETIC AGE CALCULATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: EPIGENETIC AGE CALCULATION")
    print("=" * 70)
    
    from features.epigenetic_clocks import (
        EpigeneticClockCalculator, 
        compare_age_acceleration_by_group,
        print_age_acceleration_comparison
    )
    
    calculator = EpigeneticClockCalculator(verbose=True)
    age_results = calculator.calculate_all_clocks(
        clock_sites,
        covariates['age'].values
    )
    
    # Compare age acceleration by group
    age_comparison = compare_age_acceleration_by_group(age_results, y)
    print_age_acceleration_comparison(age_comparison)
    
    # =========================================================================
    # STEP 4: FEATURE ENGINEERING
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: FEATURE ENGINEERING")
    print("=" * 70)
    
    from features.feature_engineering import MethylationFeatureEngineer
    from sklearn.model_selection import train_test_split
    
    # Prepare data dictionary for feature engineering
    data_for_features = {
        'methylation': meth_clean,
        'cpg_names': cpg_clean,
        'annotations': annotations,
        'covariates': covariates,
        'clock_sites': clock_sites
    }
    
    # Feature engineering
    engineer = MethylationFeatureEngineer(
        n_top_variance=500,
        n_pca_components=20,
        n_association_features=200,
        include_clocks=True,
        include_covariates=True,
        verbose=True
    )
    
    X, feature_names = engineer.fit_transform(data_for_features, y=y)
    
    print(f"\nEngineered features: {X.shape[1]}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.random_seed
    )
    
    print(f"\nTrain set: {len(y_train)} samples")
    print(f"Test set: {len(y_test)} samples")
    
    # =========================================================================
    # STEP 5: MODEL TRAINING
    # =========================================================================
    if not args.skip_training:
        print("\n" + "=" * 70)
        print("STEP 5: MODEL TRAINING")
        print("=" * 70)
        
        # 5a. Baseline Models
        print("\n" + "-" * 50)
        print("5a. BASELINE MODELS")
        print("-" * 50)
        
        from models.baseline_models import train_baseline_models
        
        baseline_results = train_baseline_models(
            X_train, y_train, X_test, y_test,
            feature_names, cv_folds=5, verbose=True
        )
        
        # 5b. Deep Learning Model (EpiAlcNet)
        print("\n" + "-" * 50)
        print("5b. EPIALCNET DEEP LEARNING MODEL")
        print("-" * 50)
        
        from models.deep_methylation_net import EpiAlcNet, EpiAlcNetTrainer
        
        # Prepare features for EpiAlcNet
        # Split features into methylation, covariates, and age acceleration
        n_meth = 500 + 20 + 200  # variance + PCA + association
        n_cov = 5  # age, sex, smoking, bmi, prs
        n_age = 3  # horvath, phenoage, grimage acceleration
        
        # Handle feature dimensions
        if X_train.shape[1] >= n_meth + n_cov + n_age:
            X_meth_train = X_train[:, :n_meth].astype(np.float32)
            X_cov_train = X_train[:, n_meth:n_meth+n_cov].astype(np.float32)
            X_age_train = X_train[:, n_meth+n_cov:n_meth+n_cov+n_age].astype(np.float32)
            
            X_meth_test = X_test[:, :n_meth].astype(np.float32)
            X_cov_test = X_test[:, n_meth:n_meth+n_cov].astype(np.float32)
            X_age_test = X_test[:, n_meth+n_cov:n_meth+n_cov+n_age].astype(np.float32)
        else:
            # Use all available features
            n_meth = max(100, X_train.shape[1] - 8)
            n_cov = min(5, X_train.shape[1] - n_meth - 3)
            n_age = min(3, X_train.shape[1] - n_meth - n_cov)
            
            X_meth_train = X_train[:, :n_meth].astype(np.float32)
            X_cov_train = X_train[:, n_meth:n_meth+n_cov].astype(np.float32)
            X_age_train = X_train[:, n_meth+n_cov:n_meth+n_cov+n_age].astype(np.float32)
            
            X_meth_test = X_test[:, :n_meth].astype(np.float32)
            X_cov_test = X_test[:, n_meth:n_meth+n_cov].astype(np.float32)
            X_age_test = X_test[:, n_meth+n_cov:n_meth+n_cov+n_age].astype(np.float32)
        
        # Ensure minimum dimensions
        if X_cov_train.shape[1] == 0:
            X_cov_train = np.zeros((len(X_train), 1), dtype=np.float32)
            X_cov_test = np.zeros((len(X_test), 1), dtype=np.float32)
            n_cov = 1
        if X_age_train.shape[1] == 0:
            X_age_train = np.zeros((len(X_train), 1), dtype=np.float32)
            X_age_test = np.zeros((len(X_test), 1), dtype=np.float32)
            n_age = 1
        
        # Create model
        model = EpiAlcNet(
            n_cpg_features=X_meth_train.shape[1],
            n_covariate_features=X_cov_train.shape[1],
            n_age_features=X_age_train.shape[1],
            hidden_dim=64,
            cnn_channels=16,
            lstm_hidden=32,
            dropout=0.3
        )
        
        # Train model
        trainer = EpiAlcNetTrainer(
            model=model,
            learning_rate=1e-3,
            n_epochs=50 if not args.test_mode else 20,
            batch_size=32,
            patience=10,
            verbose=True
        )
        
        history = trainer.fit(
            X_meth_train, X_cov_train, X_age_train, y_train,
            X_meth_test, X_cov_test, X_age_test, y_test
        )
        
        # Get predictions
        y_pred_proba_dl = trainer.predict_proba(X_meth_test, X_cov_test, X_age_test)[:, 1]
        y_pred_dl = (y_pred_proba_dl > 0.5).astype(int)
    else:
        print("\n[SKIPPING MODEL TRAINING]")
        baseline_results = {}
        history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
        y_pred_proba_dl = np.random.random(len(y_test))
        y_pred_dl = (y_pred_proba_dl > 0.5).astype(int)
    
    # =========================================================================
    # STEP 6: STATISTICAL ANALYSIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: STATISTICAL ANALYSIS")
    print("=" * 70)
    
    from analysis.statistical_tests import (
        calculate_metrics, compare_models, 
        feature_importance_analysis, group_statistical_comparison
    )
    
    # Calculate metrics for all models
    evaluation_results = {}
    
    # Baseline models
    for name, result in baseline_results.items():
        model = result.model
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
        else:
            y_proba = model.model_.predict_proba(X_test)[:, 1]
            y_pred = model.model_.predict(X_test)
        
        eval_result = calculate_metrics(
            y_test, y_pred, y_proba,
            model_name=result.model_name,
            verbose=True
        )
        evaluation_results[name] = eval_result
    
    # EpiAlcNet
    if not args.skip_training:
        eval_result_dl = calculate_metrics(
            y_test, y_pred_dl, y_pred_proba_dl,
            model_name='EpiAlcNet',
            verbose=True
        )
        evaluation_results['epialcnet'] = eval_result_dl
    
    # Compare all models
    comparison_df = compare_models(list(evaluation_results.values()), verbose=True)
    
    # Feature importance from best baseline model
    if baseline_results:
        best_model_name = max(baseline_results.keys(), 
                             key=lambda k: baseline_results[k].mean_cv_auc)
        best_result = baseline_results[best_model_name]
        
        if best_result.feature_importance is not None:
            importance_df = feature_importance_analysis(
                best_result.feature_importance,
                feature_names,
                n_top=25,
                verbose=True
            )
        else:
            importance_df = pd.DataFrame()
    else:
        importance_df = pd.DataFrame()
    
    # Age acceleration statistical comparison
    for clock in ['horvath', 'phenoage', 'grimage']:
        col = f'{clock}_aa_residual'
        if col in age_results.columns:
            comparison_result = group_statistical_comparison(
                age_results[col].values[y == 0],
                age_results[col].values[y == 1],
                group1_name='Control',
                group2_name='Alcohol',
                variable_name=f'{clock.capitalize()} Age Acceleration',
                verbose=True
            )
    
    # =========================================================================
    # STEP 7: VISUALIZATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 7: VISUALIZATION")
    print("=" * 70)
    
    from visualization.plotting import (
        plot_roc_curves, plot_confusion_matrices, plot_model_comparison,
        plot_feature_importance, plot_age_acceleration, plot_learning_curves,
        plot_pca_scatter, plot_methylation_heatmap
    )
    
    # Compile results for visualization
    viz_results = {
        'evaluation_results': evaluation_results,
        'comparison_df': comparison_df,
        'feature_importance': importance_df if not importance_df.empty else None,
        'age_results': age_results,
        'labels': y,
        'training_history': history,
        'methylation': meth_clean,
        'cpg_names': cpg_clean
    }
    
    # Generate figures
    print("\nGenerating figures...")
    
    # 1. ROC curves
    if evaluation_results:
        plot_roc_curves(
            evaluation_results,
            save_path=str(figures_path / 'roc_curves.png')
        )
    
    # 2. Confusion matrices
    if evaluation_results:
        plot_confusion_matrices(
            evaluation_results,
            save_path=str(figures_path / 'confusion_matrices.png')
        )
    
    # 3. Model comparison
    if not comparison_df.empty:
        plot_model_comparison(
            comparison_df,
            save_path=str(figures_path / 'model_comparison.png')
        )
    
    # 4. Feature importance
    if not importance_df.empty:
        plot_feature_importance(
            importance_df,
            save_path=str(figures_path / 'feature_importance.png')
        )
    
    # 5. Age acceleration
    plot_age_acceleration(
        age_results, y,
        save_path=str(figures_path / 'age_acceleration.png')
    )
    
    # 6. Learning curves
    if history['train_loss']:
        plot_learning_curves(
            history,
            save_path=str(figures_path / 'learning_curves.png')
        )
    
    # 7. PCA scatter
    plot_pca_scatter(
        meth_clean, y,
        save_path=str(figures_path / 'pca_scatter.png')
    )
    
    # 8. Methylation heatmap
    plot_methylation_heatmap(
        meth_clean, cpg_clean, y,
        n_cpgs=50,
        save_path=str(figures_path / 'methylation_heatmap.png')
    )
    
    # =========================================================================
    # STEP 8: SAVE RESULTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 8: SAVING RESULTS")
    print("=" * 70)
    
    # Save comparison DataFrame
    comparison_df.to_csv(results_path / 'model_comparison.csv', index=False)
    print(f"  ✓ Model comparison: {results_path / 'model_comparison.csv'}")
    
    # Save feature importance
    if not importance_df.empty:
        importance_df.to_csv(results_path / 'feature_importance.csv', index=False)
        print(f"  ✓ Feature importance: {results_path / 'feature_importance.csv'}")
    
    # Save age results
    age_results.to_csv(results_path / 'epigenetic_ages.csv', index=False)
    print(f"  ✓ Epigenetic ages: {results_path / 'epigenetic_ages.csv'}")
    
    # Save covariates
    covariates.to_csv(results_path / 'sample_covariates.csv', index=False)
    print(f"  ✓ Sample covariates: {results_path / 'sample_covariates.csv'}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\nRESULTS SUMMARY:")
    print("-" * 40)
    
    if evaluation_results:
        best_model = max(evaluation_results.keys(), 
                        key=lambda k: evaluation_results[k].auc)
        best_auc = evaluation_results[best_model].auc
        print(f"Best Model: {evaluation_results[best_model].model_name}")
        print(f"Best AUC: {best_auc:.4f}")
    
    print(f"\nAge Acceleration (Alcohol vs Control):")
    for clock, stats in age_comparison.items():
        diff = stats['mean_difference']
        p = stats['t_pvalue']
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {clock.capitalize()}: {diff:+.2f} years (p={p:.2e}) {sig}")
    
    print(f"\nOutput Files:")
    print(f"  Figures: {figures_path}")
    print(f"  Results: {results_path}")
    
    print("\n" + "=" * 70)
    print("Thank you for using AlcoholMethylationML!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
