"""
=============================================================================
VISUALIZATION MODULE
=============================================================================
AlcoholMethylationML Project
Author: Ishaan Ranjan
Date: January 2026

This module creates publication-quality visualizations for the DNA
methylation machine learning project. All figures are designed to be
suitable for academic publication and scientific communication.

FIGURE TYPES:

1. MODEL PERFORMANCE
   - ROC curves with AUC comparison
   - Precision-Recall curves
   - Confusion matrices (heatmaps)
   - Model comparison bar charts

2. FEATURE ANALYSIS
   - Feature importance bar plots
   - Top CpG sites heatmap
   - PCA scatter plots

3. EPIGENETIC AGING
   - Age acceleration violin plots by group
   - Epigenetic vs chronological age scatter
   - Age acceleration correlation matrix

4. TRAINING DYNAMICS
   - Learning curves (loss over epochs)
   - Cross-validation performance

STYLE:
All figures use a consistent style suitable for publication:
- Clear labels and legends
- Colorblind-friendly palette
- High resolution (300 DPI)
- Vector format support (PDF)
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import warnings

# Set publication-quality defaults
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False
})

# Colorblind-friendly palette
COLORS = {
    'primary': '#0077BB',
    'secondary': '#EE7733',
    'tertiary': '#009988',
    'quaternary': '#CC3311',
    'control': '#0077BB',
    'alcohol': '#EE7733',
    'deep_learning': '#009988'
}


def plot_roc_curves(
    results_dict: Dict[str, Any],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot ROC curves for multiple models.
    
    Creates a publication-quality ROC curve comparison showing
    all models on the same axes with AUC values in legend.
    
    Parameters:
    -----------
    results_dict : Dict[str, Any]
        Dictionary with model names as keys and EvaluationResults as values
    save_path : str, optional
        Path to save figure
    figsize : Tuple[int, int]
        Figure dimensions
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary'], 
              COLORS['quaternary'], '#AA4499', '#332288']
    
    for i, (name, result) in enumerate(results_dict.items()):
        color = colors[i % len(colors)]
        ax.plot(
            result.fpr, result.tpr,
            color=color, linewidth=2,
            label=f'{name} (AUC = {result.auc:.3f})'
        )
    
    # Diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random (AUC = 0.500)')
    
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title('Receiver Operating Characteristic (ROC) Curves')
    ax.legend(loc='lower right')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_confusion_matrices(
    results_dict: Dict[str, Any],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4)
) -> plt.Figure:
    """
    Plot confusion matrices for all models as heatmaps.
    
    Parameters:
    -----------
    results_dict : Dict[str, Any]
        Dictionary with model results
    save_path : str, optional
        Path to save figure
    figsize : Tuple[int, int]
        Figure dimensions
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    n_models = len(results_dict)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    
    if n_models == 1:
        axes = [axes]
    
    for ax, (name, result) in zip(axes, results_dict.items()):
        cm = result.confusion_matrix
        
        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(
            cm_normalized, annot=True, fmt='.2%',
            cmap='Blues', ax=ax,
            xticklabels=['Control', 'Alcohol'],
            yticklabels=['Control', 'Alcohol'],
            cbar=False
        )
        
        # Add raw counts
        for i in range(2):
            for j in range(2):
                ax.text(j + 0.5, i + 0.7, f'(n={cm[i, j]})',
                       ha='center', va='center', fontsize=8)
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'{name}\nAUC: {result.auc:.3f}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metrics: List[str] = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1'],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Bar chart comparing model performance across metrics.
    
    Parameters:
    -----------
    comparison_df : pd.DataFrame
        DataFrame with model comparison (from statistical_tests.compare_models)
    metrics : List[str]
        Metrics to plot
    save_path : str, optional
        Path to save figure
    figsize : Tuple[int, int]
        Figure dimensions
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(metrics))
    width = 0.8 / len(comparison_df)
    
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary'], 
              COLORS['quaternary']]
    
    for i, (_, row) in enumerate(comparison_df.iterrows()):
        values = [row[m] for m in metrics]
        offset = (i - len(comparison_df)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, 
                     label=row['Model'], color=colors[i % len(colors)])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=7)
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1.15])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    n_top: int = 25,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Horizontal bar chart of top feature importances.
    
    Parameters:
    -----------
    importance_df : pd.DataFrame
        DataFrame with 'feature' and 'importance' columns
    n_top : int
        Number of top features to show
    save_path : str, optional
        Path to save figure
    figsize : Tuple[int, int]
        Figure dimensions
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get top N features
    top_features = importance_df.head(n_top).copy()
    top_features = top_features.iloc[::-1]  # Reverse for horizontal bar
    
    # Create color gradient
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, n_top))[::-1]
    
    bars = ax.barh(top_features['feature'], top_features['importance'], color=colors)
    
    ax.set_xlabel('Importance Score')
    ax.set_ylabel('Feature')
    ax.set_title(f'Top {n_top} Most Important Features')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.annotate(f'{width:.4f}',
                   xy=(width, bar.get_y() + bar.get_height()/2),
                   xytext=(3, 0),
                   textcoords="offset points",
                   ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_age_acceleration(
    age_results: pd.DataFrame,
    alcohol_status: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Violin plots of age acceleration by alcohol status.
    
    Parameters:
    -----------
    age_results : pd.DataFrame
        DataFrame with age acceleration columns
    alcohol_status : np.ndarray
        Binary alcohol status labels
    save_path : str, optional
        Path to save figure
    figsize : Tuple[int, int]
        Figure dimensions
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    clocks = ['horvath', 'phenoage', 'grimage']
    clock_labels = ['Horvath Clock', 'PhenoAge Clock', 'GrimAge Clock']
    
    for ax, clock, label in zip(axes, clocks, clock_labels):
        col = f'{clock}_aa_residual'
        if col not in age_results.columns:
            col = f'{clock}_aa_simple'
        
        if col not in age_results.columns:
            continue
        
        # Create DataFrame for plotting
        plot_data = pd.DataFrame({
            'Group': ['Control' if s == 0 else 'Alcohol' for s in alcohol_status],
            'Age Acceleration': age_results[col].values
        })
        
        # Violin plot
        parts = ax.violinplot(
            [plot_data[plot_data['Group'] == 'Control']['Age Acceleration'],
             plot_data[plot_data['Group'] == 'Alcohol']['Age Acceleration']],
            positions=[0, 1], showmeans=True, showmedians=True
        )
        
        # Color the violins
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(COLORS['control'] if i == 0 else COLORS['alcohol'])
            pc.set_alpha(0.7)
        
        # Add individual points with jitter
        for i, group in enumerate(['Control', 'Alcohol']):
            y = plot_data[plot_data['Group'] == group]['Age Acceleration'].values
            x = np.random.normal(i, 0.04, len(y))
            ax.scatter(x, y, alpha=0.3, s=10, 
                      color=COLORS['control'] if i == 0 else COLORS['alcohol'])
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Control', 'Alcohol'])
        ax.set_ylabel('Age Acceleration (years)')
        ax.set_title(label)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Add statistical annotation
        from scipy import stats
        control_vals = plot_data[plot_data['Group'] == 'Control']['Age Acceleration']
        alcohol_vals = plot_data[plot_data['Group'] == 'Alcohol']['Age Acceleration']
        t_stat, p_val = stats.ttest_ind(control_vals, alcohol_vals)
        
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        ax.annotate(f'p = {p_val:.2e} {sig}', 
                   xy=(0.5, 0.95), xycoords='axes fraction',
                   ha='center', va='top', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_learning_curves(
    history: Dict[str, List],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 4)
) -> plt.Figure:
    """
    Plot training and validation loss over epochs.
    
    Parameters:
    -----------
    history : Dict[str, List]
        Training history with 'train_loss', 'val_loss', 'val_auc'
    save_path : str, optional
        Path to save figure
    figsize : Tuple[int, int]
        Figure dimensions
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], label='Training Loss',
                color=COLORS['primary'], linewidth=2)
    if 'val_loss' in history and history['val_loss']:
        axes[0].plot(epochs, history['val_loss'], label='Validation Loss',
                    color=COLORS['secondary'], linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    
    # AUC plot
    if 'val_auc' in history and history['val_auc']:
        axes[1].plot(epochs, history['val_auc'], label='Validation AUC',
                    color=COLORS['tertiary'], linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('AUC')
        axes[1].set_title('Validation AUC')
        axes[1].legend()
        
        # Mark best epoch
        best_epoch = np.argmax(history['val_auc']) + 1
        best_auc = max(history['val_auc'])
        axes[1].axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.5)
        axes[1].annotate(f'Best: {best_auc:.3f}',
                        xy=(best_epoch, best_auc),
                        xytext=(5, 5), textcoords='offset points')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_methylation_heatmap(
    methylation: np.ndarray,
    cpg_names: List[str],
    labels: np.ndarray,
    n_cpgs: int = 50,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Heatmap of top differential CpG sites.
    
    Parameters:
    -----------
    methylation : np.ndarray
        Methylation matrix (n_samples x n_cpgs)
    cpg_names : List[str]
        CpG site names
    labels : np.ndarray
        Sample labels (0=control, 1=alcohol)
    n_cpgs : int
        Number of top CpGs to show
    save_path : str, optional
        Path to save figure
    figsize : Tuple[int, int]
        Figure dimensions
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    from scipy import stats
    
    # Calculate t-statistics for each CpG
    t_stats = np.zeros(methylation.shape[1])
    for i in range(methylation.shape[1]):
        control = methylation[labels == 0, i]
        alcohol = methylation[labels == 1, i]
        t_stat, _ = stats.ttest_ind(control, alcohol)
        t_stats[i] = t_stat
    
    # Get top differential CpGs
    top_idx = np.argsort(np.abs(t_stats))[-n_cpgs:]
    
    # Subset data
    meth_subset = methylation[:, top_idx]
    cpg_subset = [cpg_names[i] for i in top_idx]
    
    # Sort samples by label
    sort_idx = np.argsort(labels)
    meth_sorted = meth_subset[sort_idx]
    labels_sorted = labels[sort_idx]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        meth_sorted.T, cmap='RdBu_r', center=0.5,
        xticklabels=False, yticklabels=cpg_subset if n_cpgs <= 20 else False,
        cbar_kws={'label': 'Methylation (β)'},
        ax=ax
    )
    
    # Add group separator
    n_control = (labels_sorted == 0).sum()
    ax.axvline(x=n_control, color='black', linewidth=2)
    
    ax.set_xlabel('Samples')
    ax.set_ylabel('CpG Sites')
    ax.set_title(f'Top {n_cpgs} Differentially Methylated CpG Sites')
    
    # Add labels
    ax.text(n_control/2, -0.5, 'Control', ha='center', va='bottom', fontsize=10)
    ax.text(n_control + (len(labels) - n_control)/2, -0.5, 'Alcohol', 
           ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_pca_scatter(
    methylation: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    PCA scatter plot of samples colored by group.
    
    Parameters:
    -----------
    methylation : np.ndarray
        Methylation matrix
    labels : np.ndarray
        Sample labels
    save_path : str, optional
        Path to save figure
    figsize : Tuple[int, int]
        Figure dimensions
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Standardize and reduce
    scaler = StandardScaler()
    meth_scaled = scaler.fit_transform(methylation)
    
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(meth_scaled)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, (label, color, name) in enumerate([
        (0, COLORS['control'], 'Control'),
        (1, COLORS['alcohol'], 'Alcohol')
    ]):
        mask = labels == label
        ax.scatter(
            pca_coords[mask, 0], pca_coords[mask, 1],
            c=color, label=name, alpha=0.7, s=50
        )
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax.set_title('PCA of DNA Methylation Data')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def create_all_figures(
    results: Dict,
    output_dir: str,
    verbose: bool = True
) -> Dict[str, str]:
    """
    Generate all figures for the project.
    
    Parameters:
    -----------
    results : Dict
        Dictionary containing all analysis results
    output_dir : str
        Directory to save figures
    verbose : bool
        Print progress
        
    Returns:
    --------
    Dict[str, str]
        Dictionary mapping figure names to file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    figure_paths = {}
    
    if verbose:
        print("\n" + "=" * 60)
        print("GENERATING FIGURES")
        print("=" * 60)
    
    # 1. ROC curves
    if 'evaluation_results' in results:
        path = str(output_path / 'roc_curves.png')
        plot_roc_curves(results['evaluation_results'], save_path=path)
        figure_paths['roc_curves'] = path
        if verbose:
            print(f"  ✓ ROC curves")
    
    # 2. Confusion matrices
    if 'evaluation_results' in results:
        path = str(output_path / 'confusion_matrices.png')
        plot_confusion_matrices(results['evaluation_results'], save_path=path)
        figure_paths['confusion_matrices'] = path
        if verbose:
            print(f"  ✓ Confusion matrices")
    
    # 3. Model comparison
    if 'comparison_df' in results:
        path = str(output_path / 'model_comparison.png')
        plot_model_comparison(results['comparison_df'], save_path=path)
        figure_paths['model_comparison'] = path
        if verbose:
            print(f"  ✓ Model comparison")
    
    # 4. Feature importance
    if 'feature_importance' in results:
        path = str(output_path / 'feature_importance.png')
        plot_feature_importance(results['feature_importance'], save_path=path)
        figure_paths['feature_importance'] = path
        if verbose:
            print(f"  ✓ Feature importance")
    
    # 5. Age acceleration
    if 'age_results' in results and 'labels' in results:
        path = str(output_path / 'age_acceleration.png')
        plot_age_acceleration(results['age_results'], results['labels'], save_path=path)
        figure_paths['age_acceleration'] = path
        if verbose:
            print(f"  ✓ Age acceleration")
    
    # 6. Learning curves
    if 'training_history' in results:
        path = str(output_path / 'learning_curves.png')
        plot_learning_curves(results['training_history'], save_path=path)
        figure_paths['learning_curves'] = path
        if verbose:
            print(f"  ✓ Learning curves")
    
    # 7. PCA scatter
    if 'methylation' in results and 'labels' in results:
        path = str(output_path / 'pca_scatter.png')
        plot_pca_scatter(results['methylation'], results['labels'], save_path=path)
        figure_paths['pca_scatter'] = path
        if verbose:
            print(f"  ✓ PCA scatter")
    
    # 8. Methylation heatmap
    if 'methylation' in results and 'cpg_names' in results and 'labels' in results:
        path = str(output_path / 'methylation_heatmap.png')
        plot_methylation_heatmap(
            results['methylation'], results['cpg_names'], results['labels'],
            save_path=path
        )
        figure_paths['methylation_heatmap'] = path
        if verbose:
            print(f"  ✓ Methylation heatmap")
    
    if verbose:
        print(f"\nGenerated {len(figure_paths)} figures in: {output_dir}")
    
    return figure_paths


if __name__ == "__main__":
    # Demo visualization
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Visualization Module")
    print("=" * 60)
    
    # Create sample data for demonstration
    np.random.seed(42)
    
    # Sample ROC data
    from dataclasses import dataclass
    
    @dataclass
    class DemoResult:
        model_name: str
        auc: float
        fpr: np.ndarray
        tpr: np.ndarray
        confusion_matrix: np.ndarray
    
    # Create demo results
    fpr = np.linspace(0, 1, 100)
    
    demo_results = {
        'Elastic Net': DemoResult(
            'Elastic Net', 0.82,
            fpr, np.clip(fpr * 0.7 + 0.3 + np.random.randn(100)*0.05, 0, 1),
            np.array([[40, 10], [8, 42]])
        ),
        'Random Forest': DemoResult(
            'Random Forest', 0.78,
            fpr, np.clip(fpr * 0.6 + 0.25 + np.random.randn(100)*0.05, 0, 1),
            np.array([[38, 12], [10, 40]])
        ),
        'EpiAlcNet': DemoResult(
            'EpiAlcNet', 0.88,
            fpr, np.clip(fpr * 0.8 + 0.35 + np.random.randn(100)*0.03, 0, 1),
            np.array([[44, 6], [5, 45]])
        )
    }
    
    # Plot ROC curves
    fig = plot_roc_curves(demo_results, figsize=(8, 6))
    plt.show()
    
    print("\nVisualization demonstration complete!")
