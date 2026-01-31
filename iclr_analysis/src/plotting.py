"""
Plotting Utilities for ICLR Analysis.
=====================================
Standardized, publication-quality visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .constants import (
    FIGURE_DPI, FIGURE_STYLE, REVIEW_COLORS, 
    ECHO_CHAMBER_COLORS, PAPER_TYPE_COLORS, AI_GRADIENT
)


def setup_style():
    """Set up matplotlib style for consistent plots."""
    plt.style.use(FIGURE_STYLE)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = FIGURE_DPI
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['figure.figsize'] = (10, 6)


def save_figure(fig, path, dpi=FIGURE_DPI, tight=True):
    """
    Save figure with standard settings.
    
    Parameters
    ----------
    fig : Figure
    path : str
        Output path (can be .png, .pdf, .svg)
    dpi : int
    tight : bool
        Use tight_layout
    """
    if tight:
        fig.tight_layout()
    
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {path}")


def plot_interaction_2x2(mean_table, ax=None, title='Interaction Plot',
                         xlabel='Reviewer Type', ylabel='Mean Rating'):
    """
    Plot 2×2 interaction (Paper Type × Reviewer Type).
    
    Parameters
    ----------
    mean_table : DataFrame
        Pivot table with paper_type as index, reviewer_type as columns
    ax : Axes, optional
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    for paper_type in mean_table.index:
        if paper_type not in mean_table.index:
            continue
        
        values = mean_table.loc[paper_type].values
        marker = 'o' if 'Human' in paper_type else 's'
        color = PAPER_TYPE_COLORS.get(paper_type, 'gray')
        
        ax.plot(mean_table.columns, values, f'{marker}-', 
                label=paper_type, linewidth=2.5, markersize=10, color=color)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title='Paper Type')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_heatmap(data, ax=None, title='', cmap='RdYlGn', center=None,
                 annot=True, fmt='.2f', cbar_label=''):
    """
    Plot annotated heatmap.
    
    Parameters
    ----------
    data : DataFrame or 2D array
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    if center is None and isinstance(data, (pd.DataFrame, np.ndarray)):
        center = np.nanmean(data)
    
    sns.heatmap(data, annot=annot, fmt=fmt, cmap=cmap, center=center,
                ax=ax, annot_kws={'fontsize': 12, 'fontweight': 'bold'})
    
    ax.set_title(title)
    
    return ax


def plot_grouped_bars(data, x_labels, group_labels, ax=None, title='',
                      colors=None, xlabel='', ylabel='', show_values=True):
    """
    Plot grouped bar chart.
    
    Parameters
    ----------
    data : 2D array (groups × categories)
    x_labels : list
    group_labels : list
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    n_groups = len(group_labels)
    n_categories = len(x_labels)
    
    x = np.arange(n_categories)
    width = 0.8 / n_groups
    
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, n_groups))
    
    for i, (values, label) in enumerate(zip(data, group_labels)):
        offset = (i - n_groups / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=label, color=colors[i])
        
        if show_values:
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                       f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()
    
    return ax


def plot_distribution_comparison(groups, labels, ax=None, title='',
                                 kind='violin', colors=None):
    """
    Plot distribution comparison (violin, box, or histogram).
    
    Parameters
    ----------
    groups : list of arrays
    labels : list of str
    kind : str
        'violin', 'box', or 'hist'
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    if colors is None:
        colors = [REVIEW_COLORS.get(l, 'steelblue') for l in labels]
    
    if kind == 'violin':
        parts = ax.violinplot(groups, showmeans=True, showmedians=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i] if i < len(colors) else 'gray')
            pc.set_alpha(0.7)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=30, ha='right')
        
    elif kind == 'box':
        bp = ax.boxplot(groups, labels=labels, patch_artist=True)
        for i, (patch, color) in enumerate(zip(bp['boxes'], colors)):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.tick_params(axis='x', rotation=30)
        
    elif kind == 'hist':
        for group, label, color in zip(groups, labels, colors):
            ax.hist(group, bins=20, alpha=0.5, label=label, color=color)
        ax.legend()
    
    ax.set_title(title)
    
    return ax


def plot_effect_sizes(effects_df, ax=None, title='Effect Sizes',
                      ci_col=None, baseline=0):
    """
    Forest plot of effect sizes.
    
    Parameters
    ----------
    effects_df : DataFrame
        Must have 'label' and 'effect' columns
        Optional: 'ci_lower', 'ci_upper' for confidence intervals
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, len(effects_df) * 0.5 + 2))
    
    y_pos = range(len(effects_df))
    effects = effects_df['effect'].values
    labels = effects_df['label'].values
    
    colors = ['#2ca02c' if e > baseline else '#d62728' for e in effects]
    
    ax.barh(y_pos, effects, color=colors, alpha=0.7, edgecolor='black')
    
    # Add confidence intervals if available
    if 'ci_lower' in effects_df.columns and 'ci_upper' in effects_df.columns:
        xerr = np.array([
            effects - effects_df['ci_lower'].values,
            effects_df['ci_upper'].values - effects
        ])
        ax.errorbar(effects, y_pos, xerr=xerr, fmt='none', color='black', capsize=3)
    
    ax.axvline(x=baseline, color='black', linestyle='--', linewidth=1.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Effect Size')
    ax.set_title(title)
    
    return ax


def plot_permutation_distribution(observed, null_dist, ax=None, 
                                  title='Permutation Test'):
    """
    Plot permutation null distribution with observed value.
    
    Parameters
    ----------
    observed : float
    null_dist : array
        Null distribution from permutation test
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.hist(null_dist, bins=50, density=True, alpha=0.7, 
            color='steelblue', edgecolor='white')
    
    ax.axvline(x=observed, color='red', linewidth=2.5, 
               label=f'Observed: {observed:.4f}')
    ax.axvline(x=-observed, color='red', linewidth=2.5, linestyle='--', alpha=0.5)
    
    # P-value annotation
    p_value = np.mean(np.abs(null_dist) >= np.abs(observed))
    ax.text(0.95, 0.95, f'p = {p_value:.4f}', transform=ax.transAxes,
            ha='right', va='top', fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    ax.set_xlabel('Test Statistic')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    
    return ax


def plot_dose_response(x, y, ax=None, title='Dose-Response',
                       fit_type='quadratic', show_ci=True, n_bootstrap=1000):
    """
    Plot dose-response curve with fitted line and confidence band.
    
    Parameters
    ----------
    x : array-like
        AI percentage
    y : array-like
        Outcome (e.g., rating)
    fit_type : str
        'linear', 'quadratic', or 'lowess'
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    x, y = np.asarray(x), np.asarray(y)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    
    # Scatter plot
    ax.scatter(x, y, alpha=0.1, s=10, c='gray', label='Papers')
    
    # Bin means
    bins = np.arange(0, 101, 10)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_means = []
    bin_sems = []
    
    for i in range(len(bins) - 1):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        if mask.sum() >= 10:
            bin_means.append(y[mask].mean())
            bin_sems.append(y[mask].std() / np.sqrt(mask.sum()))
        else:
            bin_means.append(np.nan)
            bin_sems.append(np.nan)
    
    ax.errorbar(bin_centers, bin_means, yerr=np.array(bin_sems) * 1.96,
                fmt='o', color='blue', markersize=8, capsize=4,
                label='Bin means (95% CI)')
    
    # Fitted curve
    x_line = np.linspace(0, 100, 100)
    
    if fit_type == 'quadratic':
        coefs = np.polyfit(x, y, 2)
        y_line = np.polyval(coefs, x_line)
        
        # Find optimum if inverted-U
        if coefs[0] < 0:
            optimum = -coefs[1] / (2 * coefs[0])
            if 0 <= optimum <= 100:
                ax.axvline(optimum, color='green', linestyle='--', alpha=0.7,
                          label=f'Optimum: {optimum:.0f}%')
    elif fit_type == 'linear':
        coefs = np.polyfit(x, y, 1)
        y_line = np.polyval(coefs, x_line)
    else:
        # LOWESS would require statsmodels
        coefs = np.polyfit(x, y, 2)
        y_line = np.polyval(coefs, x_line)
    
    ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'{fit_type.capitalize()} fit')
    
    ax.set_xlabel('AI Content (%)')
    ax.set_ylabel('Rating')
    ax.set_title(title)
    ax.legend()
    
    return ax


def create_summary_figure(submissions_df, reviews_df, save_path=None):
    """
    Create three-panel summary figure for Section 5.1.
    
    A) Distribution of papers by AI content
    B) Mean rating by AI content bin
    C) Component scores (substitution signature)
    """
    setup_style()
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # Prepare data
    df = submissions_df.dropna(subset=['ai_percentage', 'avg_rating']).copy()
    
    bins = [0, 10, 25, 50, 75, 100]
    labels = ['0-10%', '10-25%', '25-50%', '50-75%', '75-100%']
    df['ai_bin'] = pd.cut(df['ai_percentage'], bins=bins, labels=labels, include_lowest=True)
    
    colors = AI_GRADIENT
    
    # Panel A: Distribution
    ax1 = axes[0]
    counts = df['ai_bin'].value_counts().reindex(labels)
    percentages = counts / counts.sum() * 100
    
    bars1 = ax1.bar(range(len(labels)), percentages, color=colors, 
                    edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('AI Content Level')
    ax1.set_ylabel('Percentage of Submissions')
    ax1.set_title('A. Distribution of AI Content', fontweight='bold', loc='left')
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels)
    
    for bar, pct in zip(bars1, percentages):
        ax1.annotate(f'{pct:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    
    # Panel B: Mean Rating
    ax2 = axes[1]
    rating_stats = df.groupby('ai_bin', observed=True)['avg_rating'].agg(['mean', 'std', 'count'])
    rating_stats['se'] = rating_stats['std'] / np.sqrt(rating_stats['count'])
    rating_stats['ci95'] = 1.96 * rating_stats['se']
    rating_stats = rating_stats.reindex(labels)
    
    bars2 = ax2.bar(range(len(labels)), rating_stats['mean'], yerr=rating_stats['ci95'],
                    color=colors, edgecolor='black', linewidth=0.5,
                    capsize=4, error_kw={'linewidth': 1.5})
    
    overall_mean = df['avg_rating'].mean()
    ax2.axhline(y=overall_mean, color='gray', linestyle='--', linewidth=1.5,
                label=f'Overall: {overall_mean:.2f}')
    
    ax2.set_xlabel('AI Content Level')
    ax2.set_ylabel('Mean Rating')
    ax2.set_title('B. Quality Declines Monotonically', fontweight='bold', loc='left')
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels)
    ax2.legend(loc='upper right', fontsize=9)
    
    # Panel C: Component Scores
    ax3 = axes[2]
    
    if 'ai_percentage' in reviews_df.columns:
        reviews_binned = reviews_df.dropna(subset=['ai_percentage']).copy()
        reviews_binned['ai_bin'] = pd.cut(reviews_binned['ai_percentage'], 
                                           bins=bins, labels=labels, include_lowest=True)
        
        components = ['soundness', 'presentation', 'contribution']
        available = [c for c in components if c in reviews_binned.columns]
        
        if available:
            component_stats = reviews_binned.groupby('ai_bin', observed=True)[available].mean()
            component_stats = component_stats.reindex(labels)
            
            x = np.arange(len(labels))
            width = 0.25
            
            comp_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            
            for i, (comp, color) in enumerate(zip(available, comp_colors)):
                offset = (i - len(available)/2 + 0.5) * width
                ax3.bar(x + offset, component_stats[comp], width, 
                       label=comp.capitalize(), color=color, edgecolor='black', linewidth=0.5)
            
            ax3.set_xlabel('AI Content Level')
            ax3.set_ylabel('Mean Score')
            ax3.set_title('C. Component Scores', fontweight='bold', loc='left')
            ax3.set_xticks(x)
            ax3.set_xticklabels(labels)
            ax3.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig
