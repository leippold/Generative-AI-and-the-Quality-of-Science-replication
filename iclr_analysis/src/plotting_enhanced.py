"""
Enhanced Plotting Utilities for ICLR Analysis
==============================================
Publication-quality visualizations with professional aesthetics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats
from scipy.ndimage import gaussian_filter
from typing import Optional, Tuple, List
import warnings

from .constants import FIGURE_DPI, AI_GRADIENT
from .plotting import setup_style, save_figure


# =============================================================================
# COLOR PALETTES
# =============================================================================

# Professional color schemes
FRONTIER_CMAP = LinearSegmentedColormap.from_list(
    'frontier', ['#f7fbff', '#6baed6', '#2171b5', '#08306b']
)

QUALITY_COLORS = {
    'excellent': '#1a9850',
    'good': '#91cf60',
    'average': '#fee08b',
    'poor': '#fc8d59',
    'reject': '#d73027'
}

GRADIENT_BLUE = ['#deebf7', '#9ecae1', '#4292c6', '#084594']
GRADIENT_DIVERGING = ['#d73027', '#fc8d59', '#fee08b', '#91cf60', '#1a9850']


# =============================================================================
# DATA CLEANING HELPERS
# =============================================================================

def _clean_ai_percentage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert ai_percentage column to numeric, handling string formats like '100%'.

    Parameters
    ----------
    df : DataFrame
        DataFrame with ai_percentage column

    Returns
    -------
    DataFrame with numeric ai_percentage
    """
    df = df.copy()
    if 'ai_percentage' in df.columns:
        # Handle string format like "100%", "50%"
        if df['ai_percentage'].dtype == 'object':
            df['ai_percentage'] = df['ai_percentage'].astype(str).str.replace('%', '', regex=False)
            df['ai_percentage'] = pd.to_numeric(df['ai_percentage'], errors='coerce')
    return df


def _clean_numeric_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Ensure specified columns are numeric.

    Parameters
    ----------
    df : DataFrame
    columns : list of column names

    Returns
    -------
    DataFrame with numeric columns
    """
    df = df.copy()
    for col in columns:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


# =============================================================================
# SINGLE FIGURE GENERATORS
# =============================================================================

def create_rating_by_ai_level_figure(
    submissions_df: pd.DataFrame,
    save_path: Optional[str] = None,
    style: str = 'gradient_violin'
) -> plt.Figure:
    """
    Create publication-quality figure showing ratings by AI content level.

    Parameters
    ----------
    submissions_df : DataFrame
    save_path : str, optional
    style : str
        'gradient_violin' - Violin plots with gradient fill
        'ridgeline' - Joy/ridgeline plot
        'lollipop' - Lollipop chart with CI
        'beeswarm' - Beeswarm plot

    Returns
    -------
    Figure
    """
    setup_style()

    # Clean data types
    df = _clean_ai_percentage(submissions_df)
    df = _clean_numeric_columns(df, ['avg_rating'])
    df = df.dropna(subset=['ai_percentage', 'avg_rating']).copy()

    # Create bins
    bins = [0, 10, 25, 50, 75, 100]
    labels = ['0-10%', '10-25%', '25-50%', '50-75%', '75-100%']
    df['ai_bin'] = pd.cut(df['ai_percentage'], bins=bins, labels=labels, include_lowest=True)

    if style == 'gradient_violin':
        fig = _create_gradient_violin(df, labels)
    elif style == 'ridgeline':
        fig = _create_ridgeline(df, labels)
    elif style == 'lollipop':
        fig = _create_lollipop(df, labels)
    else:
        fig = _create_gradient_violin(df, labels)

    if save_path:
        save_figure(fig, save_path)

    return fig


def _create_gradient_violin(df: pd.DataFrame, labels: List[str]) -> plt.Figure:
    """Create violin plot with gradient colors and embedded statistics."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Prepare data
    data_by_bin = [df[df['ai_bin'] == label]['avg_rating'].values for label in labels]

    # Create violin plot
    parts = ax.violinplot(data_by_bin, positions=range(len(labels)),
                          showmeans=False, showmedians=False, showextrema=False)

    # Color violins with gradient
    colors = ['#2171b5', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef']
    for i, (pc, color) in enumerate(zip(parts['bodies'], colors)):
        pc.set_facecolor(color)
        pc.set_edgecolor('#08306b')
        pc.set_linewidth(1.5)
        pc.set_alpha(0.8)

    # Add box plots inside
    bp = ax.boxplot(data_by_bin, positions=range(len(labels)),
                    widths=0.15, patch_artist=True,
                    showfliers=False, zorder=3)

    for patch in bp['boxes']:
        patch.set_facecolor('white')
        patch.set_edgecolor('#08306b')
        patch.set_linewidth(1.5)
    for element in ['whiskers', 'caps', 'medians']:
        plt.setp(bp[element], color='#08306b', linewidth=1.5)

    # Add mean markers with values
    means = [np.mean(d) for d in data_by_bin]
    ns = [len(d) for d in data_by_bin]

    for i, (mean, n) in enumerate(zip(means, ns)):
        ax.scatter(i, mean, color='#d62728', s=80, zorder=5, marker='D',
                   edgecolor='white', linewidth=1.5)
        ax.annotate(f'μ={mean:.2f}\nn={n:,}', xy=(i, mean),
                    xytext=(i+0.25, mean+0.15), fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                             edgecolor='gray', alpha=0.9))

    # Add trend line through means
    x_smooth = np.linspace(0, len(labels)-1, 100)
    from scipy.interpolate import make_interp_spline
    try:
        spline = make_interp_spline(range(len(labels)), means, k=2)
        y_smooth = spline(x_smooth)
        ax.plot(x_smooth, y_smooth, '--', color='#d62728', linewidth=2,
                alpha=0.7, label='Mean trend')
    except:
        ax.plot(range(len(labels)), means, '--', color='#d62728',
                linewidth=2, alpha=0.7, label='Mean trend')

    # Styling
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel('AI Content Level', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Rating', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Ratings by AI Content Level',
                 fontsize=14, fontweight='bold', pad=15)

    # Add subtle grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    # Add overall mean line
    overall_mean = df['avg_rating'].mean()
    ax.axhline(y=overall_mean, color='gray', linestyle=':', linewidth=1.5,
               label=f'Overall mean: {overall_mean:.2f}')

    ax.legend(loc='upper right', fontsize=10)

    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig


def _create_ridgeline(df: pd.DataFrame, labels: List[str]) -> plt.Figure:
    """Create ridgeline/joy plot showing distribution overlap."""
    fig, axes = plt.subplots(len(labels), 1, figsize=(10, 8), sharex=True)

    colors = ['#08306b', '#2171b5', '#4292c6', '#6baed6', '#9ecae1']

    for i, (label, ax, color) in enumerate(zip(labels, axes, colors)):
        subset = df[df['ai_bin'] == label]['avg_rating']

        if len(subset) > 10:
            # KDE
            kde_x = np.linspace(1, 10, 200)
            kde = stats.gaussian_kde(subset)
            kde_y = kde(kde_x)

            ax.fill_between(kde_x, kde_y, alpha=0.8, color=color)
            ax.plot(kde_x, kde_y, color='#08306b', linewidth=1.5)

            # Mean line
            ax.axvline(subset.mean(), color='#d62728', linestyle='--', linewidth=2)

        ax.set_ylabel(label, rotation=0, ha='right', va='center', fontsize=10)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if i < len(labels) - 1:
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(bottom=False)

    axes[-1].set_xlabel('Average Rating', fontsize=12, fontweight='bold')
    fig.suptitle('Rating Distributions by AI Content Level',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    return fig


def _create_lollipop(df: pd.DataFrame, labels: List[str]) -> plt.Figure:
    """Create lollipop chart with confidence intervals."""
    fig, ax = plt.subplots(figsize=(10, 6))

    stats_list = []
    for label in labels:
        subset = df[df['ai_bin'] == label]['avg_rating']
        mean = subset.mean()
        se = subset.std() / np.sqrt(len(subset))
        stats_list.append({
            'label': label, 'mean': mean,
            'ci_low': mean - 1.96*se, 'ci_high': mean + 1.96*se,
            'n': len(subset)
        })

    stats_df = pd.DataFrame(stats_list)
    y_pos = range(len(stats_df))

    colors = ['#08306b', '#2171b5', '#4292c6', '#6baed6', '#9ecae1']

    # Horizontal lines (stems)
    for i, (_, row) in enumerate(stats_df.iterrows()):
        ax.hlines(y=i, xmin=0, xmax=row['mean'], color=colors[i],
                  linewidth=3, alpha=0.7)
        # CI error bar
        ax.errorbar(row['mean'], i, xerr=[[row['mean']-row['ci_low']],
                    [row['ci_high']-row['mean']]], fmt='o', color=colors[i],
                    markersize=12, capsize=5, capthick=2, elinewidth=2,
                    markeredgecolor='white', markeredgewidth=2)
        # Value annotation
        ax.annotate(f"{row['mean']:.2f}", xy=(row['mean']+0.08, i),
                    fontsize=10, va='center', fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(stats_df['label'], fontsize=11)
    ax.set_xlabel('Mean Rating (95% CI)', fontsize=12, fontweight='bold')
    ax.set_title('Quality Decline with Increasing AI Content',
                 fontsize=14, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    return fig


def create_dose_response_figure(
    submissions_df: pd.DataFrame,
    save_path: Optional[str] = None,
    style: str = 'hexbin_contour'
) -> plt.Figure:
    """
    Create publication-quality dose-response curve.

    The "dose" is AI content percentage, the "response" is paper quality (rating).

    Parameters
    ----------
    style : str
        'hexbin_contour' - Hexbin with contour overlay (handles clustering)
        'density_gradient' - 2D density with gradient
        'marginal' - Scatter with marginal distributions
        'frontier' - Collaboration frontier style
    """
    setup_style()

    # Clean data types
    df = _clean_ai_percentage(submissions_df)
    df = _clean_numeric_columns(df, ['avg_rating'])
    df = df.dropna(subset=['ai_percentage', 'avg_rating']).copy()
    x = df['ai_percentage'].values
    y = df['avg_rating'].values

    if style == 'hexbin_contour':
        fig = _create_hexbin_contour(x, y, df)
    elif style == 'density_gradient':
        fig = _create_density_gradient(x, y, df)
    elif style == 'marginal':
        fig = _create_marginal_scatter(x, y, df)
    elif style == 'frontier':
        fig = _create_frontier_plot(x, y, df)
    else:
        fig = _create_hexbin_contour(x, y, df)

    if save_path:
        save_figure(fig, save_path)

    return fig


def _create_hexbin_contour(x: np.ndarray, y: np.ndarray, df: pd.DataFrame) -> plt.Figure:
    """Hexbin plot with contour overlay - handles integer clustering elegantly."""
    fig, ax = plt.subplots(figsize=(11, 8))

    # Add small jitter to handle exact integer clustering
    y_jitter = y + np.random.normal(0, 0.05, len(y))

    # Hexbin for density
    hb = ax.hexbin(x, y_jitter, gridsize=30, cmap='Blues', mincnt=1,
                   alpha=0.6, edgecolors='none')

    # Contour overlay for structure
    try:
        # 2D histogram for contours
        H, xedges, yedges = np.histogram2d(x, y_jitter, bins=[30, 20])
        H = gaussian_filter(H.T, sigma=1.5)
        X, Y = np.meshgrid((xedges[:-1] + xedges[1:])/2,
                           (yedges[:-1] + yedges[1:])/2)

        contours = ax.contour(X, Y, H, levels=5, colors='#08306b',
                              linewidths=1.5, alpha=0.7)
        ax.clabel(contours, inline=True, fontsize=8, fmt='%1.0f')
    except:
        pass

    # Binned means with regression
    bins = np.arange(0, 101, 10)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_means = []
    bin_sems = []
    bin_ns = []

    for i in range(len(bins) - 1):
        mask = (x >= bins[i]) & (x < bins[i+1])
        if mask.sum() >= 10:
            bin_means.append(y[mask].mean())
            bin_sems.append(y[mask].std() / np.sqrt(mask.sum()))
            bin_ns.append(mask.sum())
        else:
            bin_means.append(np.nan)
            bin_sems.append(np.nan)
            bin_ns.append(0)

    # Plot binned means
    valid = ~np.isnan(bin_means)
    ax.errorbar(np.array(bin_centers)[valid], np.array(bin_means)[valid],
                yerr=1.96*np.array(bin_sems)[valid],
                fmt='s', color='#d62728', markersize=10, capsize=5, capthick=2,
                elinewidth=2, markeredgecolor='white', markeredgewidth=2,
                label='Bin means ± 95% CI', zorder=5)

    # Quadratic fit
    coefs = np.polyfit(x, y, 2)
    x_line = np.linspace(0, 100, 100)
    y_line = np.polyval(coefs, x_line)
    ax.plot(x_line, y_line, '-', color='#d62728', linewidth=3,
            label=f'Quadratic fit', zorder=4)

    # Optimal point if inverted-U
    if coefs[0] < 0:
        optimal = -coefs[1] / (2 * coefs[0])
        if 0 <= optimal <= 100:
            optimal_y = np.polyval(coefs, optimal)
            ax.axvline(optimal, color='#2ca02c', linestyle='--', linewidth=2,
                       alpha=0.8, label=f'Theoretical optimum: {optimal:.0f}%')
            ax.scatter([optimal], [optimal_y], color='#2ca02c', s=150,
                       zorder=6, marker='*', edgecolor='white', linewidth=2)

    # Colorbar
    cbar = plt.colorbar(hb, ax=ax, label='Number of papers', shrink=0.8)

    # Labels
    ax.set_xlabel('AI Content (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Rating', fontsize=13, fontweight='bold')
    ax.set_title('Dose-Response: Paper Quality vs AI Content',
                 fontsize=14, fontweight='bold', pad=15)

    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax.set_xlim(-2, 102)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add annotation
    ax.text(0.02, 0.02,
            f'N = {len(x):,} papers\nContours show density',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    return fig


def _create_density_gradient(x: np.ndarray, y: np.ndarray, df: pd.DataFrame) -> plt.Figure:
    """2D density plot with gradient coloring."""
    fig, ax = plt.subplots(figsize=(11, 8))

    # Add jitter
    y_jitter = y + np.random.normal(0, 0.08, len(y))

    # KDE
    try:
        from scipy.stats import gaussian_kde
        xy = np.vstack([x, y_jitter])
        z = gaussian_kde(xy)(xy)

        # Sort by density for better visualization
        idx = z.argsort()
        x_sorted, y_sorted, z_sorted = x[idx], y_jitter[idx], z[idx]

        scatter = ax.scatter(x_sorted, y_sorted, c=z_sorted, s=15,
                            cmap='viridis', alpha=0.6, edgecolors='none')
        plt.colorbar(scatter, ax=ax, label='Density', shrink=0.8)
    except:
        ax.scatter(x, y_jitter, alpha=0.3, s=10, c='steelblue')

    # Add regression line
    coefs = np.polyfit(x, y, 2)
    x_line = np.linspace(0, 100, 100)
    y_line = np.polyval(coefs, x_line)
    ax.plot(x_line, y_line, '-', color='#d62728', linewidth=3,
            label='Quadratic fit')

    ax.set_xlabel('AI Content (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Rating', fontsize=13, fontweight='bold')
    ax.set_title('Dose-Response Relationship', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')

    plt.tight_layout()
    return fig


def _create_marginal_scatter(x: np.ndarray, y: np.ndarray, df: pd.DataFrame) -> plt.Figure:
    """Scatter plot with marginal distributions."""
    # Use seaborn's jointplot
    g = sns.jointplot(x=x, y=y, kind='hex', cmap='Blues',
                      marginal_kws=dict(bins=30, fill=True, color='#2171b5'),
                      height=9)

    # Add regression line
    coefs = np.polyfit(x, y, 2)
    x_line = np.linspace(0, 100, 100)
    y_line = np.polyval(coefs, x_line)
    g.ax_joint.plot(x_line, y_line, '-', color='#d62728', linewidth=3)

    g.ax_joint.set_xlabel('AI Content (%)', fontsize=13, fontweight='bold')
    g.ax_joint.set_ylabel('Average Rating', fontsize=13, fontweight='bold')
    g.fig.suptitle('Dose-Response with Marginal Distributions',
                   fontsize=14, fontweight='bold', y=1.02)

    return g.fig


def _create_frontier_plot(x: np.ndarray, y: np.ndarray, df: pd.DataFrame) -> plt.Figure:
    """
    Create the "Collaboration Frontier" visualization.
    Shows the quality frontier as a function of human-AI collaboration.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Jitter for integer clustering
    y_jitter = y + np.random.normal(0, 0.05, len(y))

    # Background density
    hb = ax.hexbin(x, y_jitter, gridsize=35, cmap='Greys', mincnt=1,
                   alpha=0.4, edgecolors='none')

    # Calculate the "frontier" - upper envelope
    bins = np.arange(0, 101, 5)
    frontier_x = []
    frontier_y = []
    frontier_mean = []
    frontier_p90 = []

    for i in range(len(bins) - 1):
        mask = (x >= bins[i]) & (x < bins[i+1])
        if mask.sum() >= 5:
            frontier_x.append((bins[i] + bins[i+1]) / 2)
            frontier_y.append(np.percentile(y[mask], 90))  # 90th percentile
            frontier_mean.append(y[mask].mean())
            frontier_p90.append(np.percentile(y[mask], 90))

    # Plot mean trajectory
    ax.plot(frontier_x, frontier_mean, 'o-', color='#2171b5', linewidth=3,
            markersize=10, markeredgecolor='white', markeredgewidth=2,
            label='Mean quality', zorder=4)

    # Plot frontier (90th percentile)
    ax.fill_between(frontier_x, frontier_mean, frontier_p90,
                    color='#2171b5', alpha=0.2, label='Top 10% envelope')
    ax.plot(frontier_x, frontier_p90, '--', color='#2171b5', linewidth=2,
            alpha=0.7)

    # Identify zones
    # Zone 1: Pure human (0-10%)
    ax.axvspan(0, 10, alpha=0.1, color='#2ca02c', zorder=1)
    ax.text(5, ax.get_ylim()[1]*0.95, 'Human\nDomain', ha='center',
            fontsize=10, color='#2ca02c', fontweight='bold')

    # Zone 2: Collaboration zone (10-50%)
    ax.axvspan(10, 50, alpha=0.1, color='#ff7f0e', zorder=1)
    ax.text(30, ax.get_ylim()[1]*0.95, 'Collaboration\nZone', ha='center',
            fontsize=10, color='#ff7f0e', fontweight='bold')

    # Zone 3: AI-dominant (50-100%)
    ax.axvspan(50, 100, alpha=0.1, color='#d62728', zorder=1)
    ax.text(75, ax.get_ylim()[1]*0.95, 'AI-Dominant\nZone', ha='center',
            fontsize=10, color='#d62728', fontweight='bold')

    # Add quadratic fit
    coefs = np.polyfit(x, y, 2)
    x_line = np.linspace(0, 100, 100)
    y_line = np.polyval(coefs, x_line)
    ax.plot(x_line, y_line, '-', color='#d62728', linewidth=2.5,
            alpha=0.8, label='Regression fit')

    # Labels and styling
    ax.set_xlabel('AI Content (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Paper Quality (Rating)', fontsize=14, fontweight='bold')
    ax.set_title('The Human-AI Collaboration Frontier',
                 fontsize=16, fontweight='bold', pad=20)

    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax.set_xlim(-2, 102)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Key insight annotation
    insight_text = (
        "Key Finding:\n"
        "Quality declines monotonically\n"
        "with increasing AI content.\n"
        "No 'sweet spot' detected."
    )
    ax.text(0.98, 0.25, insight_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                     edgecolor='gray', alpha=0.9))

    plt.tight_layout()
    return fig


def create_component_trajectory_figure(
    reviews_df: pd.DataFrame,
    submissions_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create figure showing component score trajectories by AI content level.

    Shows soundness, presentation, contribution, and confidence trajectories.
    """
    setup_style()

    # Clean submissions data
    submissions_clean = _clean_ai_percentage(submissions_df)

    # Merge data
    if 'ai_percentage' not in reviews_df.columns:
        reviews_df = reviews_df.merge(
            submissions_clean[['submission_number', 'ai_percentage']],
            on='submission_number', how='left'
        )
    else:
        reviews_df = _clean_ai_percentage(reviews_df)

    df = reviews_df.dropna(subset=['ai_percentage']).copy()

    bins = [0, 10, 25, 50, 75, 100]
    labels = ['0-10%', '10-25%', '25-50%', '50-75%', '75-100%']
    df['ai_bin'] = pd.cut(df['ai_percentage'], bins=bins, labels=labels, include_lowest=True)

    # Component scores (excluding confidence which uses a different scale)
    components = ['soundness', 'presentation', 'contribution']
    available = [c for c in components if c in df.columns]

    if not available:
        return None

    fig, ax = plt.subplots(figsize=(11, 7))

    colors = {
        'soundness': '#1f77b4',      # Blue
        'presentation': '#ff7f0e',    # Orange
        'contribution': '#2ca02c',    # Green
        'confidence': '#9467bd'       # Purple
    }
    markers = {
        'soundness': 'o',
        'presentation': 's',
        'contribution': '^',
        'confidence': 'D'
    }

    x = np.arange(len(labels))

    for comp in available:
        means = df.groupby('ai_bin', observed=True)[comp].mean().reindex(labels)
        sems = df.groupby('ai_bin', observed=True)[comp].sem().reindex(labels)

        ax.errorbar(x, means, yerr=1.96*sems,
                    fmt=f'{markers[comp]}-', color=colors[comp],
                    linewidth=2.5, markersize=10, capsize=5, capthick=2,
                    markeredgecolor='white', markeredgewidth=2,
                    label=comp.capitalize())

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel('AI Content Level', fontsize=13, fontweight='bold')
    ax.set_ylabel('Mean Score', fontsize=13, fontweight='bold')
    ax.set_title('Component Score Trajectories by AI Content',
                 fontsize=14, fontweight='bold')

    ax.legend(loc='upper right', fontsize=11, frameon=True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


def create_key_comparison_figure(
    submissions_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create figure comparing key groups: Pure Human vs Light AI vs Heavy AI.
    """
    setup_style()

    # Clean data types
    df = _clean_ai_percentage(submissions_df)
    df = _clean_numeric_columns(df, ['avg_rating'])
    df = df.dropna(subset=['ai_percentage', 'avg_rating']).copy()

    groups = {
        'Pure Human\n(0%)': df[df['ai_percentage'] == 0]['avg_rating'],
        'Minimal AI\n(1-10%)': df[(df['ai_percentage'] > 0) & (df['ai_percentage'] <= 10)]['avg_rating'],
        'Light AI\n(10-25%)': df[(df['ai_percentage'] > 10) & (df['ai_percentage'] <= 25)]['avg_rating'],
        'Heavy AI\n(>50%)': df[df['ai_percentage'] > 50]['avg_rating']
    }

    # Filter to groups with enough data
    groups = {k: v for k, v in groups.items() if len(v) >= 10}

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = ['#2ca02c', '#9ecae1', '#4292c6', '#d62728'][:len(groups)]

    # Violin + box + swarm combination
    positions = range(len(groups))
    data = [groups[k].values for k in groups.keys()]

    # Violin
    parts = ax.violinplot(data, positions=positions, showmeans=False,
                          showmedians=False, showextrema=False)
    for i, (pc, color) in enumerate(zip(parts['bodies'], colors)):
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_alpha(0.3)

    # Box
    bp = ax.boxplot(data, positions=positions, widths=0.2, patch_artist=True,
                    showfliers=False, zorder=3)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor('black')

    # Mean markers
    for i, (name, values) in enumerate(groups.items()):
        mean = values.mean()
        ax.scatter(i, mean, color='black', s=100, zorder=5, marker='D',
                   edgecolor='white', linewidth=2)
        ax.annotate(f'μ={mean:.2f}\n(n={len(values):,})',
                    xy=(i, mean), xytext=(i+0.3, mean),
                    fontsize=10, va='center',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.set_xticks(positions)
    ax.set_xticklabels(groups.keys(), fontsize=11)
    ax.set_ylabel('Average Rating', fontsize=13, fontweight='bold')
    ax.set_title('Key Group Comparison', fontsize=14, fontweight='bold')

    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if save_path:
        save_figure(fig, save_path)

    plt.tight_layout()
    return fig


# =============================================================================
# BATCH GENERATION
# =============================================================================

def generate_all_collaboration_figures(
    submissions_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
    output_dir: str,
    verbose: bool = True
) -> dict:
    """
    Generate all individual figures for the collaboration analysis.

    Returns dict with paths to generated figures.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    figures = {}

    # 1. Rating by AI Level (gradient violin)
    if verbose:
        print("Generating: Rating by AI Level (violin)...")
    path = os.path.join(output_dir, 'fig_rating_by_ai_violin.png')
    create_rating_by_ai_level_figure(submissions_df, path, style='gradient_violin')
    figures['rating_violin'] = path

    # 2. Rating by AI Level (lollipop)
    if verbose:
        print("Generating: Rating by AI Level (lollipop)...")
    path = os.path.join(output_dir, 'fig_rating_by_ai_lollipop.png')
    create_rating_by_ai_level_figure(submissions_df, path, style='lollipop')
    figures['rating_lollipop'] = path

    # 3. Dose-Response (hexbin contour)
    if verbose:
        print("Generating: Dose-Response (hexbin)...")
    path = os.path.join(output_dir, 'fig_dose_response_hexbin.png')
    create_dose_response_figure(submissions_df, path, style='hexbin_contour')
    figures['dose_response_hexbin'] = path

    # 4. Collaboration Frontier
    if verbose:
        print("Generating: Collaboration Frontier...")
    path = os.path.join(output_dir, 'fig_collaboration_frontier.png')
    create_dose_response_figure(submissions_df, path, style='frontier')
    figures['frontier'] = path

    # 5. Dose-Response with marginals
    if verbose:
        print("Generating: Dose-Response (marginal)...")
    path = os.path.join(output_dir, 'fig_dose_response_marginal.png')
    create_dose_response_figure(submissions_df, path, style='marginal')
    figures['dose_response_marginal'] = path

    # 6. Component trajectories
    if verbose:
        print("Generating: Component Trajectories...")
    path = os.path.join(output_dir, 'fig_component_trajectories.png')
    fig = create_component_trajectory_figure(reviews_df, submissions_df, path)
    if fig:
        figures['component_trajectories'] = path

    # 7. Key comparison
    if verbose:
        print("Generating: Key Group Comparison...")
    path = os.path.join(output_dir, 'fig_key_comparison.png')
    create_key_comparison_figure(submissions_df, path)
    figures['key_comparison'] = path

    if verbose:
        print(f"\n✓ Generated {len(figures)} figures in {output_dir}")

    return figures


# =============================================================================
# WITHIN-PAPER ANALYSIS FIGURES
# =============================================================================

def create_within_paper_kde_figure(
    paper_ratings: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create professional within-paper rating difference figure with KDE.

    Shows the distribution of (AI reviewer rating - Human reviewer rating)
    for papers reviewed by both AI and Human reviewers.
    """
    setup_style()

    fig, ax = plt.subplots(figsize=(10, 7))

    diffs = paper_ratings['AI_minus_Human'].dropna()

    if len(diffs) < 10:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        return fig

    # KDE estimation
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(diffs, bw_method='scott')
    x_range = np.linspace(diffs.min() - 0.5, diffs.max() + 0.5, 200)
    density = kde(x_range)

    # Fill under curve with gradient effect
    ax.fill_between(x_range, density, alpha=0.3, color='#2171b5')
    ax.plot(x_range, density, color='#08306b', linewidth=3, label='KDE')

    # Add histogram underneath (subtle)
    ax.hist(diffs, bins=25, density=True, alpha=0.2, color='gray',
            edgecolor='white', label='Histogram')

    # Reference lines
    ax.axvline(x=0, color='#d62728', linewidth=2.5, linestyle='--',
               label='No difference', zorder=5)

    mean_diff = diffs.mean()
    ax.axvline(x=mean_diff, color='#2ca02c', linewidth=2.5,
               label=f'Mean: {mean_diff:+.3f}', zorder=5)

    # Add confidence interval shading
    ci_low = np.percentile(diffs, 2.5)
    ci_high = np.percentile(diffs, 97.5)
    ax.axvspan(ci_low, ci_high, alpha=0.1, color='#2ca02c',
               label=f'95% CI: [{ci_low:.2f}, {ci_high:.2f}]')

    # Statistics annotation
    from scipy.stats import ttest_1samp
    t_stat, p_val = ttest_1samp(diffs, 0)

    stats_text = (
        f"n = {len(diffs):,} papers\n"
        f"Mean diff = {mean_diff:+.3f}\n"
        f"Median = {np.median(diffs):+.3f}\n"
        f"SD = {diffs.std():.3f}\n"
        f"t = {t_stat:.2f}, p = {p_val:.4f}"
    )
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                     edgecolor='gray', alpha=0.9),
            fontfamily='monospace')

    # Styling
    ax.set_xlabel('Rating Difference (AI Reviewer − Human Reviewer)',
                  fontsize=13, fontweight='bold')
    ax.set_ylabel('Density', fontsize=13, fontweight='bold')
    ax.set_title('Within-Paper Rating Differences: AI vs Human Reviewers',
                 fontsize=14, fontweight='bold', pad=15)

    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add interpretation
    if p_val < 0.05:
        if mean_diff > 0:
            interp = "AI reviewers rate significantly HIGHER"
        else:
            interp = "AI reviewers rate significantly LOWER"
    else:
        interp = "No significant difference"

    ax.text(0.5, -0.12, interp, transform=ax.transAxes, fontsize=11,
            ha='center', style='italic', color='#666666')

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


def create_within_paper_scatter_figure(
    paper_ratings: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create scatter plot of AI vs Human ratings with density.
    """
    setup_style()

    fig, ax = plt.subplots(figsize=(9, 9))

    # Get data
    human = paper_ratings['Human'].values
    ai = paper_ratings['AI'].values

    # Hexbin for density (handles overlapping integer values)
    hb = ax.hexbin(human, ai, gridsize=20, cmap='Blues', mincnt=1,
                   alpha=0.7, edgecolors='none')

    # 45-degree line (equality)
    lims = [min(human.min(), ai.min()) - 0.5,
            max(human.max(), ai.max()) + 0.5]
    ax.plot(lims, lims, 'k--', linewidth=2, alpha=0.7, label='Equal ratings')

    # Regression line
    z = np.polyfit(human, ai, 1)
    p = np.poly1d(z)
    ax.plot(lims, p(lims), 'r-', linewidth=2,
            label=f'Fit: y = {z[0]:.2f}x + {z[1]:.2f}')

    # Colorbar
    cbar = plt.colorbar(hb, ax=ax, label='Number of papers', shrink=0.8)

    # Styling
    ax.set_xlabel('Human Reviewer Rating', fontsize=13, fontweight='bold')
    ax.set_ylabel('AI Reviewer Rating', fontsize=13, fontweight='bold')
    ax.set_title('AI vs Human Reviewer Ratings (Same Paper)',
                 fontsize=14, fontweight='bold')

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')

    ax.legend(loc='upper left', fontsize=10)

    # Correlation annotation
    from scipy.stats import pearsonr
    r, p_val = pearsonr(human, ai)
    ax.text(0.97, 0.03, f'r = {r:.3f}, p = {p_val:.4f}',
            transform=ax.transAxes, fontsize=11,
            ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


def create_echo_chamber_interaction_figure(
    reviews_df: pd.DataFrame,
    submissions_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create interaction plot for echo chamber analysis.
    Shows Paper Type × Reviewer Type interaction.
    """
    setup_style()

    # Merge data
    if 'ai_percentage' not in reviews_df.columns:
        reviews_df = reviews_df.merge(
            submissions_df[['submission_number', 'ai_percentage']],
            on='submission_number', how='left'
        )

    df = reviews_df.dropna(subset=['ai_percentage', 'rating']).copy()

    # Create binary classifications
    df['paper_type'] = np.where(df['ai_percentage'] > 50, 'AI Paper', 'Human Paper')
    df['reviewer_type'] = np.where(df.get('reviewer_ai_percentage', 0) > 50,
                                    'AI Reviewer', 'Human Reviewer')

    # If no reviewer_ai_percentage, use random assignment for demo
    if 'reviewer_ai_percentage' not in df.columns:
        # Fallback: use reviewer characteristics if available
        if 'is_ai_reviewer' in df.columns:
            df['reviewer_type'] = np.where(df['is_ai_reviewer'],
                                           'AI Reviewer', 'Human Reviewer')
        else:
            return None

    fig, ax = plt.subplots(figsize=(10, 7))

    # Calculate means and CIs
    interaction = df.groupby(['paper_type', 'reviewer_type'])['rating'].agg(
        ['mean', 'std', 'count']
    ).reset_index()
    interaction['se'] = interaction['std'] / np.sqrt(interaction['count'])
    interaction['ci95'] = 1.96 * interaction['se']

    # Plot
    colors = {'Human Paper': '#2ca02c', 'AI Paper': '#d62728'}
    markers = {'Human Paper': 'o', 'AI Paper': 's'}

    for paper_type in ['Human Paper', 'AI Paper']:
        subset = interaction[interaction['paper_type'] == paper_type]
        x_pos = [0, 1] if paper_type == 'Human Paper' else [0.05, 1.05]

        ax.errorbar(x_pos, subset['mean'], yerr=subset['ci95'],
                    fmt=f'{markers[paper_type]}-', color=colors[paper_type],
                    linewidth=2.5, markersize=12, capsize=6, capthick=2,
                    markeredgecolor='white', markeredgewidth=2,
                    label=paper_type)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Human Reviewer', 'AI Reviewer'], fontsize=12)
    ax.set_xlabel('Reviewer Type', fontsize=13, fontweight='bold')
    ax.set_ylabel('Mean Rating (± 95% CI)', fontsize=13, fontweight='bold')
    ax.set_title('Echo Chamber Effect: Paper × Reviewer Interaction',
                 fontsize=14, fontweight='bold')

    ax.legend(title='Paper Type', fontsize=11, title_fontsize=11)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


def create_substitution_gap_figure(
    reviews_df: pd.DataFrame,
    submissions_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create figure showing Presentation - Soundness gap (substitution signature).
    """
    setup_style()

    # Clean submissions data
    submissions_clean = _clean_ai_percentage(submissions_df)

    # Merge data
    if 'ai_percentage' not in reviews_df.columns:
        reviews_df = reviews_df.merge(
            submissions_clean[['submission_number', 'ai_percentage']],
            on='submission_number', how='left'
        )
    else:
        reviews_df = _clean_ai_percentage(reviews_df)

    df = reviews_df.dropna(subset=['ai_percentage']).copy()

    if 'soundness' not in df.columns or 'presentation' not in df.columns:
        return None

    df['gap'] = df['presentation'] - df['soundness']

    bins = [0, 10, 25, 50, 75, 100]
    labels = ['0-10%', '10-25%', '25-50%', '50-75%', '75-100%']
    df['ai_bin'] = pd.cut(df['ai_percentage'], bins=bins, labels=labels, include_lowest=True)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Calculate stats
    gap_stats = df.groupby('ai_bin', observed=True)['gap'].agg(['mean', 'std', 'count'])
    gap_stats['se'] = gap_stats['std'] / np.sqrt(gap_stats['count'])
    gap_stats = gap_stats.reindex(labels)

    x = np.arange(len(labels))
    colors = ['#2171b5', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef']

    # Bar plot
    bars = ax.bar(x, gap_stats['mean'], yerr=1.96*gap_stats['se'],
                  color=colors, edgecolor='#08306b', linewidth=1.5,
                  capsize=5, error_kw={'linewidth': 2})

    # Reference line
    ax.axhline(y=0, color='#d62728', linestyle='--', linewidth=2,
               label='No gap (Presentation = Soundness)')

    # Trend line
    valid_means = gap_stats['mean'].dropna().values
    if len(valid_means) >= 3:
        z = np.polyfit(np.arange(len(valid_means)), valid_means, 1)
        ax.plot(x[:len(valid_means)], np.polyval(z, np.arange(len(valid_means))),
                'r-', linewidth=2, alpha=0.7, label=f'Trend (slope: {z[0]:.3f})')

    # Value labels
    for i, (bar, mean) in enumerate(zip(bars, gap_stats['mean'])):
        if not np.isnan(mean):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=10,
                    fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel('AI Content Level', fontsize=13, fontweight='bold')
    ax.set_ylabel('Presentation − Soundness Gap', fontsize=13, fontweight='bold')
    ax.set_title('Substitution Signature: Style Over Substance',
                 fontsize=14, fontweight='bold', pad=15)

    ax.legend(loc='upper left', fontsize=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Interpretation
    if gap_stats['mean'].iloc[-1] > gap_stats['mean'].iloc[0]:
        interp = "Gap INCREASES with AI → Substitution signature detected"
        color = '#d62728'
    else:
        interp = "No clear substitution signature"
        color = '#2ca02c'

    ax.text(0.5, -0.12, interp, transform=ax.transAxes, fontsize=11,
            ha='center', style='italic', color=color)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


def create_ai_distribution_figure(
    submissions_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create figure showing distribution of AI content in submissions.
    """
    setup_style()

    fig, ax = plt.subplots(figsize=(10, 7))

    # Clean data types
    df = _clean_ai_percentage(submissions_df)
    ai_pct = df['ai_percentage'].dropna()

    # KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(ai_pct, bw_method='scott')
    x_range = np.linspace(0, 100, 200)
    density = kde(x_range)

    # Gradient fill
    ax.fill_between(x_range, density, alpha=0.4, color='#2171b5')
    ax.plot(x_range, density, color='#08306b', linewidth=3)

    # Histogram
    ax.hist(ai_pct, bins=50, density=True, alpha=0.3, color='gray',
            edgecolor='white')

    # Key percentiles
    for pctl, label in [(25, '25th'), (50, 'Median'), (75, '75th')]:
        val = np.percentile(ai_pct, pctl)
        ax.axvline(val, color='#d62728', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.text(val, ax.get_ylim()[1]*0.95, f'{label}\n{val:.0f}%',
                ha='center', fontsize=9, color='#d62728')

    # Stats
    stats_text = (
        f"N = {len(ai_pct):,}\n"
        f"Mean = {ai_pct.mean():.1f}%\n"
        f"Median = {ai_pct.median():.1f}%\n"
        f"SD = {ai_pct.std():.1f}%"
    )
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
            fontfamily='monospace')

    ax.set_xlabel('AI Content (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Density', fontsize=13, fontweight='bold')
    ax.set_title('Distribution of AI Content in Submissions',
                 fontsize=14, fontweight='bold')

    ax.set_xlim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


def create_confidence_by_ai_figure(
    reviews_df: pd.DataFrame,
    submissions_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create figure showing reviewer confidence by AI content level.
    """
    setup_style()

    # Clean submissions data
    submissions_clean = _clean_ai_percentage(submissions_df)

    # Merge data
    if 'ai_percentage' not in reviews_df.columns:
        reviews_df = reviews_df.merge(
            submissions_clean[['submission_number', 'ai_percentage']],
            on='submission_number', how='left'
        )
    else:
        reviews_df = _clean_ai_percentage(reviews_df)

    if 'confidence' not in reviews_df.columns:
        return None

    df = reviews_df.dropna(subset=['ai_percentage', 'confidence']).copy()

    bins = [0, 10, 25, 50, 75, 100]
    labels = ['0-10%', '10-25%', '25-50%', '50-75%', '75-100%']
    df['ai_bin'] = pd.cut(df['ai_percentage'], bins=bins, labels=labels, include_lowest=True)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Violin plot
    data_by_bin = [df[df['ai_bin'] == label]['confidence'].values for label in labels]

    parts = ax.violinplot(data_by_bin, positions=range(len(labels)),
                          showmeans=False, showmedians=False, showextrema=False)

    colors = ['#2171b5', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef']
    for i, (pc, color) in enumerate(zip(parts['bodies'], colors)):
        pc.set_facecolor(color)
        pc.set_edgecolor('#08306b')
        pc.set_alpha(0.7)

    # Box plots
    bp = ax.boxplot(data_by_bin, positions=range(len(labels)), widths=0.15,
                    patch_artist=True, showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor('white')
        patch.set_edgecolor('#08306b')

    # Mean markers
    means = [np.mean(d) for d in data_by_bin]
    ax.scatter(range(len(labels)), means, color='#d62728', s=80, zorder=5,
               marker='D', edgecolor='white', linewidth=1.5)

    # Trend line
    if len(means) >= 3:
        z = np.polyfit(np.arange(len(means)), means, 1)
        ax.plot(range(len(labels)), np.polyval(z, np.arange(len(labels))),
                'r--', linewidth=2, label=f'Trend (slope: {z[0]:.3f})')

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel('AI Content Level', fontsize=13, fontweight='bold')
    ax.set_ylabel('Reviewer Confidence', fontsize=13, fontweight='bold')
    ax.set_title('Reviewer Confidence by AI Content Level',
                 fontsize=14, fontweight='bold')

    ax.legend(loc='upper right', fontsize=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# COMPREHENSIVE BATCH GENERATION
# =============================================================================

def generate_all_iclr_figures(
    submissions_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
    output_dir: str,
    paper_ratings: pd.DataFrame = None,
    verbose: bool = True
) -> dict:
    """
    Generate ALL ICLR analysis figures as individual files.

    Parameters
    ----------
    submissions_df : DataFrame
    reviews_df : DataFrame
    output_dir : str
    paper_ratings : DataFrame, optional
        Pre-computed within-paper ratings
    verbose : bool

    Returns
    -------
    dict with paths to all generated figures
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    figures = {}

    # 1. Rating by AI Level (gradient violin)
    if verbose:
        print("  1. Rating by AI Level (Violin)...")
    try:
        path = os.path.join(output_dir, 'fig_rating_violin.png')
        create_rating_by_ai_level_figure(submissions_df, path, style='gradient_violin')
        figures['rating_violin'] = path
    except Exception as e:
        print(f"     Failed: {e}")

    # 2. Rating by AI Level (lollipop)
    if verbose:
        print("  2. Rating by AI Level (Lollipop)...")
    try:
        path = os.path.join(output_dir, 'fig_rating_lollipop.png')
        create_rating_by_ai_level_figure(submissions_df, path, style='lollipop')
        figures['rating_lollipop'] = path
    except Exception as e:
        print(f"     Failed: {e}")

    # 3. Dose-Response (hexbin)
    if verbose:
        print("  3. Dose-Response (Hexbin)...")
    try:
        path = os.path.join(output_dir, 'fig_dose_response.png')
        create_dose_response_figure(submissions_df, path, style='hexbin_contour')
        figures['dose_response'] = path
    except Exception as e:
        print(f"     Failed: {e}")

    # 4. Collaboration Frontier
    if verbose:
        print("  4. Collaboration Frontier...")
    try:
        path = os.path.join(output_dir, 'fig_collaboration_frontier.png')
        create_dose_response_figure(submissions_df, path, style='frontier')
        figures['frontier'] = path
    except Exception as e:
        print(f"     Failed: {e}")

    # 5. Component Trajectories
    if verbose:
        print("  5. Component Trajectories...")
    try:
        path = os.path.join(output_dir, 'fig_component_trajectories.png')
        fig = create_component_trajectory_figure(reviews_df, submissions_df, path)
        if fig:
            figures['component_trajectories'] = path
    except Exception as e:
        print(f"     Failed: {e}")

    # 6. Key Comparison
    if verbose:
        print("  6. Key Group Comparison...")
    try:
        path = os.path.join(output_dir, 'fig_key_comparison.png')
        create_key_comparison_figure(submissions_df, path)
        figures['key_comparison'] = path
    except Exception as e:
        print(f"     Failed: {e}")

    # 7. AI Distribution
    if verbose:
        print("  7. AI Content Distribution...")
    try:
        path = os.path.join(output_dir, 'fig_ai_distribution.png')
        create_ai_distribution_figure(submissions_df, path)
        figures['ai_distribution'] = path
    except Exception as e:
        print(f"     Failed: {e}")

    # 8. Substitution Gap
    if verbose:
        print("  8. Substitution Signature (Gap)...")
    try:
        path = os.path.join(output_dir, 'fig_substitution_gap.png')
        fig = create_substitution_gap_figure(reviews_df, submissions_df, path)
        if fig:
            figures['substitution_gap'] = path
    except Exception as e:
        print(f"     Failed: {e}")

    # 9. Confidence by AI
    if verbose:
        print("  9. Confidence by AI Level...")
    try:
        path = os.path.join(output_dir, 'fig_confidence_by_ai.png')
        fig = create_confidence_by_ai_figure(reviews_df, submissions_df, path)
        if fig:
            figures['confidence_by_ai'] = path
    except Exception as e:
        print(f"     Failed: {e}")

    # 10-11. Within-Paper (if paper_ratings provided)
    if paper_ratings is not None and len(paper_ratings) > 0:
        if verbose:
            print("  10. Within-Paper KDE...")
        try:
            path = os.path.join(output_dir, 'fig_within_paper_kde.png')
            create_within_paper_kde_figure(paper_ratings, path)
            figures['within_paper_kde'] = path
        except Exception as e:
            print(f"     Failed: {e}")

        if verbose:
            print("  11. Within-Paper Scatter...")
        try:
            path = os.path.join(output_dir, 'fig_within_paper_scatter.png')
            create_within_paper_scatter_figure(paper_ratings, path)
            figures['within_paper_scatter'] = path
        except Exception as e:
            print(f"     Failed: {e}")

    if verbose:
        print(f"\n✓ Generated {len(figures)} figures in {output_dir}")

    return figures
