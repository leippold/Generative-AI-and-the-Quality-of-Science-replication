"""
Enhanced Plotting Utilities for Retraction Analysis
====================================================
Publication-quality visualizations matching ICLR analysis style.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats
from scipy.ndimage import gaussian_filter
from typing import Optional, Dict, List, Tuple
import warnings

# Try to import lifelines
try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False


# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

# Professional color schemes
COHORT_COLORS = {
    'human': '#2E86AB',      # Professional blue
    'ai': '#A23B72',         # Deep magenta
    'human_light': '#7EB8DA',
    'ai_light': '#D4A5C9'
}

GRADIENT_HUMAN = ['#E8F4F8', '#7EB8DA', '#2E86AB', '#1A5276']
GRADIENT_AI = ['#F8E8F0', '#D4A5C9', '#A23B72', '#6B1D4A']

QUALITY_COLORS = {
    'excellent': '#1a9850',
    'good': '#91cf60',
    'average': '#fee08b',
    'poor': '#fc8d59',
    'critical': '#d73027'
}


def setup_style():
    """Configure matplotlib for publication-quality figures."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def save_figure(fig, path: str, dpi: int = 300):
    """Save figure with consistent settings."""
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved: {path}")


# =============================================================================
# KAPLAN-MEIER ENHANCED
# =============================================================================

def create_kaplan_meier_figure(
    df: pd.DataFrame,
    time_col: str = 'GIGO_Years',
    group_col: str = 'is_ai',
    max_time: float = 15,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create publication-quality Kaplan-Meier survival curves.

    Enhanced with:
    - Gradient confidence intervals
    - Median survival markers
    - Professional statistics box
    - Risk table option
    """
    if not HAS_LIFELINES:
        raise ImportError("lifelines required: pip install lifelines")

    setup_style()

    fig, ax = plt.subplots(figsize=(11, 8))

    df = df.copy()
    df['event'] = 1  # All are events (retractions)
    df[time_col] = df[time_col].clip(upper=max_time)

    results = {}
    kmf_objects = {}

    labels = {0: 'Human Papers', 1: 'AI-Assisted Papers'}
    colors = {0: COHORT_COLORS['human'], 1: COHORT_COLORS['ai']}
    light_colors = {0: COHORT_COLORS['human_light'], 1: COHORT_COLORS['ai_light']}

    for group_val in [0, 1]:
        mask = df[group_col] == group_val
        subset = df[mask]

        kmf = KaplanMeierFitter()
        kmf.fit(subset[time_col], event_observed=subset['event'],
                label=labels[group_val])
        kmf_objects[group_val] = kmf

        # Plot survival function with enhanced styling
        times = kmf.survival_function_.index
        survival = kmf.survival_function_.values.flatten()

        # Get confidence intervals
        ci_lower = kmf.confidence_interval_survival_function_.iloc[:, 0].values
        ci_upper = kmf.confidence_interval_survival_function_.iloc[:, 1].values

        # Fill confidence interval with gradient effect
        ax.fill_between(times, ci_lower, ci_upper,
                        color=light_colors[group_val], alpha=0.4)

        # Main survival line
        ax.step(times, survival, where='post', color=colors[group_val],
                linewidth=2.5, label=labels[group_val])

        # Store results
        results[group_val] = {
            'median_survival': kmf.median_survival_time_,
            'mean_survival': subset[time_col].mean(),
            'n': len(subset)
        }

        # Add median survival marker (when 50% of papers have been retracted)
        median = kmf.median_survival_time_
        if not np.isnan(median) and median <= max_time:
            ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
            # Add vertical line to x-axis for clarity
            ax.vlines(x=median, ymin=0, ymax=0.5, color=colors[group_val],
                      linestyle=':', alpha=0.5, linewidth=1)
            ax.plot([median], [0.5], 'o', color=colors[group_val],
                    markersize=10, markeredgecolor='white', markeredgewidth=2)

    # Log-rank test
    human = df[df[group_col] == 0]
    ai = df[df[group_col] == 1]
    lr_result = logrank_test(human[time_col], ai[time_col],
                             event_observed_A=human['event'],
                             event_observed_B=ai['event'])

    # Format plot
    ax.set_xlabel('Years Since Publication', fontsize=12, fontweight='bold')
    ax.set_ylabel('Survival Probability\n(Proportion Not Yet Retracted)', fontsize=12)
    ax.set_title('Time to Retraction: AI vs Human Papers\n(Kaplan-Meier Survival Curves)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(0, max_time)
    ax.set_ylim(0, 1.02)

    # Statistics box
    stats_text = (
        f"Human (n={results[0]['n']:,})\n"
        f"  Median: {results[0]['median_survival']:.1f} years\n\n"
        f"AI-Assisted (n={results[1]['n']:,})\n"
        f"  Median: {results[1]['median_survival']:.1f} years\n\n"
        f"Log-rank test\n"
        f"  p = {lr_result.p_value:.2e}"
    )

    bbox_props = dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='gray', alpha=0.9)
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=bbox_props, family='monospace')

    # Legend
    ax.legend(loc='lower left', frameon=True, fancybox=True,
              shadow=True, fontsize=11)

    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Add annotation explaining median markers
    ax.annotate('● = Median survival\n(50% retracted)',
                xy=(0.03, 0.52), xycoords='axes fraction',
                fontsize=9, color='gray', style='italic')

    # Add interpretation
    if results[1]['median_survival'] > results[0]['median_survival']:
        interpretation = "AI papers survive longer → Harder to detect"
        interp_color = COHORT_COLORS['ai']
    else:
        interpretation = "AI papers detected faster → Easier to catch"
        interp_color = COHORT_COLORS['human']

    ax.text(0.5, -0.12, interpretation, transform=ax.transAxes,
            fontsize=11, ha='center', style='italic', color=interp_color)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# TEMPORAL TRENDS ENHANCED
# =============================================================================

def create_temporal_trends_figure(
    df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create publication-quality temporal trends figure.

    Shows AI paper prevalence and detection speed over time.
    """
    setup_style()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Ensure pub_year exists and is valid
    if 'pub_year' not in df.columns:
        return None

    df = df.copy()
    df = df[df['pub_year'] >= 2000]
    df = df[df['pub_year'] <= df['pub_year'].max()]

    yearly = df.groupby('pub_year').agg({
        'is_ai': ['sum', 'count', 'mean'],
        'GIGO_Years': 'mean'
    }).reset_index()
    yearly.columns = ['year', 'ai_count', 'total', 'ai_rate', 'avg_gigo']

    # Panel A: AI Paper Count
    ax1 = axes[0, 0]
    bars = ax1.bar(yearly['year'], yearly['ai_count'],
                   color=COHORT_COLORS['ai'], alpha=0.8, edgecolor='white')
    ax1.set_xlabel('Publication Year')
    ax1.set_ylabel('Number of AI-Assisted Retractions')
    ax1.set_title('A. Rise of AI-Assisted Retractions', fontweight='bold')

    # Add trend line
    z = np.polyfit(yearly['year'], yearly['ai_count'], 2)
    p = np.poly1d(z)
    x_smooth = np.linspace(yearly['year'].min(), yearly['year'].max(), 100)
    ax1.plot(x_smooth, p(x_smooth), '--', color=COHORT_COLORS['ai'],
             linewidth=2, alpha=0.8, label='Trend')

    # Panel B: AI Rate
    ax2 = axes[0, 1]
    ax2.fill_between(yearly['year'], 0, yearly['ai_rate'] * 100,
                     color=COHORT_COLORS['ai_light'], alpha=0.5)
    ax2.plot(yearly['year'], yearly['ai_rate'] * 100,
             color=COHORT_COLORS['ai'], linewidth=2.5, marker='o', markersize=5)
    ax2.set_xlabel('Publication Year')
    ax2.set_ylabel('AI-Assisted Papers (%)')
    ax2.set_title('B. AI Prevalence in Retractions', fontweight='bold')
    ax2.set_ylim(0, None)

    # Panel C: Detection Speed Comparison
    ax3 = axes[1, 0]

    # Calculate yearly detection speed by cohort
    yearly_gigo = df.groupby(['pub_year', 'is_ai'])['GIGO_Years'].mean().unstack()

    if 0 in yearly_gigo.columns and 1 in yearly_gigo.columns:
        ax3.plot(yearly_gigo.index, yearly_gigo[0],
                 color=COHORT_COLORS['human'], linewidth=2.5,
                 marker='s', markersize=6, label='Human')
        ax3.plot(yearly_gigo.index, yearly_gigo[1],
                 color=COHORT_COLORS['ai'], linewidth=2.5,
                 marker='o', markersize=6, label='AI-Assisted')
        ax3.fill_between(yearly_gigo.index, yearly_gigo[0], yearly_gigo[1],
                         where=yearly_gigo[1] > yearly_gigo[0],
                         color=COHORT_COLORS['ai_light'], alpha=0.3,
                         label='AI harder to detect')

    ax3.set_xlabel('Publication Year')
    ax3.set_ylabel('Mean Detection Time (Years)')
    ax3.set_title('C. Detection Speed Over Time', fontweight='bold')
    ax3.legend(loc='upper right', frameon=True)

    # Panel D: Detection Ratio
    ax4 = axes[1, 1]

    if 0 in yearly_gigo.columns and 1 in yearly_gigo.columns:
        ratio = yearly_gigo[1] / yearly_gigo[0]
        ratio = ratio.dropna()

        colors = [COHORT_COLORS['ai'] if r > 1 else COHORT_COLORS['human'] for r in ratio]
        ax4.bar(ratio.index, ratio.values, color=colors, alpha=0.8, edgecolor='white')
        ax4.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5)

        # Add smoothed trend
        if len(ratio) > 3:
            z = np.polyfit(ratio.index, ratio.values, 2)
            p = np.poly1d(z)
            x_smooth = np.linspace(ratio.index.min(), ratio.index.max(), 100)
            ax4.plot(x_smooth, p(x_smooth), '--', color='purple',
                     linewidth=2, label='Trend')

    ax4.set_xlabel('Publication Year')
    ax4.set_ylabel('Detection Ratio (AI / Human)')
    ax4.set_title('D. Relative Detection Difficulty', fontweight='bold')
    ax4.text(0.5, -0.15, '>1 = AI harder to detect', transform=ax4.transAxes,
             ha='center', fontsize=10, style='italic')

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    plt.suptitle('Temporal Trends in AI-Assisted Paper Retractions',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# RETRACTION REASONS ENHANCED
# =============================================================================

def create_retraction_reasons_figure(
    df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create publication-quality retraction reasons comparison.
    """
    setup_style()

    if 'Reason' not in df.columns:
        return None

    df = df.copy()
    df['Reason'] = df['Reason'].fillna('Unknown')

    # Define reason categories
    reason_map = {
        'Fabrication/Falsification': ['fabricat', 'falsif', 'fake', 'manipulat'],
        'Plagiarism': ['plagiar', 'duplicate', 'overlap'],
        'Paper Mill/AI': ['paper mill', 'tortured', 'generated', 'ChatGPT', 'AI', 'LLM'],
        'Data Issues': ['data', 'error', 'mistake', 'unreliable'],
        'Image Manipulation': ['image', 'figure', 'duplication'],
        'Authorship Issues': ['author', 'consent', 'permission'],
        'Peer Review Fraud': ['peer review', 'review process'],
        'Ethical Violations': ['ethic', 'IRB', 'consent'],
        'Publisher Error': ['publisher', 'administrative'],
        'Other': []
    }

    def categorize_reason(reason_text):
        reason_lower = str(reason_text).lower()
        for category, keywords in reason_map.items():
            if any(kw in reason_lower for kw in keywords):
                return category
        return 'Other'

    df['reason_category'] = df['Reason'].apply(categorize_reason)

    # Cross-tabulation
    crosstab = pd.crosstab(
        df['reason_category'],
        df['is_ai'].map({0: 'Human', 1: 'AI'}),
        normalize='columns'
    ) * 100

    # Sort by difference
    crosstab['diff'] = crosstab['AI'] - crosstab['Human']
    crosstab = crosstab.sort_values('diff')

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Panel A: Side-by-side comparison
    ax1 = axes[0]
    y_pos = np.arange(len(crosstab))
    height = 0.35

    bars1 = ax1.barh(y_pos - height/2, crosstab['Human'], height,
                     color=COHORT_COLORS['human'], label='Human', alpha=0.9)
    bars2 = ax1.barh(y_pos + height/2, crosstab['AI'], height,
                     color=COHORT_COLORS['ai'], label='AI-Assisted', alpha=0.9)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(crosstab.index)
    ax1.set_xlabel('Percentage of Cohort (%)', fontweight='bold')
    ax1.set_title('A. Retraction Reasons by Cohort', fontweight='bold')
    ax1.legend(loc='lower right', frameon=True)
    ax1.set_xlim(0, crosstab[['Human', 'AI']].max().max() * 1.1)

    # Panel B: Difference plot (lollipop style)
    ax2 = axes[1]

    diff = crosstab['diff']
    colors = [COHORT_COLORS['ai'] if d > 0 else COHORT_COLORS['human'] for d in diff]

    # Lollipop chart
    ax2.hlines(y=y_pos, xmin=0, xmax=diff.values, colors=colors, linewidth=2)
    ax2.scatter(diff.values, y_pos, c=colors, s=100, zorder=3, edgecolors='white', linewidth=2)

    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(crosstab.index)
    ax2.set_xlabel('Difference (AI % − Human %)', fontweight='bold')
    ax2.set_title('B. Over/Under-representation in AI Cohort', fontweight='bold')

    # Add color legend
    ax2.text(0.98, 0.02, '← More in Human    More in AI →',
             transform=ax2.transAxes, ha='right', va='bottom',
             fontsize=9, style='italic')

    for ax in axes:
        ax.grid(True, alpha=0.3, axis='x')

    plt.suptitle('Retraction Reasons: AI vs Human Papers',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# SUBJECT DISTRIBUTION ENHANCED
# =============================================================================

def create_subject_distribution_figure(
    df: pd.DataFrame,
    top_n: int = 12,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create publication-quality subject distribution comparison.
    """
    setup_style()

    if 'Subject' not in df.columns:
        return None

    df = df.copy()
    df['Subject'] = df['Subject'].fillna('Unknown')

    # Get top subjects
    top_subjects = df['Subject'].value_counts().head(top_n).index.tolist()
    df_top = df[df['Subject'].isin(top_subjects)]

    # Cross-tabulation
    crosstab = pd.crosstab(
        df_top['Subject'],
        df_top['is_ai'].map({0: 'Human', 1: 'AI'}),
        normalize='columns'
    ) * 100

    # Calculate AI concentration
    ai_concentration = df_top.groupby('Subject')['is_ai'].mean() * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    # Panel A: Distribution comparison
    ax1 = axes[0]

    # Sort by total
    crosstab = crosstab.sort_values('Human', ascending=True)
    y_pos = np.arange(len(crosstab))
    height = 0.35

    ax1.barh(y_pos - height/2, crosstab['Human'], height,
             color=COHORT_COLORS['human'], label='Human', alpha=0.9)
    ax1.barh(y_pos + height/2, crosstab['AI'], height,
             color=COHORT_COLORS['ai'], label='AI-Assisted', alpha=0.9)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(crosstab.index)
    ax1.set_xlabel('Percentage of Cohort (%)', fontweight='bold')
    ax1.set_title('A. Subject Distribution by Cohort', fontweight='bold')
    ax1.legend(loc='lower right', frameon=True)

    # Panel B: AI Concentration (sorted)
    ax2 = axes[1]

    ai_conc_sorted = ai_concentration.reindex(crosstab.index)

    # Color by concentration
    colors = []
    for val in ai_conc_sorted.values:
        if val > 40:
            colors.append(COHORT_COLORS['ai'])
        elif val > 25:
            colors.append(COHORT_COLORS['ai_light'])
        elif val < 20:
            colors.append(COHORT_COLORS['human'])
        else:
            colors.append('gray')

    bars = ax2.barh(y_pos, ai_conc_sorted.values, color=colors, alpha=0.9, edgecolor='white')
    ax2.axvline(x=df['is_ai'].mean() * 100, color='black', linestyle='--',
                linewidth=2, label=f'Overall: {df["is_ai"].mean()*100:.1f}%')

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(ai_conc_sorted.index)
    ax2.set_xlabel('AI-Assisted Papers (%)', fontweight='bold')
    ax2.set_title('B. AI Concentration by Subject', fontweight='bold')
    ax2.legend(loc='lower right', frameon=True)

    for ax in axes:
        ax.grid(True, alpha=0.3, axis='x')

    plt.suptitle('Subject Area Analysis: AI vs Human Retractions',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# CITATIONS ANALYSIS ENHANCED
# =============================================================================

def create_citations_figure(
    df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create publication-quality citations analysis figure.
    """
    setup_style()

    citation_col = None
    for col in ['citations', 'citation_count', 'Cited by']:
        if col in df.columns:
            citation_col = col
            break

    if citation_col is None:
        return None

    df = df.copy()
    df['citations'] = pd.to_numeric(df[citation_col], errors='coerce')
    df = df.dropna(subset=['citations'])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Citation distribution (KDE)
    ax1 = axes[0, 0]

    human = df[df['is_ai'] == 0]['citations']
    ai = df[df['is_ai'] == 1]['citations']

    # Log transform for visualization
    human_log = np.log10(human + 1)
    ai_log = np.log10(ai + 1)

    # KDE plots
    from scipy.stats import gaussian_kde

    x_range = np.linspace(0, max(human_log.max(), ai_log.max()), 200)

    kde_human = gaussian_kde(human_log, bw_method='scott')
    kde_ai = gaussian_kde(ai_log, bw_method='scott')

    ax1.fill_between(x_range, kde_human(x_range), alpha=0.5,
                     color=COHORT_COLORS['human'], label='Human')
    ax1.fill_between(x_range, kde_ai(x_range), alpha=0.5,
                     color=COHORT_COLORS['ai'], label='AI-Assisted')
    ax1.plot(x_range, kde_human(x_range), color=COHORT_COLORS['human'], linewidth=2)
    ax1.plot(x_range, kde_ai(x_range), color=COHORT_COLORS['ai'], linewidth=2)

    ax1.set_xlabel('Citations (log₁₀ scale)', fontweight='bold')
    ax1.set_ylabel('Density')
    ax1.set_title('A. Citation Distribution', fontweight='bold')
    ax1.legend(loc='upper right', frameon=True)

    # Panel B: Box comparison
    ax2 = axes[0, 1]

    data_to_plot = [human, ai]
    bp = ax2.boxplot(data_to_plot, labels=['Human', 'AI-Assisted'],
                     patch_artist=True, widths=0.6)

    colors_box = [COHORT_COLORS['human'], COHORT_COLORS['ai']]
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_ylabel('Citations', fontweight='bold')
    ax2.set_title('B. Citation Comparison', fontweight='bold')
    ax2.set_yscale('log')

    # Add mean markers
    means = [human.mean(), ai.mean()]
    ax2.scatter([1, 2], means, marker='D', color='black', s=50, zorder=5, label='Mean')
    ax2.legend(loc='upper right')

    # Panel C: Citations vs GIGO
    ax3 = axes[1, 0]

    if 'GIGO_Years' in df.columns:
        # Hexbin for density
        hb = ax3.hexbin(df['citations'], df['GIGO_Years'],
                        gridsize=30, cmap='YlOrRd', mincnt=1)
        plt.colorbar(hb, ax=ax3, label='Count')

        ax3.set_xlabel('Citations', fontweight='bold')
        ax3.set_ylabel('Time to Retraction (Years)', fontweight='bold')
        ax3.set_title('C. Citations vs Detection Time', fontweight='bold')
        ax3.set_xscale('log')

    # Panel D: Citation quantiles
    ax4 = axes[1, 1]

    quantiles = [0.25, 0.5, 0.75, 0.9, 0.95]
    human_q = [human.quantile(q) for q in quantiles]
    ai_q = [ai.quantile(q) for q in quantiles]

    x_pos = np.arange(len(quantiles))
    width = 0.35

    ax4.bar(x_pos - width/2, human_q, width,
            color=COHORT_COLORS['human'], label='Human', alpha=0.9)
    ax4.bar(x_pos + width/2, ai_q, width,
            color=COHORT_COLORS['ai'], label='AI-Assisted', alpha=0.9)

    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'{int(q*100)}th' for q in quantiles])
    ax4.set_xlabel('Percentile', fontweight='bold')
    ax4.set_ylabel('Citations', fontweight='bold')
    ax4.set_title('D. Citation Quantiles', fontweight='bold')
    ax4.legend(loc='upper left', frameon=True)
    ax4.set_yscale('log')

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    plt.suptitle('Citation Analysis: AI vs Human Retractions',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# MATCHED COHORT ENHANCED
# =============================================================================

def create_matched_cohort_figure(
    df: pd.DataFrame,
    time_col: str = 'GIGO_Years',
    max_time: float = 15,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create publication-quality matched cohort analysis figure.
    """
    if not HAS_LIFELINES:
        raise ImportError("lifelines required")

    setup_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    df = df.copy()
    df['event'] = 1
    df[time_col] = df[time_col].clip(upper=max_time)

    # Left panel: Matched survival curves by year
    ax1 = axes[0]

    years = sorted(df['pub_year'].dropna().unique())
    recent_years = [y for y in years if y >= 2015][:5]

    cmap = plt.cm.viridis
    colors_year = {y: cmap(i / len(recent_years)) for i, y in enumerate(recent_years)}

    for year in recent_years:
        year_df = df[df['pub_year'] == year]

        for cohort, style in [(0, '--'), (1, '-')]:
            subset = year_df[year_df['is_ai'] == cohort]
            if len(subset) < 10:
                continue

            kmf = KaplanMeierFitter()
            kmf.fit(subset[time_col], event_observed=subset['event'])

            label = f'{int(year)}' if cohort == 1 else None
            ax1.step(kmf.survival_function_.index,
                     kmf.survival_function_.values.flatten(),
                     where='post', color=colors_year[year],
                     linestyle=style, linewidth=2, label=label, alpha=0.8)

    ax1.set_xlabel('Years Since Publication', fontweight='bold')
    ax1.set_ylabel('Survival Probability')
    ax1.set_title('A. Survival by Publication Year\n(Solid=AI, Dashed=Human)', fontweight='bold')
    ax1.legend(title='Year', loc='lower left', frameon=True)
    ax1.set_xlim(0, max_time)
    ax1.set_ylim(0, 1)

    # Right panel: Hazard ratio by year
    ax2 = axes[1]

    hazard_ratios = []
    for year in years:
        year_df = df[df['pub_year'] == year]
        human_gigo = year_df[year_df['is_ai'] == 0][time_col].mean()
        ai_gigo = year_df[year_df['is_ai'] == 1][time_col].mean()

        if human_gigo > 0 and ai_gigo > 0:
            hazard_ratios.append({
                'year': year,
                'ratio': ai_gigo / human_gigo,
                'n_ai': (year_df['is_ai'] == 1).sum(),
                'n_human': (year_df['is_ai'] == 0).sum()
            })

    hr_df = pd.DataFrame(hazard_ratios)
    hr_df = hr_df[hr_df['n_ai'] >= 5]  # Require minimum sample

    colors = [COHORT_COLORS['ai'] if r > 1 else COHORT_COLORS['human']
              for r in hr_df['ratio']]
    ax2.bar(hr_df['year'], hr_df['ratio'], color=colors, alpha=0.8, edgecolor='white')
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=2)

    ax2.set_xlabel('Publication Year', fontweight='bold')
    ax2.set_ylabel('Detection Time Ratio (AI / Human)')
    ax2.set_title('B. Relative Detection Difficulty by Year', fontweight='bold')
    ax2.text(0.5, -0.12, '>1 = AI takes longer to detect',
             transform=ax2.transAxes, ha='center', fontsize=10, style='italic')

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.suptitle('Matched Cohort Analysis: AI vs Human by Publication Year',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# SINGLE PANEL: SURVIVAL BY YEAR
# =============================================================================

def create_survival_by_year_figure(
    df: pd.DataFrame,
    time_col: str = 'GIGO_Years',
    max_time: float = 10,
    start_year: int = 2015,
    end_year: int = 2023,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create single-panel survival curves by publication year.

    Shows AI (solid) vs Human (dashed) survival for each year.

    Parameters
    ----------
    df : DataFrame
    time_col : str
        Time to event column
    max_time : float
        Maximum time for x-axis (default: 10 years)
    start_year : int
        First year to include (default: 2015)
    end_year : int
        Last year to include (default: 2023)
    save_path : str, optional
    """
    if not HAS_LIFELINES:
        raise ImportError("lifelines required")

    setup_style()

    fig, ax = plt.subplots(figsize=(12, 8))

    df = df.copy()
    df['event'] = 1
    df[time_col] = df[time_col].clip(upper=max_time)

    # Get years in range
    years = sorted(df['pub_year'].dropna().unique())
    years_to_plot = [y for y in years if start_year <= y <= end_year]

    if not years_to_plot:
        ax.text(0.5, 0.5, 'No data in year range', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        return fig

    # Color map for years
    n_years = len(years_to_plot)
    cmap = plt.cm.viridis
    colors_year = {y: cmap(i / max(n_years - 1, 1)) for i, y in enumerate(years_to_plot)}

    for year in years_to_plot:
        year_df = df[df['pub_year'] == year]

        for cohort, style, lw in [(0, '--', 1.5), (1, '-', 2.5)]:
            subset = year_df[year_df['is_ai'] == cohort]
            if len(subset) < 5:
                continue

            kmf = KaplanMeierFitter()
            kmf.fit(subset[time_col], event_observed=subset['event'])

            # Only label AI lines (solid) to avoid cluttering legend
            label = f'{int(year)}' if cohort == 1 else None
            ax.step(kmf.survival_function_.index,
                    kmf.survival_function_.values.flatten(),
                    where='post', color=colors_year[year],
                    linestyle=style, linewidth=lw, label=label, alpha=0.85)

    ax.set_xlabel('Years Since Publication', fontsize=13, fontweight='bold')
    ax.set_ylabel('Survival Probability\n(Proportion Not Yet Retracted)', fontsize=12)
    ax.set_title('Time to Retraction by Publication Year\n(Solid = AI-Assisted, Dashed = Human)',
                 fontsize=14, fontweight='bold', pad=15)

    ax.set_xlim(0, max_time)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Legend for years
    ax.legend(title='Publication Year', loc='lower left', fontsize=10,
              frameon=True, fancybox=True, ncol=2)

    # Add explanatory note
    ax.text(0.98, 0.98, 'Solid lines = AI-assisted papers\nDashed lines = Human papers',
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# OPTION A: STRATIFIED POOLED SURVIVAL CURVES
# =============================================================================

def create_stratified_survival_figure(
    df: pd.DataFrame,
    time_col: str = 'GIGO_Years',
    max_time: float = 15,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Option A: Stratified pooled survival curves.

    Shows one curve per cohort (AI vs Human), pooled across all years,
    but the visual presentation avoids the crossing issue by using
    year-stratified Kaplan-Meier estimation.

    This averages the survival experience across years, giving each year
    equal weight regardless of sample size.
    """
    if not HAS_LIFELINES:
        raise ImportError("lifelines required")

    setup_style()

    fig, ax = plt.subplots(figsize=(11, 8))

    df = df.copy()
    df['event'] = 1
    df[time_col] = df[time_col].clip(upper=max_time)

    # Get years with sufficient data in both cohorts
    year_counts = df.groupby(['pub_year', 'is_ai']).size().unstack(fill_value=0)
    valid_years = year_counts[(year_counts[0] >= 10) & (year_counts[1] >= 10)].index.tolist()

    if len(valid_years) < 3:
        ax.text(0.5, 0.5, 'Insufficient data for stratified analysis',
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        return fig

    df_valid = df[df['pub_year'].isin(valid_years)]

    # Create time points for evaluation
    time_points = np.linspace(0, max_time, 100)

    # For each cohort, average survival across years
    for cohort, label, color, light_color in [
        (0, 'Human Papers', COHORT_COLORS['human'], COHORT_COLORS['human_light']),
        (1, 'AI-Assisted Papers', COHORT_COLORS['ai'], COHORT_COLORS['ai_light'])
    ]:
        survival_curves = []

        for year in valid_years:
            subset = df_valid[(df_valid['pub_year'] == year) & (df_valid['is_ai'] == cohort)]
            if len(subset) < 5:
                continue

            kmf = KaplanMeierFitter()
            kmf.fit(subset[time_col], event_observed=subset['event'])

            # Interpolate survival at standard time points
            surv_interp = np.interp(time_points,
                                     kmf.survival_function_.index,
                                     kmf.survival_function_.values.flatten(),
                                     right=0)
            survival_curves.append(surv_interp)

        if not survival_curves:
            continue

        # Average across years (equal weighting)
        survival_matrix = np.array(survival_curves)
        mean_survival = np.mean(survival_matrix, axis=0)
        std_survival = np.std(survival_matrix, axis=0)

        # Plot mean with CI band (±1 SE across years)
        se_survival = std_survival / np.sqrt(len(survival_curves))
        ci_lower = np.clip(mean_survival - 1.96 * se_survival, 0, 1)
        ci_upper = np.clip(mean_survival + 1.96 * se_survival, 0, 1)

        ax.fill_between(time_points, ci_lower, ci_upper,
                        color=light_color, alpha=0.4)
        ax.plot(time_points, mean_survival, color=color, linewidth=2.5,
                label=f'{label} (n={len(df_valid[df_valid["is_ai"]==cohort]):,})')

        # Add median marker
        median_idx = np.searchsorted(-mean_survival, -0.5)
        if median_idx < len(time_points):
            median_time = time_points[median_idx]
            ax.plot([median_time], [0.5], 'o', color=color, markersize=10,
                    markeredgecolor='white', markeredgewidth=2)
            ax.vlines(x=median_time, ymin=0, ymax=0.5, color=color,
                      linestyle=':', alpha=0.5)

    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)

    ax.set_xlabel('Years Since Publication', fontsize=12, fontweight='bold')
    ax.set_ylabel('Survival Probability\n(Proportion Not Yet Retracted)', fontsize=12)
    ax.set_title('Time to Retraction: AI vs Human Papers\n(Year-Stratified Average, Equal Year Weights)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(0, max_time)
    ax.set_ylim(0, 1.02)

    ax.legend(loc='lower left', frameon=True, fancybox=True, fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add note about methodology
    ax.text(0.98, 0.98, f'Averaged across {len(valid_years)} years\n({min(valid_years)}-{max(valid_years)})',
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# OPTION B: MEDIAN SURVIVAL / HAZARD RATIO BY YEAR
# =============================================================================

def create_median_survival_by_year_figure(
    df: pd.DataFrame,
    time_col: str = 'GIGO_Years',
    start_year: int = 2015,
    end_year: int = 2023,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Option B: Bar chart showing median survival time by year and cohort.

    Clean, easy-to-read visualization that avoids crossing curves entirely.
    Shows the key finding (detection time difference) directly.
    """
    if not HAS_LIFELINES:
        raise ImportError("lifelines required")

    setup_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    df = df.copy()
    df['event'] = 1

    # Filter to year range
    df = df[(df['pub_year'] >= start_year) & (df['pub_year'] <= end_year)]

    years = sorted(df['pub_year'].dropna().unique())

    results = []
    for year in years:
        year_df = df[df['pub_year'] == year]

        human_data = year_df[year_df['is_ai'] == 0][time_col]
        ai_data = year_df[year_df['is_ai'] == 1][time_col]

        if len(human_data) >= 5 and len(ai_data) >= 5:
            # Fit KM for median survival
            kmf_human = KaplanMeierFitter()
            kmf_human.fit(human_data, event_observed=np.ones(len(human_data)))

            kmf_ai = KaplanMeierFitter()
            kmf_ai.fit(ai_data, event_observed=np.ones(len(ai_data)))

            results.append({
                'year': int(year),
                'human_median': kmf_human.median_survival_time_,
                'ai_median': kmf_ai.median_survival_time_,
                'human_mean': human_data.mean(),
                'ai_mean': ai_data.mean(),
                'n_human': len(human_data),
                'n_ai': len(ai_data),
                'ratio': ai_data.mean() / human_data.mean() if human_data.mean() > 0 else np.nan
            })

    if not results:
        axes[0].text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                     transform=axes[0].transAxes, fontsize=14)
        return fig

    results_df = pd.DataFrame(results)

    # Panel A: Grouped bar chart of median survival
    ax1 = axes[0]
    x = np.arange(len(results_df))
    width = 0.35

    bars1 = ax1.bar(x - width/2, results_df['human_median'], width,
                    label='Human', color=COHORT_COLORS['human'], alpha=0.9,
                    edgecolor='white', linewidth=1)
    bars2 = ax1.bar(x + width/2, results_df['ai_median'], width,
                    label='AI-Assisted', color=COHORT_COLORS['ai'], alpha=0.9,
                    edgecolor='white', linewidth=1)

    ax1.set_xlabel('Publication Year', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Median Time to Retraction (Years)', fontsize=12)
    ax1.set_title('A. Median Detection Time by Year', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(results_df['year'].astype(int))
    ax1.legend(loc='upper right', frameon=True)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax1.annotate(f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

    # Panel B: Detection ratio (AI/Human) - lollipop chart
    ax2 = axes[1]

    colors = [COHORT_COLORS['ai'] if r > 1 else COHORT_COLORS['human']
              for r in results_df['ratio']]

    # Lollipop chart
    ax2.hlines(y=x, xmin=1, xmax=results_df['ratio'], colors=colors, linewidth=3)
    ax2.scatter(results_df['ratio'], x, c=colors, s=150, zorder=3,
                edgecolors='white', linewidth=2)

    ax2.axvline(x=1.0, color='black', linestyle='--', linewidth=2, label='Equal detection time')

    ax2.set_yticks(x)
    ax2.set_yticklabels(results_df['year'].astype(int))
    ax2.set_xlabel('Detection Time Ratio (AI / Human)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Publication Year', fontsize=12)
    ax2.set_title('B. Relative Detection Difficulty', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    # Add interpretation
    ax2.text(0.98, 0.02, '←Faster    Slower→\n   (AI vs Human)',
             transform=ax2.transAxes, ha='right', va='bottom', fontsize=9, style='italic')

    # Add value labels
    for i, (ratio, year) in enumerate(zip(results_df['ratio'], results_df['year'])):
        if not np.isnan(ratio):
            offset = 0.05 if ratio > 1 else -0.05
            ha = 'left' if ratio > 1 else 'right'
            ax2.annotate(f'{ratio:.2f}', xy=(ratio, i), xytext=(ratio + offset, i),
                        ha=ha, va='center', fontsize=9, fontweight='bold')

    plt.suptitle('Detection Time Analysis by Publication Year',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# OPTION C: RESTRICTED TIME WINDOW ANALYSIS
# =============================================================================

def create_restricted_window_survival_figure(
    df: pd.DataFrame,
    time_col: str = 'GIGO_Years',
    pub_year_start: int = 2018,
    pub_year_end: int = 2021,
    min_followup: float = 2.0,
    max_time: float = 8,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Option C: Restricted time window analysis.

    Only includes papers published in a narrow window (e.g., 2018-2021)
    that have had sufficient follow-up time. This ensures:
    1. Comparable observation periods for both cohorts
    2. Avoids right-censoring issues from very recent papers
    3. Cleaner comparison without cohort mixing effects

    Parameters
    ----------
    pub_year_start, pub_year_end : int
        Publication year range to include
    min_followup : float
        Minimum years of follow-up required (filters recent papers)
    max_time : float
        Maximum time for visualization
    """
    if not HAS_LIFELINES:
        raise ImportError("lifelines required")

    setup_style()

    fig, ax = plt.subplots(figsize=(11, 8))

    df = df.copy()
    df['event'] = 1

    # Filter to publication year range
    df = df[(df['pub_year'] >= pub_year_start) & (df['pub_year'] <= pub_year_end)]

    # Estimate current year for follow-up calculation
    # (papers need to have been "at risk" for min_followup years)
    current_year = 2024  # Approximate
    df['max_possible_followup'] = current_year - df['pub_year']

    # Only include papers that COULD have been observed for min_followup
    df = df[df['max_possible_followup'] >= min_followup]

    # Cap observation time
    df[time_col] = df[time_col].clip(upper=max_time)

    n_human = (df['is_ai'] == 0).sum()
    n_ai = (df['is_ai'] == 1).sum()

    if n_human < 20 or n_ai < 20:
        ax.text(0.5, 0.5, f'Insufficient data in restricted window\n(Human: {n_human}, AI: {n_ai})',
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        return fig

    results = {}

    for cohort, label, color, light_color in [
        (0, 'Human Papers', COHORT_COLORS['human'], COHORT_COLORS['human_light']),
        (1, 'AI-Assisted Papers', COHORT_COLORS['ai'], COHORT_COLORS['ai_light'])
    ]:
        subset = df[df['is_ai'] == cohort]

        kmf = KaplanMeierFitter()
        kmf.fit(subset[time_col], event_observed=subset['event'], label=label)

        # Get survival function and CI
        times = kmf.survival_function_.index
        survival = kmf.survival_function_.values.flatten()
        ci_lower = kmf.confidence_interval_survival_function_.iloc[:, 0].values
        ci_upper = kmf.confidence_interval_survival_function_.iloc[:, 1].values

        # Plot
        ax.fill_between(times, ci_lower, ci_upper, color=light_color, alpha=0.4)
        ax.step(times, survival, where='post', color=color, linewidth=2.5, label=label)

        # Store results
        results[cohort] = {
            'median': kmf.median_survival_time_,
            'mean': subset[time_col].mean(),
            'n': len(subset)
        }

        # Median marker
        median = kmf.median_survival_time_
        if not np.isnan(median) and median <= max_time:
            ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
            ax.vlines(x=median, ymin=0, ymax=0.5, color=color, linestyle=':', alpha=0.5)
            ax.plot([median], [0.5], 'o', color=color, markersize=10,
                    markeredgecolor='white', markeredgewidth=2)

    # Log-rank test
    human = df[df['is_ai'] == 0]
    ai = df[df['is_ai'] == 1]
    lr_result = logrank_test(human[time_col], ai[time_col],
                             event_observed_A=human['event'],
                             event_observed_B=ai['event'])

    # Statistics box
    stats_text = (
        f"Restricted Analysis\n"
        f"Published: {pub_year_start}-{pub_year_end}\n"
        f"Min follow-up: {min_followup} years\n\n"
        f"Human (n={results[0]['n']:,})\n"
        f"  Median: {results[0]['median']:.2f} years\n\n"
        f"AI-Assisted (n={results[1]['n']:,})\n"
        f"  Median: {results[1]['median']:.2f} years\n\n"
        f"Log-rank p = {lr_result.p_value:.2e}"
    )

    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='gray', alpha=0.9),
            family='monospace')

    ax.set_xlabel('Years Since Publication', fontsize=12, fontweight='bold')
    ax.set_ylabel('Survival Probability\n(Proportion Not Yet Retracted)', fontsize=12)
    ax.set_title(f'Time to Retraction: Restricted Cohort Analysis\n(Papers Published {pub_year_start}-{pub_year_end}, ≥{min_followup} Years Follow-up)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(0, max_time)
    ax.set_ylim(0, 1.02)

    ax.legend(loc='lower left', frameon=True, fancybox=True, fontsize=11)
    ax.grid(True, alpha=0.3)

    # Interpretation
    diff = results[1]['median'] - results[0]['median']
    if diff > 0:
        interpretation = f"AI papers take {diff:.2f} years longer to detect"
        color = COHORT_COLORS['ai']
    else:
        interpretation = f"AI papers detected {-diff:.2f} years faster"
        color = COHORT_COLORS['human']

    ax.text(0.5, -0.1, interpretation, transform=ax.transAxes,
            fontsize=11, ha='center', style='italic', color=color)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# PRELIMINARY TRENDS: EARLY DETECTION IN RECENT COHORTS (2022-2024)
# =============================================================================

def create_preliminary_trends_figure(
    df: pd.DataFrame,
    time_col: str = 'GIGO_Years',
    max_time: float = 2.0,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Preliminary trends figure for recent cohorts (2022-2024).

    Shows early detection patterns (first 1-2 years) for recent publications,
    with explicit caveats about incomplete follow-up. This provides a
    preliminary signal about whether LLM-era papers are harder to detect.

    IMPORTANT: This figure must be interpreted with caution due to right-censoring.
    Papers from 2023-2024 have not had sufficient time to be fully detected.
    """
    if not HAS_LIFELINES:
        raise ImportError("lifelines required")

    setup_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    df = df.copy()
    df['event'] = 1
    df[time_col] = df[time_col].clip(upper=max_time)

    # Define cohort eras
    eras = {
        'Pre-LLM (2018-2021)': (2018, 2021),
        'Transition (2022)': (2022, 2022),
        'LLM Era (2023-2024)': (2023, 2024)
    }

    era_colors = {
        'Pre-LLM (2018-2021)': '#2E86AB',
        'Transition (2022)': '#F6AE2D',
        'LLM Era (2023-2024)': '#A23B72'
    }

    # Panel A: Early survival curves by era (AI papers only)
    ax1 = axes[0]

    time_points = np.linspace(0, max_time, 50)
    era_results = {}

    for era_name, (start_yr, end_yr) in eras.items():
        era_df = df[(df['pub_year'] >= start_yr) & (df['pub_year'] <= end_yr)]
        ai_df = era_df[era_df['is_ai'] == 1]

        if len(ai_df) < 20:
            continue

        kmf = KaplanMeierFitter()
        kmf.fit(ai_df[time_col], event_observed=ai_df['event'])

        # Interpolate for plotting
        surv_interp = np.interp(time_points,
                                 kmf.survival_function_.index,
                                 kmf.survival_function_.values.flatten(),
                                 right=kmf.survival_function_.values[-1])

        ax1.plot(time_points, surv_interp, color=era_colors[era_name],
                 linewidth=2.5, label=f'{era_name} (n={len(ai_df):,})')

        # Store 1-year survival rate
        idx_1yr = np.searchsorted(time_points, 1.0)
        era_results[era_name] = {
            'n': len(ai_df),
            'survival_1yr': surv_interp[idx_1yr] if idx_1yr < len(surv_interp) else surv_interp[-1],
            'median': kmf.median_survival_time_
        }

    ax1.set_xlabel('Years Since Publication', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Survival Probability\n(Proportion Not Yet Retracted)', fontsize=12)
    ax1.set_title('A. Early Detection: AI-Assisted Papers by Era\n(First 2 Years Only)', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, max_time)
    ax1.set_ylim(0, 1.02)
    ax1.legend(loc='lower left', frameon=True, fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Add caution note
    ax1.text(0.98, 0.98, 'PRELIMINARY\nIncomplete follow-up\nfor 2023-2024',
             transform=ax1.transAxes, ha='right', va='top', fontsize=9,
             color='red', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEEEE', edgecolor='red', alpha=0.9))

    # Panel B: 1-Year survival rate comparison (bar chart)
    ax2 = axes[1]

    if era_results:
        era_names = list(era_results.keys())
        survival_rates = [era_results[e]['survival_1yr'] * 100 for e in era_names]
        colors = [era_colors[e] for e in era_names]
        n_values = [era_results[e]['n'] for e in era_names]

        x = np.arange(len(era_names))
        bars = ax2.bar(x, survival_rates, color=colors, alpha=0.9, edgecolor='white', linewidth=2)

        ax2.set_xticks(x)
        ax2.set_xticklabels([f'{e}\n(n={n:,})' for e, n in zip(era_names, n_values)], fontsize=10)
        ax2.set_ylabel('1-Year Survival Rate (%)\n(Still Undetected After 1 Year)', fontsize=12)
        ax2.set_title('B. Early Detection Difficulty by Era\n(Higher = Harder to Detect)', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, rate in zip(bars, survival_rates):
            ax2.annotate(f'{rate:.1f}%',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Add trend arrow if increasing
        if len(survival_rates) >= 2 and survival_rates[-1] > survival_rates[0]:
            ax2.annotate('', xy=(len(survival_rates)-0.5, survival_rates[-1] + 5),
                        xytext=(0.5, survival_rates[0] + 5),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2))
            ax2.text(len(survival_rates)/2, max(survival_rates) + 10,
                    'Detection becoming harder?',
                    ha='center', fontsize=10, color='red', style='italic')

    # Add interpretation box
    interpretation = (
        "INTERPRETATION CAUTION:\n"
        "• 2023-2024 papers have <2 years follow-up\n"
        "• Higher survival may reflect incomplete observation\n"
        "• True detection difficulty requires 3+ years data\n"
        "• Trend direction is preliminary only"
    )
    ax2.text(0.02, 0.98, interpretation, transform=ax2.transAxes,
             fontsize=9, va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFFACD', edgecolor='orange', alpha=0.9))

    plt.suptitle('Preliminary Analysis: Detection Trends in the LLM Era',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# GENERATE ALL FIGURES
# =============================================================================

def generate_all_retraction_figures(
    df: pd.DataFrame,
    output_dir: str,
    verbose: bool = True
) -> Dict[str, str]:
    """
    Generate all enhanced retraction analysis figures.

    Parameters
    ----------
    df : DataFrame
        Retraction data with is_ai, GIGO_Years, etc.
    output_dir : str
        Directory for output figures
    verbose : bool
        Print progress

    Returns
    -------
    dict with paths to generated figures
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    figures = {}

    # 1. Kaplan-Meier
    if verbose:
        print("  1. Kaplan-Meier survival curves...")
    try:
        path = os.path.join(output_dir, 'fig_kaplan_meier_enhanced.png')
        create_kaplan_meier_figure(df, save_path=path)
        figures['kaplan_meier'] = path
    except Exception as e:
        print(f"     Failed: {e}")

    # 2. Temporal trends
    if verbose:
        print("  2. Temporal trends...")
    try:
        path = os.path.join(output_dir, 'fig_temporal_enhanced.png')
        create_temporal_trends_figure(df, save_path=path)
        figures['temporal'] = path
    except Exception as e:
        print(f"     Failed: {e}")

    # 3. Retraction reasons
    if verbose:
        print("  3. Retraction reasons...")
    try:
        path = os.path.join(output_dir, 'fig_reasons_enhanced.png')
        create_retraction_reasons_figure(df, save_path=path)
        figures['reasons'] = path
    except Exception as e:
        print(f"     Failed: {e}")

    # 4. Subject distribution
    if verbose:
        print("  4. Subject distribution...")
    try:
        path = os.path.join(output_dir, 'fig_subjects_enhanced.png')
        create_subject_distribution_figure(df, save_path=path)
        figures['subjects'] = path
    except Exception as e:
        print(f"     Failed: {e}")

    # 5. Citations (only if citation data available)
    if verbose:
        print("  5. Citations analysis...")
    try:
        path = os.path.join(output_dir, 'fig_citations_enhanced.png')
        fig = create_citations_figure(df, save_path=path)
        if fig is not None:
            figures['citations'] = path
        else:
            print("     Skipped: No citation column found in data")
    except Exception as e:
        print(f"     Failed: {e}")

    # 6. Matched cohort
    if verbose:
        print("  6. Matched cohort analysis...")
    try:
        path = os.path.join(output_dir, 'fig_matched_cohort_enhanced.png')
        create_matched_cohort_figure(df, save_path=path)
        figures['matched_cohort'] = path
    except Exception as e:
        print(f"     Failed: {e}")

    # 7. Survival by Year (single panel, 2015-2023)
    if verbose:
        print("  7. Survival by year (single panel)...")
    try:
        path = os.path.join(output_dir, 'fig_survival_by_year.png')
        create_survival_by_year_figure(df, max_time=10, start_year=2015, end_year=2023,
                                       save_path=path)
        figures['survival_by_year'] = path
    except Exception as e:
        print(f"     Failed: {e}")

    # 8. OPTION A: Stratified pooled survival curves
    if verbose:
        print("  8. Option A: Stratified pooled survival...")
    try:
        path = os.path.join(output_dir, 'fig_option_a_stratified_survival.png')
        create_stratified_survival_figure(df, max_time=15, save_path=path)
        figures['option_a_stratified'] = path
    except Exception as e:
        print(f"     Failed: {e}")

    # 9. OPTION B: Median survival by year bar chart
    if verbose:
        print("  9. Option B: Median survival by year...")
    try:
        path = os.path.join(output_dir, 'fig_option_b_median_by_year.png')
        create_median_survival_by_year_figure(df, start_year=2015, end_year=2023,
                                               save_path=path)
        figures['option_b_median_by_year'] = path
    except Exception as e:
        print(f"     Failed: {e}")

    # 10. OPTION C: Restricted time window analysis
    if verbose:
        print("  10. Option C: Restricted window survival...")
    try:
        path = os.path.join(output_dir, 'fig_option_c_restricted_window.png')
        create_restricted_window_survival_figure(df, pub_year_start=2018, pub_year_end=2021,
                                                  min_followup=2.0, max_time=8,
                                                  save_path=path)
        figures['option_c_restricted'] = path
    except Exception as e:
        print(f"     Failed: {e}")

    # 11. Preliminary trends (LLM era - 2022-2024)
    if verbose:
        print("  11. Preliminary trends (LLM era)...")
    try:
        path = os.path.join(output_dir, 'fig_preliminary_trends_llm_era.png')
        create_preliminary_trends_figure(df, max_time=2.0, save_path=path)
        figures['preliminary_trends'] = path
    except Exception as e:
        print(f"     Failed: {e}")

    if verbose:
        print(f"\n✓ Generated {len(figures)} enhanced figures in {output_dir}")

    return figures
