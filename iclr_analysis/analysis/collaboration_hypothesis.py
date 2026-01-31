"""
AI-Human Collaboration Hypothesis Test
=======================================

Research Question:
Does moderate AI assistance improve paper quality compared to:
1. Pure human-written papers (no AI)?
2. Heavily AI-generated papers?

Tests three hypotheses:
H1 (Augmentation): Light AI editing improves quality
    → Quality(light_AI) > Quality(human_only)

H2 (Degradation): Any AI use degrades quality
    → Quality(human_only) > Quality(light_AI) > Quality(heavy_AI)

H3 (Inverted-U): Moderate AI is optimal
    → Quality peaks at intermediate AI levels

Statistical Methods:
- Quadratic regression (test for inverted-U shape)
- Pairwise comparisons with FDR correction
- Dose-response analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional
from scipy.stats import spearmanr

import sys
sys.path.insert(0, '..')

from src.data_loading import load_data, create_ai_categories
from src.stats_utils import (
    mann_whitney_test, pairwise_comparisons, fdr_correction,
    bootstrap_ci, comprehensive_comparison
)
from src.plotting import setup_style, save_figure, plot_dose_response
from src.constants import AI_CONTENT_LABELS, AI_GRADIENT


def analyze_ratings_by_ai_level(submissions_df: pd.DataFrame,
                                 verbose: bool = True) -> Dict:
    """
    Descriptive analysis of ratings across AI content levels.
    """
    if verbose:
        print("\n" + "="*70)
        print("RATINGS BY AI CONTENT LEVEL")
        print("="*70)
    
    df = submissions_df.dropna(subset=['ai_percentage', 'avg_rating']).copy()
    
    if verbose:
        print(f"\nSample: {len(df):,} papers")
        print(f"Papers with exactly 0% AI: {(df['ai_percentage'] == 0).sum():,}")
    
    # Create categories
    def categorize(pct):
        if pct == 0:
            return '0% (Pure Human)'
        elif pct <= 10:
            return '1-10% (Minimal)'
        elif pct <= 25:
            return '10-25% (Light)'
        elif pct <= 50:
            return '25-50% (Moderate)'
        elif pct <= 75:
            return '50-75% (Heavy)'
        else:
            return '75-100% (Full AI)'
    
    df['ai_category'] = df['ai_percentage'].apply(categorize)
    
    category_order = ['0% (Pure Human)', '1-10% (Minimal)', '10-25% (Light)',
                      '25-50% (Moderate)', '50-75% (Heavy)', '75-100% (Full AI)']
    
    # Summary statistics
    summary = []
    for cat in category_order:
        subset = df[df['ai_category'] == cat]['avg_rating']
        if len(subset) > 0:
            boot = bootstrap_ci(subset.values)
            summary.append({
                'Category': cat,
                'n': len(subset),
                'Mean': subset.mean(),
                'Std': subset.std(),
                'SE': subset.std() / np.sqrt(len(subset)),
                'CI_lower': boot['ci_lower'],
                'CI_upper': boot['ci_upper'],
                'Median': subset.median()
            })
    
    summary_df = pd.DataFrame(summary)
    
    if verbose:
        print("\n" + "-"*50)
        print("Summary Statistics by AI Level")
        print("-"*50)
        print(summary_df.round(3).to_string(index=False))
        
        # Find peak
        peak_idx = summary_df['Mean'].idxmax()
        peak_cat = summary_df.loc[peak_idx, 'Category']
        peak_rating = summary_df.loc[peak_idx, 'Mean']
        print(f"\n→ Highest mean rating: {peak_cat} ({peak_rating:.3f})")
    
    # Check for monotonicity
    means = summary_df['Mean'].values
    is_monotonic_dec = all(means[i] >= means[i+1] for i in range(len(means)-1))
    is_monotonic_inc = all(means[i] <= means[i+1] for i in range(len(means)-1))
    
    if verbose:
        print("\n" + "-"*50)
        print("Pattern Detection")
        print("-"*50)
        if is_monotonic_dec:
            print("→ Pattern: MONOTONICALLY DECREASING (more AI = worse)")
        elif is_monotonic_inc:
            print("→ Pattern: MONOTONICALLY INCREASING (more AI = better)")
        else:
            print("→ Pattern: NON-MONOTONIC (possible inverted-U)")
    
    return {
        'df': df,
        'summary': summary_df,
        'is_monotonic': is_monotonic_dec or is_monotonic_inc,
        'peak_category': peak_cat
    }


def test_quadratic_relationship(submissions_df: pd.DataFrame,
                                 verbose: bool = True) -> Dict:
    """
    Test for inverted-U relationship using quadratic regression.
    
    Model: Rating = β₀ + β₁(AI%) + β₂(AI%)² + ε
    
    If β₂ < 0 and significant: inverted-U (optimal point exists)
    If β₂ > 0 and significant: U-shaped (worst point exists)
    If β₂ not significant: linear relationship
    """
    if verbose:
        print("\n" + "="*70)
        print("QUADRATIC REGRESSION (INVERTED-U TEST)")
        print("="*70)
    
    try:
        import statsmodels.api as sm
    except ImportError:
        if verbose:
            print("statsmodels required for regression")
        return None
    
    df = submissions_df.dropna(subset=['ai_percentage', 'avg_rating']).copy()
    
    # Prepare variables
    df['ai_pct'] = df['ai_percentage']
    df['ai_pct_sq'] = df['ai_percentage'] ** 2
    
    # Model 1: Linear
    X1 = sm.add_constant(df['ai_pct'])
    model1 = sm.OLS(df['avg_rating'], X1).fit()
    
    # Model 2: Quadratic
    X2 = sm.add_constant(df[['ai_pct', 'ai_pct_sq']])
    model2 = sm.OLS(df['avg_rating'], X2).fit()
    
    results = {
        'linear_coef': model1.params['ai_pct'],
        'linear_pval': model1.pvalues['ai_pct'],
        'linear_rsq': model1.rsquared,
        'quad_beta1': model2.params['ai_pct'],
        'quad_beta2': model2.params['ai_pct_sq'],
        'quad_beta2_pval': model2.pvalues['ai_pct_sq'],
        'quad_rsq': model2.rsquared
    }
    
    # F-test for quadratic improvement
    from scipy.stats import f as f_dist
    rss1, rss2 = model1.ssr, model2.ssr
    df1, df2 = model1.df_resid, model2.df_resid
    f_stat = ((rss1 - rss2) / (df1 - df2)) / (rss2 / df2)
    p_f = 1 - f_dist.cdf(f_stat, df1 - df2, df2)
    
    results['f_stat'] = f_stat
    results['f_pval'] = p_f
    
    if verbose:
        print("\n--- Model 1: Linear ---")
        print(f"β₁ (AI%): {model1.params['ai_pct']:.6f} (p = {model1.pvalues['ai_pct']:.2e})")
        print(f"R² = {model1.rsquared:.4f}")
        
        print("\n--- Model 2: Quadratic ---")
        print(f"β₁ (AI%): {model2.params['ai_pct']:.6f} (p = {model2.pvalues['ai_pct']:.2e})")
        print(f"β₂ (AI%²): {model2.params['ai_pct_sq']:.8f} (p = {model2.pvalues['ai_pct_sq']:.2e})")
        print(f"R² = {model2.rsquared:.4f}")
        
        print(f"\nF-test (quadratic vs linear): F = {f_stat:.2f}, p = {p_f:.4f}")
    
    # Interpret
    beta2 = model2.params['ai_pct_sq']
    p_beta2 = model2.pvalues['ai_pct_sq']
    
    if p_beta2 < 0.05:
        if beta2 < 0:
            # Inverted-U
            optimal = -model2.params['ai_pct'] / (2 * beta2)
            optimal = max(0, min(100, optimal))
            results['shape'] = 'inverted-U'
            results['optimal_ai'] = optimal
            
            if verbose:
                print(f"\n✓ INVERTED-U DETECTED (β₂ < 0, p < 0.05)")
                print(f"→ Optimal AI content: {optimal:.1f}%")
        else:
            # U-shaped
            worst = -model2.params['ai_pct'] / (2 * beta2)
            worst = max(0, min(100, worst))
            results['shape'] = 'U'
            results['worst_ai'] = worst
            
            if verbose:
                print(f"\n✓ U-SHAPED (β₂ > 0, p < 0.05)")
                print(f"→ Worst AI content: {worst:.1f}%")
    else:
        results['shape'] = 'linear'
        
        if verbose:
            print(f"\n✗ NO SIGNIFICANT QUADRATIC TERM (p = {p_beta2:.3f})")
            print("→ Relationship is approximately LINEAR")
            
            if model1.params['ai_pct'] < 0:
                print("→ More AI = lower ratings (monotonic decline)")
            else:
                print("→ More AI = higher ratings (monotonic increase)")
    
    return results


def test_pairwise_comparisons(submissions_df: pd.DataFrame,
                               verbose: bool = True) -> pd.DataFrame:
    """
    Pairwise comparisons of each AI level against baseline.
    """
    if verbose:
        print("\n" + "="*70)
        print("PAIRWISE COMPARISONS")
        print("="*70)
    
    df = submissions_df.dropna(subset=['ai_percentage', 'avg_rating']).copy()
    
    # Define groups
    def categorize(pct):
        if pct == 0:
            return '0% (Pure Human)'
        elif pct <= 10:
            return '1-10%'
        elif pct <= 25:
            return '10-25%'
        elif pct <= 50:
            return '25-50%'
        else:
            return '>50%'
    
    df['ai_group'] = df['ai_percentage'].apply(categorize)
    
    groups = ['0% (Pure Human)', '1-10%', '10-25%', '25-50%', '>50%']
    available = [g for g in groups if len(df[df['ai_group'] == g]) >= 10]
    
    # Determine baseline
    if '0% (Pure Human)' in available and len(df[df['ai_group'] == '0% (Pure Human)']) >= 30:
        baseline = '0% (Pure Human)'
        if verbose:
            print(f"\n✓ Using TRUE 0% AI as baseline")
    else:
        baseline = available[0]
        if verbose:
            print(f"\n⚠ Using {baseline} as baseline (limited pure human data)")
    
    # Extract group data
    group_data = [df[df['ai_group'] == g]['avg_rating'].values for g in available]
    
    # Run comparisons
    results = pairwise_comparisons(group_data, available, 
                                    baseline_idx=available.index(baseline),
                                    correction='fdr')
    
    if verbose:
        print("\n" + "-"*50)
        print(f"Comparisons vs {baseline}")
        print("-"*50)
        print(results.round(4).to_string(index=False))
    
    return results


def create_collaboration_figure(submissions_df: pd.DataFrame,
                                 results: Dict,
                                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Create visualization for collaboration hypothesis.
    """
    setup_style()
    
    df = submissions_df.dropna(subset=['ai_percentage', 'avg_rating']).copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel A: Mean ratings by AI bin
    ax1 = axes[0, 0]
    if 'summary' in results:
        summary = results['summary']
        x = range(len(summary))
        
        ax1.bar(x, summary['Mean'], yerr=1.96*summary['SE'],
               color=AI_GRADIENT[:len(summary)], edgecolor='black', capsize=4)
        
        overall_mean = df['avg_rating'].mean()
        ax1.axhline(y=overall_mean, color='gray', linestyle='--',
                   label=f'Overall: {overall_mean:.2f}')
        
        ax1.set_xticks(x)
        ax1.set_xticklabels([c.split('(')[0].strip() for c in summary['Category']], rotation=20)
        ax1.set_ylabel('Mean Rating')
        ax1.set_title('A. Rating by AI Content Level')
        ax1.legend()
    
    # Panel B: Dose-response with quadratic fit
    ax2 = axes[0, 1]
    plot_dose_response(df['ai_percentage'], df['avg_rating'], ax=ax2,
                       title='B. Dose-Response Curve', fit_type='quadratic')
    
    # Panel C: Key comparison (0% vs light vs heavy)
    ax3 = axes[1, 0]
    
    groups_data = {}
    for name, condition in [('Pure Human\n(0%)', df['ai_percentage'] == 0),
                           ('Light\n(5-15%)', (df['ai_percentage'] > 5) & (df['ai_percentage'] <= 15)),
                           ('Heavy\n(>50%)', df['ai_percentage'] > 50)]:
        subset = df.loc[condition, 'avg_rating']
        if len(subset) >= 10:
            groups_data[name] = subset
    
    if groups_data:
        box_data = [groups_data[k].values for k in groups_data.keys()]
        colors = ['#2ca02c', 'steelblue', '#d62728'][:len(groups_data)]
        
        bp = ax3.boxplot(box_data, labels=groups_data.keys(), patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add mean markers
        for i, (name, data) in enumerate(groups_data.items()):
            ax3.scatter(i+1, data.mean(), color='black', s=100, zorder=5, marker='D')
            ax3.annotate(f'μ={data.mean():.2f}', xy=(i+1, data.mean()),
                        xytext=(i+1.2, data.mean()), fontsize=9)
        
        ax3.set_ylabel('Rating')
        ax3.set_title('C. Key Comparison')
    
    # Panel D: Conclusion text
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Generate conclusion
    quad_results = results.get('quadratic', {})
    shape = quad_results.get('shape', 'unknown')
    
    if shape == 'inverted-U':
        optimal = quad_results.get('optimal_ai', 'N/A')
        conclusion = [
            "CONCLUSION: COLLABORATION HYPOTHESIS",
            "=" * 40,
            "",
            f"✓ Inverted-U relationship detected",
            f"✓ Optimal AI content: ~{optimal:.0f}%",
            "",
            "Interpretation:",
            "Moderate AI assistance improves quality.",
            "Too little AI = no benefit.",
            "Too much AI = quality degradation.",
            "",
            f"Sweet spot: ~{optimal:.0f}% AI content"
        ]
    elif shape == 'linear':
        slope = quad_results.get('linear_coef', 0)
        if slope < 0:
            conclusion = [
                "CONCLUSION: DEGRADATION HYPOTHESIS",
                "=" * 40,
                "",
                "✗ Monotonically decreasing relationship",
                "✗ No optimal 'sweet spot' for AI",
                "",
                "Interpretation:",
                "MORE AI content → LOWER ratings",
                "AI appears to degrade quality",
                "at ALL levels of use."
            ]
        else:
            conclusion = [
                "CONCLUSION: UNEXPECTED",
                "=" * 40,
                "",
                "? Monotonically increasing relationship",
                "? More AI = higher ratings",
                "",
                "Possible explanations:",
                "- Selection effects",
                "- Confounding variables"
            ]
    else:
        conclusion = ["Analysis inconclusive"]
    
    ax4.text(0.1, 0.9, '\n'.join(conclusion), transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(facecolor='lightyellow', alpha=0.8, edgecolor='gray'))
    ax4.set_title('D. Summary')
    
    plt.suptitle('AI-Human Collaboration Hypothesis Test', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def run_collaboration_analysis(submissions_path_or_df, reviews_path_or_df=None,
                                save_figures: bool = True,
                                output_dir: str = '.') -> Dict:
    """
    Run complete collaboration hypothesis analysis.
    """
    # Load data
    if isinstance(submissions_path_or_df, str):
        submissions_df, reviews_df = load_data(submissions_path_or_df, reviews_path_or_df)
    else:
        submissions_df = submissions_path_or_df
        reviews_df = reviews_path_or_df
    
    print("="*70)
    print("AI-HUMAN COLLABORATION HYPOTHESIS TEST")
    print("="*70)
    
    # Analyses
    descriptive = analyze_ratings_by_ai_level(submissions_df)
    quadratic = test_quadratic_relationship(submissions_df)
    pairwise = test_pairwise_comparisons(submissions_df)
    
    results = {
        'summary': descriptive['summary'],
        'peak_category': descriptive['peak_category'],
        'quadratic': quadratic,
        'pairwise': pairwise
    }
    
    # Figure
    if save_figures:
        import os
        os.makedirs(output_dir, exist_ok=True)
        fig = create_collaboration_figure(submissions_df, results)
        save_figure(fig, os.path.join(output_dir, 'fig_collaboration.png'))
    
    return results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) >= 3:
        results = run_collaboration_analysis(sys.argv[1], sys.argv[2])
    else:
        print("""
AI-Human Collaboration Hypothesis Test
======================================

Usage:
    python collaboration_hypothesis.py submissions.csv reviews.csv

Or in Python:
    from analysis.collaboration_hypothesis import run_collaboration_analysis
    results = run_collaboration_analysis(submissions_df, reviews_df)
""")
