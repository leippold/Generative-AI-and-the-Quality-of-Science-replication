"""
Effort Proxy Analysis: Substitution Signature
==============================================

Tests whether AI papers show "polish without substance":
- High Presentation scores (AI writes well)
- Low Soundness/Contribution scores (AI doesn't improve methodology)

This tests φ ≈ 0 (substitution) vs φ > φ̄ (augmentation)

The "substitution signature" is when:
    Presentation - Soundness gap INCREASES with AI content
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional

import sys
sys.path.insert(0, '..')

from src.data_loading import load_data, merge_paper_info, create_ai_categories
from src.stats_utils import (
    ols_with_clustered_se, ols_robust, bootstrap_ci,
    kruskal_wallis_test, pairwise_comparisons, fdr_correction
)
from src.plotting import setup_style, save_figure
from src.constants import AI_CONTENT_LABELS, AI_GRADIENT, COMPONENT_METRICS


def analyze_substitution_signature(reviews_df: pd.DataFrame,
                                    submissions_df: pd.DataFrame,
                                    verbose: bool = True) -> Dict:
    """
    Test for substitution signature: Presentation - Soundness gap.

    IMPORTANT: Uses paper-level aggregation for proper statistics.
    Multiple reviews per paper are first averaged, then analyzed.

    Returns
    -------
    dict with gap analysis results
    """
    if verbose:
        print("\n" + "="*70)
        print("SUBSTITUTION SIGNATURE: Presentation - Soundness Gap")
        print("="*70)

    # Merge and prepare
    merged = merge_paper_info(reviews_df, submissions_df)

    # Check available metrics
    available = [m for m in COMPONENT_METRICS if m in merged.columns]
    if verbose:
        print(f"\nAvailable metrics: {available}")

    if 'soundness' not in available or 'presentation' not in available:
        if verbose:
            print("ERROR: Need both soundness and presentation columns")
        return None

    clean = merged.dropna(subset=['soundness', 'presentation', 'ai_percentage']).copy()

    if verbose:
        print(f"Sample: {len(clean):,} reviews")

    # Calculate gap at review level (needed for paper-level aggregation)
    clean['pres_minus_sound'] = clean['presentation'] - clean['soundness']

    # =========================================================================
    # AGGREGATE TO PAPER LEVEL for proper statistics
    # =========================================================================
    agg_cols = {'pres_minus_sound': 'mean', 'ai_percentage': 'first',
                'soundness': 'mean', 'presentation': 'mean'}
    for m in available:
        if m not in agg_cols:
            agg_cols[m] = 'mean'
    if 'confidence' in clean.columns:
        agg_cols['confidence'] = 'mean'

    paper_df = clean.groupby('submission_number').agg(agg_cols).reset_index()

    if verbose:
        print(f"Aggregated to paper level: {len(clean):,} reviews → {len(paper_df):,} papers")

    # Create AI bins on paper-level data
    paper_df['ai_bin'] = create_ai_categories(paper_df)
    clean['ai_bin'] = create_ai_categories(clean)  # Keep for plotting

    results = {}

    # =========================================================================
    # ANALYSIS 1: Gap by AI Content Level (PAPER-LEVEL)
    # =========================================================================
    if verbose:
        print("\n" + "-"*50)
        print("ANALYSIS 1: Gap by AI Content Level (Paper-Level)")
        print("-"*50)

    gap_stats = paper_df.groupby('ai_bin', observed=True)['pres_minus_sound'].agg(
        ['mean', 'std', 'count']
    )
    gap_stats['se'] = gap_stats['std'] / np.sqrt(gap_stats['count'])

    results['gap_by_ai'] = gap_stats

    if verbose:
        print("\nMean (Presentation - Soundness) by AI Content:")
        print(gap_stats.round(4).to_string())

    # Test for trend (on paper-level data)
    kw = kruskal_wallis_test([
        paper_df[paper_df['ai_bin'] == cat]['pres_minus_sound'].values
        for cat in paper_df['ai_bin'].cat.categories
        if len(paper_df[paper_df['ai_bin'] == cat]) > 0
    ])

    results['kruskal_wallis'] = kw

    if verbose:
        print(f"\nKruskal-Wallis test: H = {kw['h_stat']:.2f}, p = {kw['p_value']:.4e}")

    # =========================================================================
    # ANALYSIS 2: Regression Analysis (PAPER-LEVEL)
    # =========================================================================
    if verbose:
        print("\n" + "-"*50)
        print("ANALYSIS 2: Regression - Gap ~ AI Percentage (Paper-Level)")
        print("-"*50)

    # OLS on paper-level data (no clustering needed - one obs per paper)
    try:
        reg = ols_robust(paper_df, 'pres_minus_sound ~ ai_percentage', 'HC3')
        results['regression_gap'] = reg

        coef = reg['params']['ai_percentage']
        p_val = reg['p_values']['ai_percentage']

        if verbose:
            print(f"\nCoefficient (per 1% AI): {coef:.6f}")
            print(f"Per 100% AI: {coef * 100:.4f} point increase in gap")
            print(f"p-value: {p_val:.4e}")
            print(f"R²: {reg['rsquared']:.4f}")
            print(f"N (papers): {len(paper_df):,}")

            if coef > 0 and p_val < 0.05:
                print("\n✓ SUBSTITUTION SIGNATURE DETECTED")
                print("  AI papers have higher presentation relative to soundness")
                print("  → Consistent with φ ≈ 0 (AI used for polish, not substance)")
            else:
                print("\n→ No significant substitution signature")

    except Exception as e:
        if verbose:
            print(f"Regression failed: {e}")

    # =========================================================================
    # ANALYSIS 3: Individual Component Trajectories (PAPER-LEVEL)
    # =========================================================================
    if verbose:
        print("\n" + "-"*50)
        print("ANALYSIS 3: Individual Component Trajectories (Paper-Level)")
        print("-"*50)

    component_results = []

    for metric in available:
        if metric in paper_df.columns:
            try:
                reg = ols_robust(paper_df, f'{metric} ~ ai_percentage', 'HC3')
                component_results.append({
                    'Metric': metric.capitalize(),
                    'Coef_per_100pct': reg['params']['ai_percentage'] * 100,
                    'p_value': reg['p_values']['ai_percentage'],
                    'R_squared': reg['rsquared']
                })

            except Exception as e:
                pass

    if component_results:
        comp_df = pd.DataFrame(component_results)

        # FDR correction
        fdr = fdr_correction(comp_df['p_value'].values)
        comp_df['p_adjusted'] = fdr['adjusted_p']
        comp_df['significant'] = fdr['significant']

        results['component_regression'] = comp_df

        if verbose:
            print("\nCoefficient (per 100% AI content):")
            print(comp_df.round(4).to_string(index=False))

    # =========================================================================
    # ANALYSIS 4: Confidence as Verification Difficulty (PAPER-LEVEL)
    # =========================================================================
    if 'confidence' in paper_df.columns:
        if verbose:
            print("\n" + "-"*50)
            print("ANALYSIS 4: Reviewer Confidence (Paper-Level)")
            print("-"*50)

        conf_stats = paper_df.groupby('ai_bin', observed=True)['confidence'].agg(['mean', 'std', 'count'])
        results['confidence_by_ai'] = conf_stats

        if verbose:
            print("\nMean Confidence by AI Content:")
            print(conf_stats.round(3).to_string())

        try:
            reg_conf = ols_robust(paper_df, 'confidence ~ ai_percentage', 'HC3')
            results['confidence_regression'] = reg_conf

            if verbose:
                coef = reg_conf['params']['ai_percentage']
                p_val = reg_conf['p_values']['ai_percentage']
                print(f"\nRegression: Confidence ~ AI%")
                print(f"  Coefficient (per 1%): {coef:.6f}")
                print(f"  p-value: {p_val:.4e}")

                if coef < 0 and p_val < 0.05:
                    print("\n→ Lower confidence for AI papers")
                    print("  AI papers are harder to evaluate → wider verification gap")
        except:
            pass

    # Store paper-level data for other functions
    results['paper_level_data'] = paper_df

    return results


def create_effort_proxies_figure(results: Dict,
                                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Create visualization for effort proxy analysis.

    Uses paper-level aggregated data from results for proper statistics.
    """
    setup_style()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Get paper-level data from results
    paper_df = results.get('paper_level_data')
    if paper_df is None:
        raise ValueError("No paper_level_data in results. Run analyze_substitution_signature first.")

    # Panel A: Gap by AI bin (paper-level)
    ax1 = axes[0]
    if 'gap_by_ai' in results:
        gap_stats = results['gap_by_ai']
        x = range(len(gap_stats))

        ax1.bar(x, gap_stats['mean'], yerr=1.96*gap_stats['se'],
               color=AI_GRADIENT, edgecolor='black', capsize=5)
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
        ax1.set_xticks(x)
        ax1.set_xticklabels(gap_stats.index, rotation=15)
        ax1.set_ylabel('Presentation - Soundness')
        ax1.set_xlabel('AI Content')
        ax1.set_title('A. Substitution Signature (Paper-Level)\n(Positive = polish over substance)')

    # Panel B: Component scores (paper-level)
    ax2 = axes[1]
    means = None
    for metric in COMPONENT_METRICS:
        if metric in paper_df.columns:
            means = paper_df.groupby('ai_bin', observed=True)[metric].mean()
            ax2.plot(range(len(means)), means.values, 'o-',
                    label=metric.capitalize(), linewidth=2, markersize=8)

    if means is not None:
        ax2.set_xticks(range(len(means)))
        ax2.set_xticklabels(means.index, rotation=15)
    ax2.set_ylabel('Score')
    ax2.set_xlabel('AI Content')
    ax2.set_title('B. Component Scores by AI Content\n(Paper-Level Means)')
    ax2.legend()

    # Panel C: Confidence (paper-level)
    ax3 = axes[2]
    if 'confidence' in paper_df.columns:
        conf_means = paper_df.groupby('ai_bin', observed=True)['confidence'].mean()
        conf_sems = paper_df.groupby('ai_bin', observed=True)['confidence'].sem()

        ax3.bar(range(len(conf_means)), conf_means.values,
               yerr=1.96*conf_sems.values,
               color='steelblue', edgecolor='black', capsize=5)
        ax3.set_xticks(range(len(conf_means)))
        ax3.set_xticklabels(conf_means.index, rotation=15)
        ax3.set_ylabel('Reviewer Confidence')
        ax3.set_xlabel('AI Content')
        ax3.set_title('C. Verification Difficulty\n(Lower = harder to evaluate)')
    else:
        ax3.text(0.5, 0.5, 'Confidence data\nnot available',
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('C. Reviewer Confidence')

    plt.suptitle('Effort Proxies: Substitution Signature in AI Papers (Paper-Level)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


def run_effort_proxy_analysis(submissions_path_or_df, reviews_path_or_df=None,
                               save_figures: bool = True,
                               output_dir: str = '.') -> Dict:
    """
    Run complete effort proxy analysis.

    All analysis is done at paper-level to avoid pseudo-replication
    from multiple reviews per paper.
    """
    # Load data
    if isinstance(submissions_path_or_df, str):
        submissions_df, reviews_df = load_data(submissions_path_or_df, reviews_path_or_df)
    else:
        submissions_df = submissions_path_or_df
        reviews_df = reviews_path_or_df

    print("="*70)
    print("EFFORT PROXY ANALYSIS: Substitution Signature (Paper-Level)")
    print("="*70)

    # Run analysis (now includes paper-level aggregation internally)
    results = analyze_substitution_signature(reviews_df, submissions_df)

    # Figure (now uses paper-level data from results)
    if save_figures and results:
        import os
        os.makedirs(output_dir, exist_ok=True)
        fig = create_effort_proxies_figure(results)
        save_figure(fig, os.path.join(output_dir, 'fig_effort_proxies.png'))

    return results


# =============================================================================
# STACKED REGRESSION TEST (QJE/REStud-Ready)
# =============================================================================

def test_substitution_stacked_regression(reviews_df: pd.DataFrame,
                                          submissions_df: pd.DataFrame,
                                          verbose: bool = True,
                                          create_figure: bool = True,
                                          save_path: Optional[str] = None) -> Dict:
    """
    QJE/REStud-ready test for substitution signature using stacked regression.

    This is the PREFERRED test for top journals because:
    1. No arbitrary bins - uses continuous AI%
    2. Single, pre-specifiable hypothesis test
    3. Formal coefficient comparison with proper inference

    Model:
        Score ~ AI% + is_soundness + AI% × is_soundness + controls

    Interpretation:
        - Interaction term (AI% × is_soundness) tests differential effect
        - If negative & significant: AI hurts soundness MORE than presentation
        - This is the "substitution signature" - AI helps style, not substance

    Parameters
    ----------
    reviews_df : DataFrame
        Review-level data
    submissions_df : DataFrame
        Paper-level submission data
    verbose : bool
        Print detailed output
    create_figure : bool
        Create coefficient comparison figure
    save_path : str, optional
        Path to save figure

    Returns
    -------
    dict with regression results, coefficient comparison, and figure
    """
    if verbose:
        print("\n" + "="*70)
        print("STACKED REGRESSION TEST: Substitution Signature")
        print("="*70)
        print("\nThis test compares AI effects on Soundness vs Presentation")
        print("H₀: AI affects both components equally (no substitution)")
        print("H₁: AI affects Soundness more negatively (substitution signature)")

    # Merge data
    merged = merge_paper_info(reviews_df, submissions_df)

    # Check required columns
    if 'soundness' not in merged.columns or 'presentation' not in merged.columns:
        raise ValueError("Need both 'soundness' and 'presentation' columns")

    clean = merged.dropna(subset=['soundness', 'presentation', 'ai_percentage']).copy()

    if verbose:
        print(f"\nSample: {len(clean):,} reviews from {clean['submission_number'].nunique():,} papers")

    # =========================================================================
    # STEP 1: Aggregate to paper level
    # =========================================================================
    paper_df = clean.groupby('submission_number').agg({
        'soundness': 'mean',
        'presentation': 'mean',
        'ai_percentage': 'first'
    }).reset_index()

    if verbose:
        print(f"Aggregated to paper level: {len(paper_df):,} papers")

    # =========================================================================
    # STEP 2: Stack data (long format)
    # =========================================================================
    # Each paper gets 2 rows: one for soundness, one for presentation
    stacked = pd.DataFrame({
        'paper_id': list(paper_df['submission_number']) * 2,
        'ai_percentage': list(paper_df['ai_percentage']) * 2,
        'score': list(paper_df['soundness']) + list(paper_df['presentation']),
        'is_soundness': [1] * len(paper_df) + [0] * len(paper_df),
        'component': ['Soundness'] * len(paper_df) + ['Presentation'] * len(paper_df)
    })

    if verbose:
        print(f"Stacked data: {len(stacked):,} observations (2 per paper)")

    # =========================================================================
    # STEP 3: Run stacked regression with clustered SEs
    # =========================================================================
    if verbose:
        print("\n" + "-"*50)
        print("STACKED REGRESSION MODEL")
        print("-"*50)
        print("\nModel: Score ~ AI% + is_soundness + AI% × is_soundness")
        print("Clustered SEs at paper level")

    # Run regression with interaction
    reg_results = ols_with_clustered_se(
        stacked,
        'score ~ ai_percentage * is_soundness',
        cluster_col='paper_id'
    )

    results = {'stacked_regression': reg_results}

    # Extract key coefficients
    ai_coef = reg_results['params'].get('ai_percentage', 0)
    ai_se = reg_results['se'].get('ai_percentage', 0)
    ai_p = reg_results['p_values'].get('ai_percentage', 1)

    soundness_coef = reg_results['params'].get('is_soundness', 0)
    soundness_se = reg_results['se'].get('is_soundness', 0)
    soundness_p = reg_results['p_values'].get('is_soundness', 1)

    interaction_coef = reg_results['params'].get('ai_percentage:is_soundness', 0)
    interaction_se = reg_results['se'].get('ai_percentage:is_soundness', 0)
    interaction_p = reg_results['p_values'].get('ai_percentage:is_soundness', 1)

    if verbose:
        print("\n" + "-"*50)
        print("REGRESSION RESULTS")
        print("-"*50)
        print(f"\n{'Coefficient':<35} {'Estimate':>10} {'SE':>10} {'p-value':>12}")
        print("-"*70)
        print(f"{'AI% (effect on Presentation)':<35} {ai_coef:>10.5f} {ai_se:>10.5f} {ai_p:>12.4e}")
        print(f"{'is_soundness (level difference)':<35} {soundness_coef:>10.5f} {soundness_se:>10.5f} {soundness_p:>12.4e}")
        print(f"{'AI% × is_soundness (INTERACTION)':<35} {interaction_coef:>10.5f} {interaction_se:>10.5f} {interaction_p:>12.4e}")

    # =========================================================================
    # STEP 4: Compute implied effects
    # =========================================================================
    # Effect of AI on Presentation = ai_coef
    # Effect of AI on Soundness = ai_coef + interaction_coef

    effect_presentation = ai_coef
    effect_soundness = ai_coef + interaction_coef

    # Standard error for soundness effect (sum of coefficients)
    # Var(a+b) = Var(a) + Var(b) + 2*Cov(a,b)
    # We approximate with just summing SEs (conservative)
    # For proper inference, we use the interaction term directly

    results['effects'] = {
        'presentation': {
            'coef': effect_presentation,
            'se': ai_se,
            'per_100pct': effect_presentation * 100
        },
        'soundness': {
            'coef': effect_soundness,
            'se': None,  # Complex due to covariance
            'per_100pct': effect_soundness * 100
        },
        'difference': {
            'coef': interaction_coef,
            'se': interaction_se,
            'p_value': interaction_p,
            'per_100pct': interaction_coef * 100
        }
    }

    if verbose:
        print("\n" + "-"*50)
        print("IMPLIED EFFECTS (per 100% AI content)")
        print("-"*50)
        print(f"\nEffect on Presentation: {effect_presentation * 100:+.4f} points")
        print(f"Effect on Soundness:    {effect_soundness * 100:+.4f} points")
        print(f"Difference:             {interaction_coef * 100:+.4f} points")
        print(f"\n*** KEY TEST: Interaction (AI% × is_soundness) ***")
        print(f"    Coefficient: {interaction_coef:.6f}")
        print(f"    p-value:     {interaction_p:.4e}")

    # =========================================================================
    # STEP 5: Interpretation
    # =========================================================================
    if verbose:
        print("\n" + "="*70)
        print("INTERPRETATION")
        print("="*70)

        if interaction_coef < 0 and interaction_p < 0.05:
            print("\n✓ SUBSTITUTION SIGNATURE CONFIRMED")
            print(f"\n  AI content has a significantly MORE NEGATIVE effect on Soundness")
            print(f"  than on Presentation (interaction p = {interaction_p:.4e})")
            print(f"\n  Per 100% AI content:")
            print(f"    - Presentation changes by {effect_presentation * 100:+.3f} points")
            print(f"    - Soundness changes by    {effect_soundness * 100:+.3f} points")
            print(f"    - Difference:             {interaction_coef * 100:+.3f} points")
            print(f"\n  This is consistent with AI providing stylistic polish")
            print(f"  without improving (or while degrading) intellectual substance.")
            results['conclusion'] = 'substitution_confirmed'

        elif interaction_coef < 0 and interaction_p < 0.10:
            print("\n⚠ SUGGESTIVE EVIDENCE OF SUBSTITUTION")
            print(f"\n  The interaction is negative but only marginally significant")
            print(f"  (p = {interaction_p:.4f})")
            results['conclusion'] = 'suggestive'

        else:
            print("\n✗ NO SIGNIFICANT SUBSTITUTION SIGNATURE")
            print(f"\n  AI does not differentially affect Soundness vs Presentation")
            print(f"  (interaction p = {interaction_p:.4f})")
            print(f"\n  The AI effect appears uniform across review components.")
            results['conclusion'] = 'no_signature'

    # =========================================================================
    # STEP 6: Also run separate regressions for robustness table
    # =========================================================================
    if verbose:
        print("\n" + "-"*50)
        print("ROBUSTNESS: Separate Component Regressions (Paper-Level)")
        print("-"*50)

    separate_results = {}
    for component in ['soundness', 'presentation']:
        try:
            from src.stats_utils import ols_robust
            reg = ols_robust(paper_df, f'{component} ~ ai_percentage', 'HC3')
            separate_results[component] = {
                'coef': reg['params']['ai_percentage'],
                'se': reg['se']['ai_percentage'],
                'p_value': reg['p_values']['ai_percentage'],
                'per_100pct': reg['params']['ai_percentage'] * 100
            }
            if verbose:
                c = reg['params']['ai_percentage']
                se = reg['se']['ai_percentage']
                p = reg['p_values']['ai_percentage']
                print(f"\n{component.capitalize()} ~ AI%:")
                print(f"  Coefficient (per 100%): {c*100:+.4f}")
                print(f"  SE: {se*100:.4f}, p-value: {p:.4e}")
        except Exception as e:
            if verbose:
                print(f"  {component}: regression failed - {e}")

    results['separate_regressions'] = separate_results

    # Coefficient difference from separate regressions
    if 'soundness' in separate_results and 'presentation' in separate_results:
        diff = separate_results['soundness']['coef'] - separate_results['presentation']['coef']
        results['coefficient_difference'] = diff * 100  # per 100% AI

        if verbose:
            print(f"\nCoefficient difference (Soundness - Presentation):")
            print(f"  {diff * 100:+.4f} points per 100% AI")
            print(f"  (Formal test via interaction term above)")

    # =========================================================================
    # STEP 7: Create figure
    # =========================================================================
    if create_figure:
        fig = _create_stacked_regression_figure(results, paper_df)
        results['figure'] = fig

        if save_path:
            save_figure(fig, save_path)
            if verbose:
                print(f"\nFigure saved to: {save_path}")

    return results


def _create_stacked_regression_figure(results: Dict, paper_df: pd.DataFrame) -> plt.Figure:
    """Create publication-quality figure for stacked regression results."""
    setup_style()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Coefficient comparison
    ax1 = axes[0]

    components = ['Presentation', 'Soundness']
    effects = results['effects']

    coefs = [
        effects['presentation']['per_100pct'],
        effects['soundness']['per_100pct']
    ]

    # Use SE from separate regressions for error bars
    sep = results.get('separate_regressions', {})
    errors = [
        sep.get('presentation', {}).get('se', 0) * 100 * 1.96,
        sep.get('soundness', {}).get('se', 0) * 100 * 1.96
    ]

    colors = ['#2ecc71', '#e74c3c']  # Green for presentation, red for soundness
    bars = ax1.bar(components, coefs, yerr=errors, color=colors,
                   edgecolor='black', capsize=8, linewidth=1.5)

    ax1.axhline(0, color='black', linewidth=1, linestyle='-')
    ax1.set_ylabel('Effect per 100% AI Content\n(points)', fontsize=11)
    ax1.set_title('A. AI Effect by Review Component', fontsize=12, fontweight='bold')

    # Add significance stars
    for i, (bar, comp) in enumerate(zip(bars, ['presentation', 'soundness'])):
        p = sep.get(comp, {}).get('p_value', 1)
        if p < 0.001:
            sig = '***'
        elif p < 0.01:
            sig = '**'
        elif p < 0.05:
            sig = '*'
        else:
            sig = ''

        height = bar.get_height()
        offset = errors[i] + 0.02 if height >= 0 else -errors[i] - 0.05
        ax1.annotate(sig, xy=(bar.get_x() + bar.get_width()/2, height + offset),
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=14, fontweight='bold')

    # Add difference annotation
    diff = effects['difference']['per_100pct']
    diff_p = effects['difference']['p_value']
    sig_text = f"Δ = {diff:+.3f}"
    if diff_p < 0.001:
        sig_text += "***"
    elif diff_p < 0.01:
        sig_text += "**"
    elif diff_p < 0.05:
        sig_text += "*"

    ax1.annotate(sig_text, xy=(0.5, 0.95), xycoords='axes fraction',
                ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Panel B: Scatter with regression lines
    ax2 = axes[1]

    # Create scatter
    ax2.scatter(paper_df['ai_percentage'], paper_df['presentation'],
               alpha=0.15, color='#2ecc71', label='Presentation', s=20)
    ax2.scatter(paper_df['ai_percentage'], paper_df['soundness'],
               alpha=0.15, color='#e74c3c', label='Soundness', s=20)

    # Add regression lines
    x_range = np.linspace(paper_df['ai_percentage'].min(),
                          paper_df['ai_percentage'].max(), 100)

    # Get coefficients
    pres_coef = effects['presentation']['coef']
    sound_coef = effects['soundness']['coef']

    # Get intercepts from separate regressions if available
    pres_intercept = paper_df['presentation'].mean() - pres_coef * paper_df['ai_percentage'].mean()
    sound_intercept = paper_df['soundness'].mean() - sound_coef * paper_df['ai_percentage'].mean()

    ax2.plot(x_range, pres_intercept + pres_coef * x_range,
            color='#27ae60', linewidth=2.5, label='Presentation fit')
    ax2.plot(x_range, sound_intercept + sound_coef * x_range,
            color='#c0392b', linewidth=2.5, label='Soundness fit')

    ax2.set_xlabel('AI Content (%)', fontsize=11)
    ax2.set_ylabel('Score', fontsize=11)
    ax2.set_title('B. Component Scores vs AI Content', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)

    # Add interpretation text
    conclusion = results.get('conclusion', '')
    if conclusion == 'substitution_confirmed':
        conclusion_text = "Substitution signature confirmed (p < 0.05)"
        color = '#27ae60'
    elif conclusion == 'suggestive':
        conclusion_text = "Suggestive evidence (p < 0.10)"
        color = '#f39c12'
    else:
        conclusion_text = "No significant differential effect"
        color = '#7f8c8d'

    fig.text(0.5, 0.02, conclusion_text, ha='center', fontsize=11,
             style='italic', color=color, fontweight='bold')

    plt.suptitle('Substitution Signature Test: AI Effect on Soundness vs Presentation',
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) >= 3:
        results = run_effort_proxy_analysis(sys.argv[1], sys.argv[2])
    else:
        print("""
Effort Proxy Analysis (Substitution Signature)
==============================================

Usage:
    python effort_proxies.py submissions.csv reviews.csv

Or in Python:
    from analysis.effort_proxies import run_effort_proxy_analysis
    results = run_effort_proxy_analysis(submissions_df, reviews_df)
""")
