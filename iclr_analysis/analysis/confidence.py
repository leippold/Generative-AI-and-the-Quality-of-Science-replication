"""
Confidence Analysis Module
==========================

Analyzes reviewer confidence patterns and their relationship to:
1. AI content in reviews
2. AI content in papers
3. Rating patterns (confidence-weighted analysis)

Key Questions:
- Do AI reviewers report different confidence levels?
- Are high-confidence AI reviewers more/less accurate?
- Does confidence moderate the echo chamber effect?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional

import sys
sys.path.insert(0, '..')

from src.data_loading import (
    load_data, prepare_echo_chamber_data, merge_paper_info, create_ai_categories
)
from src.stats_utils import (
    mann_whitney_test, kruskal_wallis_test, ols_with_clustered_se,
    bootstrap_ci, paper_weighted_rating, confident_leniency_index,
    permutation_test_interaction
)
from src.plotting import setup_style, save_figure
from src.constants import REVIEW_COLORS, ECHO_CHAMBER_COLORS


def analyze_confidence_by_reviewer_type(reviews_df: pd.DataFrame,
                                         verbose: bool = True) -> Dict:
    """
    Compare confidence levels between human and AI reviewers.
    """
    if verbose:
        print("\n" + "="*70)
        print("CONFIDENCE BY REVIEWER TYPE")
        print("="*70)
    
    if 'confidence' not in reviews_df.columns:
        if verbose:
            print("No confidence column available")
        return None
    
    clean = reviews_df.dropna(subset=['confidence', 'ai_classification']).copy()
    
    # Map to binary reviewer types
    clean['reviewer_type'] = clean['ai_classification'].map({
        'Fully human-written': 'Human',
        'Fully AI-generated': 'AI'
    })
    clean = clean.dropna(subset=['reviewer_type'])
    
    human_conf = clean[clean['reviewer_type'] == 'Human']['confidence']
    ai_conf = clean[clean['reviewer_type'] == 'AI']['confidence']
    
    if verbose:
        print(f"\nSample: {len(human_conf):,} human reviews, {len(ai_conf):,} AI reviews")
    
    # Descriptives
    results = {
        'human_mean': human_conf.mean(),
        'human_std': human_conf.std(),
        'ai_mean': ai_conf.mean(),
        'ai_std': ai_conf.std(),
        'diff': ai_conf.mean() - human_conf.mean()
    }
    
    # Statistical test
    mw = mann_whitney_test(ai_conf.values, human_conf.values)
    results.update({
        'p_value': mw['p_value'],
        'effect_size': mw['effect_size']
    })
    
    # Bootstrap CI for difference
    boot = bootstrap_ci(ai_conf.values - human_conf.mean(), statistic=np.mean)
    results['ci_lower'] = boot['ci_lower']
    results['ci_upper'] = boot['ci_upper']
    
    if verbose:
        print(f"\nHuman reviewers: {results['human_mean']:.3f} ± {results['human_std']:.3f}")
        print(f"AI reviewers: {results['ai_mean']:.3f} ± {results['ai_std']:.3f}")
        print(f"Difference: {results['diff']:+.3f}")
        print(f"p-value: {results['p_value']:.4e}")
        print(f"Effect size (r): {results['effect_size']:.3f}")
        
        if results['p_value'] < 0.05:
            if results['diff'] > 0:
                print("\n✓ AI reviewers report HIGHER confidence")
            else:
                print("\n✓ AI reviewers report LOWER confidence")
        else:
            print("\n→ No significant difference in confidence")
    
    return results


def analyze_confidence_2x2(reviews_df: pd.DataFrame,
                           submissions_df: pd.DataFrame,
                           verbose: bool = True) -> Dict:
    """
    Analyze confidence in the 2×2 (Paper Type × Reviewer Type) design.
    """
    if verbose:
        print("\n" + "="*70)
        print("CONFIDENCE IN 2×2 DESIGN")
        print("="*70)
    
    if 'confidence' not in reviews_df.columns:
        return None
    
    clean = prepare_echo_chamber_data(reviews_df, submissions_df)
    clean = clean.dropna(subset=['confidence'])
    
    # Cell means
    conf_table = clean.pivot_table(
        values='confidence',
        index='paper_type',
        columns='reviewer_type',
        aggfunc='mean'
    )
    
    count_table = pd.crosstab(clean['paper_type'], clean['reviewer_type'])
    
    if verbose:
        print("\nMean Confidence by Cell:")
        print(conf_table.round(3).to_string())
        print("\nSample Sizes:")
        print(count_table.to_string())
    
    # Interaction effect for confidence
    try:
        hh = conf_table.loc['Human Paper', 'Human Review']
        ha = conf_table.loc['Human Paper', 'AI Review']
        ah = conf_table.loc['AI Paper', 'Human Review']
        aa = conf_table.loc['AI Paper', 'AI Review']
        
        interaction = (aa - ah) - (ha - hh)
        
        if verbose:
            print(f"\nConfidence interaction: {interaction:+.4f}")
    except:
        interaction = np.nan
    
    # Regression test
    try:
        ols = ols_with_clustered_se(
            clean, 'confidence ~ paper_AI * reviewer_AI', 'submission_number'
        )
        
        if verbose:
            print("\nOLS (clustered SE):")
            for param in ['paper_AI', 'reviewer_AI', 'paper_AI:reviewer_AI']:
                coef = ols['params'].get(param, np.nan)
                p = ols['p_values'].get(param, np.nan)
                sig = "*" if p < 0.05 else ""
                print(f"  {param}: {coef:+.4f} (p={p:.4f}) {sig}")
                
    except Exception as e:
        if verbose:
            print(f"Regression failed: {e}")
        ols = None
    
    return {
        'conf_table': conf_table,
        'count_table': count_table,
        'interaction': interaction,
        'regression': ols
    }


def analyze_confidence_rating_relationship(reviews_df: pd.DataFrame,
                                            verbose: bool = True) -> Dict:
    """
    Analyze how confidence relates to rating.

    IMPORTANT: Aggregates to paper level for proper statistics.
    Each paper has multiple reviews - we take the mean confidence and rating.

    Tests:
    1. Do high-confidence papers (on average) get different ratings?
    2. Is this pattern different for AI vs human reviewers?
    """
    if verbose:
        print("\n" + "="*70)
        print("CONFIDENCE-RATING RELATIONSHIP (Paper-Level)")
        print("="*70)

    if 'confidence' not in reviews_df.columns or 'rating' not in reviews_df.columns:
        return None

    clean = reviews_df.dropna(subset=['confidence', 'rating', 'ai_classification']).copy()
    clean['reviewer_type'] = clean['ai_classification'].map({
        'Fully human-written': 'Human',
        'Fully AI-generated': 'AI'
    })
    clean = clean.dropna(subset=['reviewer_type'])

    results = {}

    # Aggregate to paper level for overall analysis
    if 'submission_number' in clean.columns:
        paper_df = clean.groupby('submission_number').agg({
            'confidence': 'mean',
            'rating': 'mean'
        }).reset_index()

        if verbose:
            print(f"Aggregated to paper level: {len(clean):,} reviews → {len(paper_df):,} papers")
    else:
        paper_df = clean
        if verbose:
            print("Warning: No submission_number column, using review-level data")

    # Overall correlation (paper-level)
    from scipy.stats import spearmanr
    rho, p = spearmanr(paper_df['confidence'], paper_df['rating'])

    results['overall'] = {'spearman_rho': rho, 'p_value': p, 'n_papers': len(paper_df)}

    if verbose:
        print(f"\nOverall (paper-level): Spearman ρ = {rho:.3f}, p = {p:.4e}, n = {len(paper_df)}")

    # By reviewer type (need to aggregate within reviewer type and paper)
    if verbose:
        print("\nBy reviewer type (review-level with clustered interpretation):")

    for rev_type in ['Human', 'AI']:
        subset = clean[clean['reviewer_type'] == rev_type]
        if len(subset) > 10:
            # Aggregate to paper level within reviewer type
            if 'submission_number' in subset.columns:
                type_paper = subset.groupby('submission_number').agg({
                    'confidence': 'mean',
                    'rating': 'mean'
                }).reset_index()
                rho_t, p_t = spearmanr(type_paper['confidence'], type_paper['rating'])
                n = len(type_paper)
            else:
                rho_t, p_t = spearmanr(subset['confidence'], subset['rating'])
                n = len(subset)

            results[rev_type] = {'spearman_rho': rho_t, 'p_value': p_t, 'n': n}

            if verbose:
                print(f"  {rev_type} reviewers: ρ = {rho_t:.3f}, p = {p_t:.4e}, n = {n} papers")

    # Confident Leniency Index (paper-level)
    if verbose:
        print("\n" + "-"*50)
        print("CONFIDENT LENIENCY INDEX (Paper-Level)")
        print("-"*50)

    for rev_type in ['Human', 'AI']:
        subset = clean[clean['reviewer_type'] == rev_type]
        if len(subset) > 10:
            # Aggregate to paper level for CLI
            if 'submission_number' in subset.columns:
                type_paper = subset.groupby('submission_number').agg({
                    'confidence': 'mean',
                    'rating': 'mean'
                }).reset_index()
                cli = confident_leniency_index(type_paper)
            else:
                cli = confident_leniency_index(subset)

            results[f'{rev_type}_CLI'] = cli

            if verbose:
                print(f"\n{rev_type} reviewers:")
                print(f"  CLI = {cli['cli']:+.3f}")
                print(f"  High-conf mean: {cli['high_conf_mean']:.3f}")
                print(f"  Low-conf mean: {cli['low_conf_mean']:.3f}")
                print(f"  p = {cli['p_value']:.4f}")

    return results


def analyze_confidence_weighted_echo_chamber(reviews_df: pd.DataFrame,
                                              submissions_df: pd.DataFrame,
                                              verbose: bool = True) -> Dict:
    """
    Test echo chamber using confidence-weighted ratings.
    
    Weights ratings by reviewer confidence.
    More confident reviews get more weight.
    """
    if verbose:
        print("\n" + "="*70)
        print("CONFIDENCE-WEIGHTED ECHO CHAMBER ANALYSIS")
        print("="*70)
    
    if 'confidence' not in reviews_df.columns:
        return None
    
    clean = prepare_echo_chamber_data(reviews_df, submissions_df)
    clean = clean.dropna(subset=['confidence'])
    
    # Unweighted (for comparison)
    unweighted = clean.pivot_table(
        values='rating', index='paper_type', columns='reviewer_type', aggfunc='mean'
    )
    
    # Weighted
    def weighted_mean(x):
        ratings = x['rating'].values
        weights = x['confidence'].values
        return np.average(ratings, weights=weights)
    
    weighted = clean.groupby(['paper_type', 'reviewer_type']).apply(weighted_mean).unstack()
    
    if verbose:
        print("\nUnweighted means:")
        print(unweighted.round(3).to_string())
        print("\nConfidence-weighted means:")
        print(weighted.round(3).to_string())
    
    # Calculate interactions
    try:
        # Unweighted
        uw_int = ((unweighted.loc['AI Paper', 'AI Review'] - 
                   unweighted.loc['AI Paper', 'Human Review']) -
                  (unweighted.loc['Human Paper', 'AI Review'] -
                   unweighted.loc['Human Paper', 'Human Review']))
        
        # Weighted
        w_int = ((weighted.loc['AI Paper', 'AI Review'] -
                  weighted.loc['AI Paper', 'Human Review']) -
                 (weighted.loc['Human Paper', 'AI Review'] -
                  weighted.loc['Human Paper', 'Human Review']))
        
        if verbose:
            print(f"\nUnweighted interaction: {uw_int:+.4f}")
            print(f"Weighted interaction: {w_int:+.4f}")
            print(f"Change: {w_int - uw_int:+.4f}")
            
            if abs(w_int) > abs(uw_int):
                print("\n→ Weighting by confidence STRENGTHENS the echo chamber")
            elif abs(w_int) < abs(uw_int):
                print("\n→ Weighting by confidence WEAKENS the echo chamber")
            else:
                print("\n→ Confidence weighting has minimal effect")
                
    except Exception as e:
        uw_int, w_int = np.nan, np.nan
    
    return {
        'unweighted': unweighted,
        'weighted': weighted,
        'unweighted_interaction': uw_int,
        'weighted_interaction': w_int
    }


def create_confidence_figure(reviews_df: pd.DataFrame,
                              results: Dict,
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Create visualization for confidence analysis.
    """
    setup_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    clean = reviews_df.dropna(subset=['confidence', 'rating', 'ai_classification']).copy()
    clean['reviewer_type'] = clean['ai_classification'].map({
        'Fully human-written': 'Human',
        'Fully AI-generated': 'AI'
    })
    clean = clean.dropna(subset=['reviewer_type'])
    
    # Panel A: Confidence distribution by reviewer type
    ax1 = axes[0, 0]
    
    human_conf = clean[clean['reviewer_type'] == 'Human']['confidence']
    ai_conf = clean[clean['reviewer_type'] == 'AI']['confidence']
    
    ax1.hist(human_conf, bins=10, alpha=0.6, label='Human', color='#2ca02c')
    ax1.hist(ai_conf, bins=10, alpha=0.6, label='AI', color='#d62728')
    ax1.axvline(human_conf.mean(), color='#2ca02c', linestyle='--', linewidth=2)
    ax1.axvline(ai_conf.mean(), color='#d62728', linestyle='--', linewidth=2)
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Count')
    ax1.set_title('A. Confidence Distribution')
    ax1.legend()
    
    # Panel B: Confidence vs Rating by reviewer type
    # Note: Uses review-level data with transparency to show density
    ax2 = axes[0, 1]

    for rev_type, color in [('Human', '#2ca02c'), ('AI', '#d62728')]:
        subset = clean[clean['reviewer_type'] == rev_type]
        ax2.scatter(subset['confidence'], subset['rating'],
                   alpha=0.15, s=15, label=rev_type, color=color)

    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Rating')
    ax2.set_title('B. Confidence vs Rating\n(Review-level, α=0.15 for density)')
    ax2.legend()
    
    # Panel C: Confidence heatmap if 2×2 data available
    ax3 = axes[1, 0]
    
    if 'conf_2x2' in results and 'conf_table' in results['conf_2x2']:
        conf_table = results['conf_2x2']['conf_table']
        sns.heatmap(conf_table, annot=True, fmt='.2f', cmap='Blues', ax=ax3)
        ax3.set_title('C. Confidence in 2×2 Design')
    else:
        ax3.text(0.5, 0.5, '2×2 data\nnot available', ha='center', va='center',
                transform=ax3.transAxes)
        ax3.set_title('C. Confidence in 2×2 Design')
    
    # Panel D: Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary = [
        "CONFIDENCE ANALYSIS SUMMARY",
        "=" * 30,
        ""
    ]
    
    if 'by_reviewer' in results:
        r = results['by_reviewer']
        summary.append(f"Human conf: {r['human_mean']:.2f} ± {r['human_std']:.2f}")
        summary.append(f"AI conf: {r['ai_mean']:.2f} ± {r['ai_std']:.2f}")
        summary.append(f"Diff: {r['diff']:+.2f} (p={r['p_value']:.4f})")
    
    ax4.text(0.1, 0.9, '\n'.join(summary), transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(facecolor='lightyellow', alpha=0.8, edgecolor='gray'))
    ax4.set_title('D. Summary')
    
    plt.suptitle('Reviewer Confidence Analysis',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def run_confidence_analysis(submissions_path_or_df, reviews_path_or_df=None,
                             save_figures: bool = True,
                             output_dir: str = '.') -> Dict:
    """
    Run complete confidence analysis.
    """
    # Load data
    if isinstance(submissions_path_or_df, str):
        submissions_df, reviews_df = load_data(submissions_path_or_df, reviews_path_or_df)
    else:
        submissions_df = submissions_path_or_df
        reviews_df = reviews_path_or_df
    
    print("="*70)
    print("CONFIDENCE ANALYSIS")
    print("="*70)
    
    results = {}
    
    # Analyses
    results['by_reviewer'] = analyze_confidence_by_reviewer_type(reviews_df)
    results['conf_2x2'] = analyze_confidence_2x2(reviews_df, submissions_df)
    results['conf_rating'] = analyze_confidence_rating_relationship(reviews_df)
    results['weighted'] = analyze_confidence_weighted_echo_chamber(reviews_df, submissions_df)
    
    # Figure
    if save_figures:
        import os
        os.makedirs(output_dir, exist_ok=True)
        fig = create_confidence_figure(reviews_df, results)
        save_figure(fig, os.path.join(output_dir, 'fig_confidence.png'))
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) >= 3:
        results = run_confidence_analysis(sys.argv[1], sys.argv[2])
    else:
        print("""
Confidence Analysis
===================

Usage:
    python confidence.py submissions.csv reviews.csv
""")
