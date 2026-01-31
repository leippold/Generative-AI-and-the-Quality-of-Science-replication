"""
AI Echo Chamber Analysis - CONSOLIDATED & ROBUST
=================================================

Tests for systematic bias in AI reviewer ratings of AI papers.

Research Questions:
1. Do AI reviewers rate AI papers differently than human reviewers?
2. Is there a "same-type advantage" (diagonal effect)?
3. Are AI reviewers more lenient overall?

Statistical Methods:
- OLS with clustered standard errors (by submission)
- Permutation test for interaction effect
- Bootstrap confidence intervals
- Multiple robustness checks across thresholds

METHODOLOGICAL NOTES:
- Reviews are clustered within submissions
- Ordinal ratings treated as approximately continuous for interaction analysis
- Multiple threshold specifications for robustness
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional
import warnings

import sys
sys.path.insert(0, '..')

from src.data_loading import (
    load_data, prepare_echo_chamber_data, get_cell_data, merge_paper_info
)
from src.stats_utils import (
    mann_whitney_test, levene_test, cohens_d, hedges_g,
    permutation_test_interaction, bootstrap_diff_ci,
    ols_with_clustered_se, ols_robust, compute_icc,
    comprehensive_comparison
)
from src.plotting import (
    setup_style, save_figure, plot_interaction_2x2, 
    plot_heatmap, plot_permutation_distribution
)
from src.constants import (
    ROBUSTNESS_THRESHOLDS, ECHO_CHAMBER_COLORS, PAPER_TYPE_COLORS,
    N_PERMUTATIONS, RANDOM_SEED, MIN_CELL_SIZE, ALPHA
)


def compute_interaction_effect(df: pd.DataFrame) -> Dict:
    """
    Compute 2×2 interaction effect and cell statistics.
    
    Interaction = (AA - AH) - (HA - HH)
    
    Positive interaction → AI reviewers give relatively higher ratings to AI papers
    
    Parameters
    ----------
    df : DataFrame
        From prepare_echo_chamber_data()
        
    Returns
    -------
    dict with cell_means, main_effects, interaction, sample_sizes
    """
    # Cell means
    mean_table = df.pivot_table(
        values='rating',
        index='paper_type',
        columns='reviewer_type',
        aggfunc='mean'
    )
    
    # Extract values (handle missing cells)
    try:
        HH = mean_table.loc['Human Paper', 'Human Review']
        HA = mean_table.loc['Human Paper', 'AI Review']
        AH = mean_table.loc['AI Paper', 'Human Review']
        AA = mean_table.loc['AI Paper', 'AI Review']
    except KeyError as e:
        raise ValueError(f"Missing cell in 2×2 design: {e}")
    
    # Main effects
    main_paper = ((AH + AA) - (HH + HA)) / 2
    main_reviewer = ((HA + AA) - (HH + AH)) / 2
    
    # Interaction
    interaction = (AA - AH) - (HA - HH)
    
    # Sample sizes
    count_table = pd.crosstab(df['paper_type'], df['reviewer_type'])
    
    return {
        'cell_means': {
            'HH': HH, 'HA': HA, 'AH': AH, 'AA': AA
        },
        'mean_table': mean_table,
        'main_effect_paper': main_paper,
        'main_effect_reviewer': main_reviewer,
        'interaction': interaction,
        'count_table': count_table,
        'n_total': len(df)
    }


def test_interaction_comprehensive(df: pd.DataFrame, verbose: bool = True) -> Dict:
    """
    Comprehensive statistical testing of interaction effect.
    
    Includes:
    1. OLS with clustered SEs
    2. OLS with robust (HC3) SEs
    3. Permutation test
    4. Bootstrap CI
    
    Parameters
    ----------
    df : DataFrame
        From prepare_echo_chamber_data()
    verbose : bool
        Print results
        
    Returns
    -------
    dict with all test results
    """
    results = {}
    
    # Get cell data
    hh = get_cell_data(df, 'Human Paper', 'Human Review')
    ha = get_cell_data(df, 'Human Paper', 'AI Review')
    ah = get_cell_data(df, 'AI Paper', 'Human Review')
    aa = get_cell_data(df, 'AI Paper', 'AI Review')
    
    # Basic interaction
    basic = compute_interaction_effect(df)
    results.update(basic)
    
    if verbose:
        print("\n" + "="*70)
        print("INTERACTION EFFECT ANALYSIS")
        print("="*70)
        print(f"\nSample: {len(df):,} reviews")
        print(f"\nCell counts:")
        print(basic['count_table'].to_string())
        print(f"\nCell means:")
        print(basic['mean_table'].round(3).to_string())
        print(f"\nMain effect (Paper): {basic['main_effect_paper']:+.4f}")
        print(f"Main effect (Reviewer): {basic['main_effect_reviewer']:+.4f}")
        print(f"\n*** INTERACTION: {basic['interaction']:+.4f} ***")
    
    # Check minimum cell sizes
    min_cell = basic['count_table'].min().min()
    if min_cell < MIN_CELL_SIZE:
        warnings.warn(f"Small cell size ({min_cell}). Results may be unreliable.")
    
    # =========================================================================
    # TEST 1: OLS with Clustered SEs
    # =========================================================================
    if verbose:
        print("\n" + "-"*70)
        print("TEST 1: OLS with Clustered Standard Errors")
        print("-"*70)
    
    try:
        ols_cl = ols_with_clustered_se(
            df, 
            'rating ~ paper_AI * reviewer_AI',
            'submission_number'
        )
        results['ols_clustered'] = ols_cl
        
        if verbose:
            print(f"\nInteraction coefficient: {ols_cl['params']['paper_AI:reviewer_AI']:.4f}")
            print(f"Clustered SE: {ols_cl['se']['paper_AI:reviewer_AI']:.4f}")
            print(f"p-value: {ols_cl['p_values']['paper_AI:reviewer_AI']:.4e}")
            print(f"95% CI: [{ols_cl['ci_lower']['paper_AI:reviewer_AI']:.4f}, "
                  f"{ols_cl['ci_upper']['paper_AI:reviewer_AI']:.4f}]")
            
    except Exception as e:
        if verbose:
            print(f"Clustered OLS failed: {e}")
        results['ols_clustered'] = None
    
    # =========================================================================
    # TEST 2: OLS with Robust SEs (HC3)
    # =========================================================================
    if verbose:
        print("\n" + "-"*70)
        print("TEST 2: OLS with Robust (HC3) Standard Errors")
        print("-"*70)
    
    try:
        ols_hc3 = ols_robust(df, 'rating ~ paper_AI * reviewer_AI', 'HC3')
        results['ols_robust'] = ols_hc3
        
        if verbose:
            print(f"\nInteraction coefficient: {ols_hc3['params']['paper_AI:reviewer_AI']:.4f}")
            print(f"HC3 SE: {ols_hc3['se']['paper_AI:reviewer_AI']:.4f}")
            print(f"p-value: {ols_hc3['p_values']['paper_AI:reviewer_AI']:.4e}")
            
    except Exception as e:
        if verbose:
            print(f"Robust OLS failed: {e}")
        results['ols_robust'] = None
    
    # =========================================================================
    # TEST 3: Permutation Test
    # =========================================================================
    if verbose:
        print("\n" + "-"*70)
        print("TEST 3: Permutation Test (10,000 iterations)")
        print("-"*70)
    
    perm_results = permutation_test_interaction(
        hh, ha, ah, aa,
        n_permutations=N_PERMUTATIONS,
        random_state=RANDOM_SEED
    )
    results['permutation'] = perm_results
    
    if verbose:
        print(f"\nObserved interaction: {perm_results['observed_interaction']:.4f}")
        print(f"Null mean: {perm_results['null_mean']:.4f}")
        print(f"Null std: {perm_results['null_std']:.4f}")
        print(f"Permutation p-value: {perm_results['p_value']:.4f}")
    
    # =========================================================================
    # TEST 4: Bootstrap CI
    # =========================================================================
    if verbose:
        print("\n" + "-"*70)
        print("TEST 4: Bootstrap Confidence Interval")
        print("-"*70)
    
    # Bootstrap for each cell difference
    ai_premium_ai_papers = bootstrap_diff_ci(aa, ah)  # AI reviewer premium on AI papers
    ai_premium_human_papers = bootstrap_diff_ci(ha, hh)  # AI reviewer premium on human papers
    
    results['bootstrap_ai_papers'] = ai_premium_ai_papers
    results['bootstrap_human_papers'] = ai_premium_human_papers
    
    if verbose:
        print(f"\nAI reviewer premium on AI papers:")
        print(f"  Diff: {ai_premium_ai_papers['diff']:+.4f}")
        print(f"  95% CI: [{ai_premium_ai_papers['ci_lower']:.4f}, {ai_premium_ai_papers['ci_upper']:.4f}]")
        
        print(f"\nAI reviewer premium on Human papers:")
        print(f"  Diff: {ai_premium_human_papers['diff']:+.4f}")
        print(f"  95% CI: [{ai_premium_human_papers['ci_lower']:.4f}, {ai_premium_human_papers['ci_upper']:.4f}]")
    
    # =========================================================================
    # Effect Size
    # =========================================================================
    effect_size = basic['interaction'] / df['rating'].std()
    results['cohens_d'] = effect_size
    
    if verbose:
        print(f"\n" + "-"*70)
        print("EFFECT SIZE")
        print("-"*70)
        print(f"Cohen's d (interaction / SD): {effect_size:.4f}")
        
        if abs(effect_size) < 0.2:
            print("  → Small effect")
        elif abs(effect_size) < 0.5:
            print("  → Small-to-medium effect")
        elif abs(effect_size) < 0.8:
            print("  → Medium effect")
        else:
            print("  → Large effect")
    
    # =========================================================================
    # Summary
    # =========================================================================
    if verbose:
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        # Collect p-values
        p_ols_cl = results.get('ols_clustered', {})
        p_ols_cl = p_ols_cl.get('p_values', {}).get('paper_AI:reviewer_AI', np.nan) if p_ols_cl else np.nan
        p_perm = perm_results['p_value']
        
        print(f"\nInteraction effect: {basic['interaction']:+.4f}")
        print(f"\np-values:")
        print(f"  OLS (clustered): {p_ols_cl:.4e}")
        print(f"  Permutation: {p_perm:.4f}")
        
        sig_cl = p_ols_cl < ALPHA if not np.isnan(p_ols_cl) else False
        sig_perm = p_perm < ALPHA
        
        if sig_cl or sig_perm:
            direction = "HIGHER" if basic['interaction'] > 0 else "LOWER"
            print(f"\n✓ SIGNIFICANT INTERACTION DETECTED")
            print(f"  AI reviewers give RELATIVELY {direction} ratings to AI papers")
            print(f"  compared to how human reviewers rate them.")
        else:
            print(f"\n→ Interaction NOT statistically significant")
            if basic['interaction'] > 0:
                print(f"  Direction suggests AI reviewers may favor AI papers,")
                print(f"  but we cannot rule out chance.")
    
    return results


def run_robustness_checks(reviews_df: pd.DataFrame, 
                          submissions_df: pd.DataFrame,
                          verbose: bool = True) -> pd.DataFrame:
    """
    Run echo chamber analysis across multiple threshold specifications.
    
    Tests robustness to arbitrary cutoffs for AI Paper / Human Paper.
    
    Returns
    -------
    DataFrame with results for each specification
    """
    from src.data_loading import prepare_echo_chamber_data
    
    if verbose:
        print("\n" + "="*70)
        print("ROBUSTNESS CHECKS: Multiple Threshold Specifications")
        print("="*70)
    
    results = []
    
    for ai_th, hu_th, label in ROBUSTNESS_THRESHOLDS:
        if verbose:
            print(f"\n{'-'*50}")
            print(f"Threshold: {label} (AI≥{ai_th}%, Human≤{hu_th}%)")
            print(f"{'-'*50}")
        
        try:
            # Prepare data with this threshold
            clean = prepare_echo_chamber_data(
                reviews_df, submissions_df,
                ai_paper_threshold=ai_th,
                human_paper_threshold=hu_th
            )
            
            # Check cell sizes
            counts = pd.crosstab(clean['paper_type'], clean['reviewer_type'])
            n_aa = counts.loc['AI Paper', 'AI Review'] if 'AI Paper' in counts.index else 0
            
            if verbose:
                print(f"N = {len(clean):,}, AI-AI cell = {n_aa}")
            
            if n_aa < 5:
                if verbose:
                    print("  SKIPPING: Insufficient AI-AI cases")
                continue
            
            # Run tests
            basic = compute_interaction_effect(clean)
            
            # OLS with clustered SE
            try:
                ols_cl = ols_with_clustered_se(
                    clean, 'rating ~ paper_AI * reviewer_AI', 'submission_number'
                )
                p_ols = ols_cl['p_values']['paper_AI:reviewer_AI']
                se_ols = ols_cl['se']['paper_AI:reviewer_AI']
            except:
                p_ols = np.nan
                se_ols = np.nan
            
            # Permutation test (fewer iterations for speed)
            hh = get_cell_data(clean, 'Human Paper', 'Human Review')
            ha = get_cell_data(clean, 'Human Paper', 'AI Review')
            ah = get_cell_data(clean, 'AI Paper', 'Human Review')
            aa = get_cell_data(clean, 'AI Paper', 'AI Review')
            
            perm = permutation_test_interaction(hh, ha, ah, aa, n_permutations=5000)
            
            # Effect size
            d = basic['interaction'] / clean['rating'].std()
            
            results.append({
                'Threshold': label,
                'AI_threshold': ai_th,
                'Human_threshold': hu_th,
                'N': len(clean),
                'N_AA': n_aa,
                'Interaction': basic['interaction'],
                'SE': se_ols,
                'p_OLS': p_ols,
                'p_Perm': perm['p_value'],
                'Cohens_d': d,
                'Significant': (p_ols < ALPHA) or (perm['p_value'] < ALPHA)
            })
            
            if verbose:
                sig = "✓" if results[-1]['Significant'] else "✗"
                print(f"  Interaction: {basic['interaction']:+.4f}, p={min(p_ols, perm['p_value']):.4f} {sig}")
            
        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")
    
    results_df = pd.DataFrame(results)
    
    if verbose and len(results_df) > 0:
        print("\n" + "="*70)
        print("ROBUSTNESS SUMMARY")
        print("="*70)
        print(results_df.round(4).to_string(index=False))
        
        n_sig = results_df['Significant'].sum()
        n_total = len(results_df)
        print(f"\nSignificant in {n_sig}/{n_total} specifications")
        
        if n_sig == n_total:
            print("✓ ROBUST: Effect significant across ALL specifications")
        elif n_sig >= n_total / 2:
            print("→ PARTIALLY ROBUST: Effect significant in most specifications")
        else:
            print("→ NOT ROBUST: Effect depends on threshold choice")
    
    return results_df


def analyze_additional_effects(df: pd.DataFrame, verbose: bool = True) -> Dict:
    """
    Additional analyses beyond the main interaction.
    
    1. AI reviewer leniency (main effect)
    2. Detection asymmetry
    3. Variance patterns
    4. Confidence patterns (if available)
    """
    results = {}
    
    if verbose:
        print("\n" + "="*70)
        print("ADDITIONAL ANALYSES")
        print("="*70)
    
    # Get cell data
    hh = get_cell_data(df, 'Human Paper', 'Human Review')
    ha = get_cell_data(df, 'Human Paper', 'AI Review')
    ah = get_cell_data(df, 'AI Paper', 'Human Review')
    aa = get_cell_data(df, 'AI Paper', 'AI Review')
    
    # =========================================================================
    # 1. AI Reviewer Leniency
    # =========================================================================
    if verbose:
        print("\n" + "-"*50)
        print("1. AI REVIEWER LENIENCY (Overall)")
        print("-"*50)
    
    ai_reviews = np.concatenate([ha, aa])
    human_reviews = np.concatenate([hh, ah])
    
    leniency = comprehensive_comparison(ai_reviews, human_reviews, 'AI Review', 'Human Review')
    results['leniency'] = leniency
    
    if verbose:
        print(f"\nAI Review mean: {leniency['mean1']:.3f}")
        print(f"Human Review mean: {leniency['mean2']:.3f}")
        print(f"Difference: {leniency['mean_diff']:+.3f}")
        print(f"Mann-Whitney p = {leniency['mann_whitney_p']:.4e}")
        print(f"Permutation p = {leniency['permutation_p']:.4f}")
        print(f"Cohen's d = {leniency['cohens_d']:.3f}")
    
    # =========================================================================
    # 2. Detection Asymmetry
    # =========================================================================
    if verbose:
        print("\n" + "-"*50)
        print("2. DETECTION ASYMMETRY")
        print("-"*50)
        print("Do human reviewers penalize AI papers more than AI reviewers do?")
    
    human_penalty = np.mean(hh) - np.mean(ah)  # Rating drop for AI papers (human reviewers)
    ai_penalty = np.mean(ha) - np.mean(aa)     # Rating drop for AI papers (AI reviewers)
    
    results['human_penalty'] = human_penalty
    results['ai_penalty'] = ai_penalty
    results['detection_asymmetry'] = human_penalty - ai_penalty
    
    if verbose:
        print(f"\nHuman reviewer penalty for AI papers: {human_penalty:+.3f}")
        print(f"AI reviewer penalty for AI papers: {ai_penalty:+.3f}")
        print(f"Detection asymmetry: {human_penalty - ai_penalty:+.3f}")
        
        if human_penalty > ai_penalty:
            print("→ Human reviewers MORE sensitive to AI papers")
        else:
            print("→ AI reviewers MORE sensitive to AI papers")
    
    # =========================================================================
    # 3. Variance Patterns
    # =========================================================================
    if verbose:
        print("\n" + "-"*50)
        print("3. VARIANCE PATTERNS")
        print("-"*50)
        print("Is there less discrimination in the AI echo chamber?")
    
    var_hh = np.var(hh, ddof=1)
    var_aa = np.var(aa, ddof=1)
    
    levene_result = levene_test(hh, aa)
    results['variance'] = {
        'var_HH': var_hh,
        'var_AA': var_aa,
        'var_ratio': var_aa / var_hh if var_hh > 0 else np.nan,
        'levene_p': levene_result['p_value']
    }
    
    if verbose:
        print(f"\nHuman-Human variance: {var_hh:.3f}")
        print(f"AI-AI variance: {var_aa:.3f}")
        print(f"Ratio (AA/HH): {var_aa/var_hh:.3f}")
        print(f"Levene's test p = {levene_result['p_value']:.4f}")
        
        if var_aa < var_hh and levene_result['p_value'] < 0.05:
            print("→ AI-AI shows LESS variance (less discrimination)")
        elif var_aa > var_hh and levene_result['p_value'] < 0.05:
            print("→ AI-AI shows MORE variance")
    
    # =========================================================================
    # 4. Confidence Patterns (if available)
    # =========================================================================
    if 'confidence' in df.columns:
        if verbose:
            print("\n" + "-"*50)
            print("4. CONFIDENCE PATTERNS")
            print("-"*50)
        
        conf_table = df.pivot_table(
            values='confidence',
            index='paper_type',
            columns='reviewer_type',
            aggfunc='mean'
        )
        results['confidence_table'] = conf_table
        
        if verbose:
            print("\nMean Confidence by Cell:")
            print(conf_table.round(3).to_string())
    
    return results


def create_echo_chamber_figure(df: pd.DataFrame, results: Dict,
                                save_path: Optional[str] = None) -> plt.Figure:
    """
    Create comprehensive 6-panel echo chamber visualization.
    """
    setup_style()
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    mean_table = results['mean_table']
    count_table = results['count_table']
    
    # Panel 1: Heatmap of means
    ax1 = axes[0, 0]
    plot_heatmap(mean_table, ax=ax1, title='A. Mean Ratings', cmap='RdYlGn')
    
    # Panel 2: Sample sizes
    ax2 = axes[0, 1]
    sns.heatmap(count_table, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_title('B. Sample Sizes')
    
    # Panel 3: Interaction plot
    ax3 = axes[0, 2]
    plot_interaction_2x2(mean_table, ax=ax3, title='C. Interaction Plot')
    
    # Add interaction value
    interaction = results['interaction']
    p_val = results.get('permutation', {}).get('p_value', np.nan)
    sig = '*' if p_val < 0.05 else ''
    ax3.text(0.95, 0.05, f'Interaction: {interaction:+.3f}{sig}',
             transform=ax3.transAxes, ha='right', fontsize=11,
             bbox=dict(facecolor='wheat', alpha=0.8))
    
    # Panel 4: Box plots by cell
    ax4 = axes[1, 0]
    order = ['Human Paper + Human Review', 'Human Paper + AI Review',
             'AI Paper + Human Review', 'AI Paper + AI Review']
    available = [o for o in order if o in df['match_type'].values]
    
    colors = [ECHO_CHAMBER_COLORS.get(o, 'gray') for o in available]
    
    sns.boxplot(data=df, x='match_type', y='rating', order=available,
                palette=colors, ax=ax4)
    ax4.set_xlabel('')
    ax4.set_ylabel('Rating')
    ax4.set_title('D. Rating Distribution by Cell')
    ax4.tick_params(axis='x', rotation=20)
    
    # Panel 5: Permutation distribution
    ax5 = axes[1, 1]
    if 'permutation' in results and 'null_distribution' in results['permutation']:
        perm = results['permutation']
        plot_permutation_distribution(
            perm['observed_interaction'],
            perm['null_distribution'],
            ax=ax5,
            title='E. Permutation Test'
        )
    else:
        ax5.text(0.5, 0.5, 'Permutation results\nnot available',
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('E. Permutation Test')
    
    # Panel 6: Effect summary
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Create summary text
    summary_text = [
        "ECHO CHAMBER ANALYSIS SUMMARY",
        "=" * 35,
        "",
        f"Interaction Effect: {results['interaction']:+.4f}",
        f"Main Effect (Paper): {results['main_effect_paper']:+.4f}",
        f"Main Effect (Reviewer): {results['main_effect_reviewer']:+.4f}",
        "",
        "Interpretation:",
    ]
    
    if results['interaction'] > 0:
        summary_text.append("AI reviewers give relatively HIGHER")
        summary_text.append("ratings to AI papers compared to")
        summary_text.append("how human reviewers rate them.")
    else:
        summary_text.append("AI reviewers give relatively LOWER")
        summary_text.append("ratings to AI papers compared to")
        summary_text.append("how human reviewers rate them.")
    
    ax6.text(0.1, 0.9, '\n'.join(summary_text), transform=ax6.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(facecolor='lightyellow', alpha=0.8, edgecolor='gray'))
    ax6.set_title('F. Summary')
    
    plt.suptitle('AI Echo Chamber Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def run_echo_chamber_analysis(submissions_path_or_df, reviews_path_or_df=None,
                               ai_paper_threshold: int = 75,
                               human_paper_threshold: int = 25,
                               run_robustness: bool = True,
                               save_figures: bool = True,
                               output_dir: str = '.') -> Dict:
    """
    Run complete echo chamber analysis.
    
    Parameters
    ----------
    submissions_path_or_df : str or DataFrame
    reviews_path_or_df : str or DataFrame
    ai_paper_threshold : int
    human_paper_threshold : int
    run_robustness : bool
        Whether to run robustness checks across thresholds
    save_figures : bool
    output_dir : str
    
    Returns
    -------
    dict with all results
    """
    # Load data
    if isinstance(submissions_path_or_df, str):
        submissions_df, reviews_df = load_data(submissions_path_or_df, reviews_path_or_df)
    else:
        submissions_df = submissions_path_or_df
        reviews_df = reviews_path_or_df
    
    print("="*70)
    print("AI ECHO CHAMBER ANALYSIS")
    print("="*70)
    print(f"\nData: {len(submissions_df):,} submissions, {len(reviews_df):,} reviews")
    
    # Prepare 2×2 data
    clean = prepare_echo_chamber_data(
        reviews_df, submissions_df,
        ai_paper_threshold=ai_paper_threshold,
        human_paper_threshold=human_paper_threshold
    )
    
    print(f"Clean 2×2 sample: {len(clean):,} reviews")
    
    # Check ICC for clustering
    icc = compute_icc(clean, 'rating', 'submission_number')
    print(f"ICC (clustering): {icc:.3f}")
    if icc > 0.1:
        print("  → Substantial clustering. Using clustered SEs.")
    
    # Main interaction test
    results = test_interaction_comprehensive(clean, verbose=True)
    
    # Additional analyses
    additional = analyze_additional_effects(clean, verbose=True)
    results['additional'] = additional
    
    # Robustness checks
    if run_robustness:
        robustness_df = run_robustness_checks(reviews_df, submissions_df, verbose=True)
        results['robustness'] = robustness_df
    
    # Figures
    if save_figures:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        fig = create_echo_chamber_figure(clean, results)
        save_figure(fig, os.path.join(output_dir, 'fig_echo_chamber.png'))
        
        # Robustness figure
        if run_robustness and len(results.get('robustness', [])) > 0:
            rob_df = results['robustness']
            
            fig2, ax = plt.subplots(figsize=(10, 6))
            x = range(len(rob_df))
            colors = ['green' if s else 'gray' for s in rob_df['Significant']]
            
            ax.bar(x, rob_df['Interaction'], color=colors, edgecolor='black', alpha=0.7)
            ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
            
            for i, row in rob_df.iterrows():
                sig = '*' if row['Significant'] else ''
                ax.text(i, row['Interaction'] + 0.02, sig, ha='center', fontsize=14)
            
            ax.set_xticks(x)
            ax.set_xticklabels(rob_df['Threshold'], rotation=15)
            ax.set_ylabel('Interaction Effect')
            ax.set_title('Robustness: Interaction Across Threshold Specifications')
            
            save_figure(fig2, os.path.join(output_dir, 'fig_robustness.png'))
    
    return results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) >= 3:
        results = run_echo_chamber_analysis(sys.argv[1], sys.argv[2])
    else:
        print("""
AI Echo Chamber Analysis
========================

Usage:
    python echo_chamber.py submissions.csv reviews.csv

Or in Python:
    from analysis.echo_chamber import run_echo_chamber_analysis
    results = run_echo_chamber_analysis(submissions_df, reviews_df)
""")
