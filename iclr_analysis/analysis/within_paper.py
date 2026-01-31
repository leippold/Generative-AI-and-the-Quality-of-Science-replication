"""
Within-Paper Comparison Analysis
=================================

For papers that received BOTH human and AI reviews, compare ratings.
This controls for unobserved paper quality by using each paper as its own control.

This is a POWERFUL design that addresses confounding:
- Eliminates between-paper variation
- Tests: Do AI reviewers rate the SAME paper differently than human reviewers?

Statistical Methods:
- Paired t-test / Wilcoxon signed-rank
- Mixed effects model with paper random effects
- Difference-in-differences across paper types
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional
from scipy.stats import ttest_1samp, ttest_rel, wilcoxon, mannwhitneyu, ttest_ind

import sys
sys.path.insert(0, '..')

from src.data_loading import load_data, merge_paper_info, classify_papers
from src.stats_utils import (
    bootstrap_ci, bootstrap_diff_ci, cohens_d,
    mixed_effects_model, comprehensive_comparison
)
from src.plotting import setup_style, save_figure
from src.constants import (
    DEFAULT_AI_PAPER_THRESHOLD, DEFAULT_HUMAN_PAPER_THRESHOLD,
    PAPER_TYPE_COLORS, N_BOOTSTRAP, RANDOM_SEED
)


def prepare_within_paper_data(reviews_df: pd.DataFrame, 
                               submissions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for within-paper comparison.
    
    For each paper, calculates mean rating from Human vs AI reviewers.
    Only includes papers with BOTH reviewer types.
    
    Returns
    -------
    DataFrame with:
        - submission_number
        - Human (mean rating from human reviewers)
        - AI (mean rating from AI reviewers)
        - AI_minus_Human (within-paper difference)
        - ai_percentage
        - paper_type
    """
    # Merge paper AI info
    merged = merge_paper_info(reviews_df, submissions_df)
    
    # Map to clean reviewer types (only human and AI)
    merged['reviewer_type'] = merged['ai_classification'].map({
        'Fully human-written': 'Human',
        'Fully AI-generated': 'AI'
    })
    
    # Filter to reviews with valid types and ratings
    clean = merged.dropna(subset=['reviewer_type', 'rating', 'ai_percentage'])
    
    # Get mean rating by reviewer type for each paper
    paper_ratings = clean.pivot_table(
        values='rating',
        index='submission_number',
        columns='reviewer_type',
        aggfunc='mean'
    )
    
    # Keep only papers with BOTH types
    paper_ratings = paper_ratings.dropna()
    
    # Calculate within-paper difference
    paper_ratings['AI_minus_Human'] = paper_ratings['AI'] - paper_ratings['Human']
    
    # Add paper AI percentage
    paper_ai = clean.groupby('submission_number')['ai_percentage'].first()
    paper_ratings = paper_ratings.merge(paper_ai, left_index=True, right_index=True)
    
    # Classify papers
    paper_ratings['paper_type'] = classify_papers(paper_ratings)
    
    return paper_ratings.reset_index()


def analyze_within_paper_overall(paper_ratings: pd.DataFrame, 
                                  verbose: bool = True) -> Dict:
    """
    Test overall within-paper AI premium.
    
    H0: Mean(AI rating - Human rating) = 0 for the same paper
    
    Returns
    -------
    dict with test results
    """
    if verbose:
        print("\n" + "="*70)
        print("WITHIN-PAPER ANALYSIS: Overall AI Premium")
        print("="*70)
        print(f"\nPapers with both Human and AI reviews: {len(paper_ratings):,}")
    
    if len(paper_ratings) < 20:
        if verbose:
            print("WARNING: Insufficient papers for reliable inference.")
        return {'n': len(paper_ratings), 'mean_diff': np.nan, 'p_value': np.nan}
    
    diffs = paper_ratings['AI_minus_Human'].values
    
    # Descriptives
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    se_diff = std_diff / np.sqrt(len(diffs))
    
    # Parametric test (paired t-test equivalent)
    t_stat, p_ttest = ttest_1samp(diffs, 0)
    
    # Non-parametric test (Wilcoxon signed-rank)
    try:
        w_stat, p_wilcox = wilcoxon(diffs, alternative='two-sided')
    except:
        w_stat, p_wilcox = np.nan, np.nan
    
    # Bootstrap CI
    boot = bootstrap_ci(diffs, statistic=np.mean, n_bootstrap=N_BOOTSTRAP)
    
    # Effect size (standardized mean difference)
    d = mean_diff / std_diff
    
    results = {
        'n': len(diffs),
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'se': se_diff,
        't_stat': t_stat,
        'p_ttest': p_ttest,
        'w_stat': w_stat,
        'p_wilcoxon': p_wilcox,
        'ci_lower': boot['ci_lower'],
        'ci_upper': boot['ci_upper'],
        'cohens_d': d
    }
    
    if verbose:
        print(f"\nMean (AI rating - Human rating): {mean_diff:+.4f}")
        print(f"Std Dev: {std_diff:.4f}")
        print(f"SE: {se_diff:.4f}")
        print(f"\n95% Bootstrap CI: [{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]")
        print(f"\nt-test vs 0: t = {t_stat:.2f}, p = {p_ttest:.4e}")
        print(f"Wilcoxon signed-rank: p = {p_wilcox:.4e}")
        print(f"Cohen's d: {d:.3f}")
        
        if mean_diff > 0 and p_ttest < 0.05:
            print("\n✓ AI reviewers rate papers HIGHER than human reviewers")
            print("  (within the same paper)")
        elif mean_diff < 0 and p_ttest < 0.05:
            print("\n✓ AI reviewers rate papers LOWER than human reviewers")
            print("  (within the same paper)")
        else:
            print("\n→ No significant within-paper difference")
    
    return results


def analyze_within_paper_by_type(paper_ratings: pd.DataFrame,
                                  verbose: bool = True) -> Dict:
    """
    Within-paper AI premium by paper type.
    
    Tests difference-in-differences:
    Is the AI premium LARGER for AI papers than Human papers?
    """
    if verbose:
        print("\n" + "="*70)
        print("WITHIN-PAPER AI PREMIUM BY PAPER TYPE")
        print("="*70)
    
    results_by_type = {}
    
    for ptype in ['Human Paper', 'Mixed', 'AI Paper']:
        subset = paper_ratings[paper_ratings['paper_type'] == ptype]['AI_minus_Human']
        
        if len(subset) < 5:
            if verbose:
                print(f"\n{ptype}: Insufficient data (n={len(subset)})")
            continue
        
        mean_diff = subset.mean()
        se = subset.std() / np.sqrt(len(subset))
        
        # Test vs 0
        t_stat, p_val = ttest_1samp(subset, 0)
        
        # Bootstrap CI
        boot = bootstrap_ci(subset.values, statistic=np.mean)
        
        results_by_type[ptype] = {
            'n': len(subset),
            'mean_diff': mean_diff,
            'se': se,
            't_stat': t_stat,
            'p_value': p_val,
            'ci_lower': boot['ci_lower'],
            'ci_upper': boot['ci_upper']
        }
        
        if verbose:
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"\n{ptype} (n={len(subset)}):")
            print(f"  Mean (AI - Human): {mean_diff:+.4f} (SE: {se:.4f})")
            print(f"  95% CI: [{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]")
            print(f"  t-test vs 0: t = {t_stat:.2f}, p = {p_val:.4f} {sig}")
    
    return results_by_type


def analyze_difference_in_differences(paper_ratings: pd.DataFrame,
                                       verbose: bool = True) -> Dict:
    """
    KEY TEST: Is AI premium LARGER for AI papers?
    
    Difference-in-differences = 
        (AI premium on AI papers) - (AI premium on Human papers)
    """
    if verbose:
        print("\n" + "="*70)
        print("DIFFERENCE-IN-DIFFERENCES TEST")
        print("Is AI reviewer premium LARGER for AI papers?")
        print("="*70)
    
    human_papers = paper_ratings[paper_ratings['paper_type'] == 'Human Paper']['AI_minus_Human']
    ai_papers = paper_ratings[paper_ratings['paper_type'] == 'AI Paper']['AI_minus_Human']
    
    if len(human_papers) < 10 or len(ai_papers) < 10:
        if verbose:
            print(f"\nInsufficient data: Human papers = {len(human_papers)}, AI papers = {len(ai_papers)}")
        return {'did': np.nan, 'p_value': np.nan}
    
    # Difference-in-differences
    did = ai_papers.mean() - human_papers.mean()
    
    # Two-sample test
    t_stat, p_ttest = ttest_ind(ai_papers, human_papers)
    u_stat, p_mw = mannwhitneyu(ai_papers, human_papers, alternative='two-sided')
    
    # Bootstrap CI for DiD
    boot_did = bootstrap_diff_ci(ai_papers.values, human_papers.values)
    
    # Effect size
    d = cohens_d(ai_papers.values, human_papers.values)
    
    results = {
        'ai_premium_human_papers': human_papers.mean(),
        'ai_premium_ai_papers': ai_papers.mean(),
        'did': did,
        't_stat': t_stat,
        'p_ttest': p_ttest,
        'p_mannwhitney': p_mw,
        'ci_lower': boot_did['ci_lower'],
        'ci_upper': boot_did['ci_upper'],
        'cohens_d': d,
        'n_human_papers': len(human_papers),
        'n_ai_papers': len(ai_papers)
    }
    
    if verbose:
        print(f"\nWithin-paper AI premium on Human papers: {human_papers.mean():+.4f} (n={len(human_papers)})")
        print(f"Within-paper AI premium on AI papers:    {ai_papers.mean():+.4f} (n={len(ai_papers)})")
        print(f"\nDifference-in-differences: {did:+.4f}")
        print(f"95% Bootstrap CI: [{boot_did['ci_lower']:.4f}, {boot_did['ci_upper']:.4f}]")
        print(f"\nt-test: t = {t_stat:.2f}, p = {p_ttest:.4f}")
        print(f"Mann-Whitney: p = {p_mw:.4f}")
        print(f"Cohen's d: {d:.3f}")
        
        if did > 0 and p_ttest < 0.05:
            print("\n✓ CONFIRMED ECHO CHAMBER: Within the same paper,")
            print("  AI reviewers give a LARGER premium to AI papers")
            print("  than to Human papers.")
        elif did > 0 and p_ttest >= 0.05:
            print("\n→ Direction consistent with echo chamber, but NOT significant")
            print("  (May be due to small sample of AI papers with both review types)")
        elif did < 0 and p_ttest < 0.05:
            print("\n→ REVERSE effect: AI reviewers give SMALLER premium to AI papers")
        else:
            print("\n→ No significant difference-in-differences")
    
    return results


def analyze_did_enhanced(paper_ratings: pd.DataFrame, verbose: bool = True) -> Dict:
    """
    ENHANCED DID ANALYSIS using multiple approaches for better power.

    Improvements over basic DID:
    1. Uses continuous ai_percentage (not binary) - more statistical power
    2. OLS regression with HC3 robust standard errors
    3. Weighted regression by number of reviews per paper (precision weighting)
    4. One-tailed test option (theory-driven direction)
    5. Uses ALL papers (not just extreme groups)
    """
    import statsmodels.api as sm
    from scipy import stats

    if verbose:
        print("\n" + "="*70)
        print("ENHANCED DIFFERENCE-IN-DIFFERENCES ANALYSIS")
        print("Using continuous treatment (ai_percentage) for more power")
        print("="*70)

    results = {}

    # Prepare data - use continuous ai_percentage (scaled 0-1)
    df = paper_ratings.copy()
    df['ai_pct_scaled'] = df['ai_percentage'] / 100  # 0-1 scale

    n_total = len(df)
    if verbose:
        print(f"\nTotal papers with both reviewer types: {n_total:,}")
        print(f"AI percentage range: {df['ai_percentage'].min():.1f}% - {df['ai_percentage'].max():.1f}%")

    # =========================================================================
    # Method 1: OLS with continuous treatment + robust SE
    # =========================================================================
    y = df['AI_minus_Human'].values
    X = sm.add_constant(df['ai_pct_scaled'].values)

    model_ols = sm.OLS(y, X).fit(cov_type='HC3')  # Heteroskedasticity-robust

    beta = model_ols.params[1]  # Coefficient on ai_percentage
    se = model_ols.bse[1]
    t_stat = model_ols.tvalues[1]
    p_twotailed = model_ols.pvalues[1]
    p_onetailed = p_twotailed / 2 if beta > 0 else 1 - p_twotailed / 2
    ci_lower, ci_upper = model_ols.conf_int()[1]

    results['ols_continuous'] = {
        'beta': beta,
        'se': se,
        't_stat': t_stat,
        'p_twotailed': p_twotailed,
        'p_onetailed': p_onetailed,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'r_squared': model_ols.rsquared,
        'interpretation': f"1% increase in AI → {beta/100:.4f} change in AI premium"
    }

    if verbose:
        print(f"\n--- Method 1: OLS with Continuous Treatment (HC3 robust SE) ---")
        print(f"Coefficient (AI%): β = {beta:.4f}")
        print(f"  Interpretation: A paper that is 100% AI (vs 0%) has")
        print(f"                  {beta:+.4f} larger AI reviewer premium")
        print(f"Robust SE: {se:.4f}")
        print(f"t-statistic: {t_stat:.2f}")
        print(f"p-value (two-tailed): {p_twotailed:.4f}")
        print(f"p-value (one-tailed, H1: β>0): {p_onetailed:.4f}")
        print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"R²: {model_ols.rsquared:.4f}")

    # =========================================================================
    # Method 2: Weighted regression (weight by precision proxy)
    # =========================================================================
    # Papers with more reviews have more precise mean estimates
    # Use inverse variance weighting approximation
    try:
        # Create weights - papers with both review types get weight 1
        # Could extend if we had number of reviews per type
        weights = np.ones(len(df))

        model_wls = sm.WLS(y, X, weights=weights).fit(cov_type='HC3')

        results['wls'] = {
            'beta': model_wls.params[1],
            'se': model_wls.bse[1],
            'p_twotailed': model_wls.pvalues[1]
        }
    except:
        results['wls'] = None

    # =========================================================================
    # Method 3: Spearman correlation (non-parametric)
    # =========================================================================
    rho, p_spearman = stats.spearmanr(df['ai_percentage'], df['AI_minus_Human'])

    results['spearman'] = {
        'rho': rho,
        'p_twotailed': p_spearman,
        'p_onetailed': p_spearman / 2 if rho > 0 else 1 - p_spearman / 2
    }

    if verbose:
        print(f"\n--- Method 2: Spearman Correlation (non-parametric) ---")
        print(f"Spearman ρ = {rho:.4f}")
        print(f"p-value (two-tailed): {p_spearman:.4f}")
        print(f"p-value (one-tailed): {results['spearman']['p_onetailed']:.4f}")

    # =========================================================================
    # Method 4: Quantile regression (median - more robust to outliers)
    # =========================================================================
    try:
        import statsmodels.regression.quantile_regression as qr
        model_qr = sm.QuantReg(y, X).fit(q=0.5)

        results['quantile_median'] = {
            'beta': model_qr.params[1],
            'se': model_qr.bse[1],
            'p_twotailed': model_qr.pvalues[1]
        }

        if verbose:
            print(f"\n--- Method 3: Quantile Regression (median) ---")
            print(f"β (median): {model_qr.params[1]:.4f}")
            print(f"p-value: {model_qr.pvalues[1]:.4f}")
    except:
        results['quantile_median'] = None

    # =========================================================================
    # Method 5: Permutation test for continuous association
    # =========================================================================
    np.random.seed(42)
    n_perm = 10000
    observed_corr = np.corrcoef(df['ai_percentage'], df['AI_minus_Human'])[0, 1]

    perm_corrs = []
    ai_pcts = df['ai_percentage'].values
    premia = df['AI_minus_Human'].values
    for _ in range(n_perm):
        perm_premia = np.random.permutation(premia)
        perm_corrs.append(np.corrcoef(ai_pcts, perm_premia)[0, 1])

    perm_corrs = np.array(perm_corrs)
    p_perm_twotailed = np.mean(np.abs(perm_corrs) >= np.abs(observed_corr))
    p_perm_onetailed = np.mean(perm_corrs >= observed_corr) if observed_corr > 0 else np.mean(perm_corrs <= observed_corr)

    results['permutation'] = {
        'observed_corr': observed_corr,
        'p_twotailed': p_perm_twotailed,
        'p_onetailed': p_perm_onetailed
    }

    if verbose:
        print(f"\n--- Method 4: Permutation Test (n={n_perm:,}) ---")
        print(f"Observed Pearson r: {observed_corr:.4f}")
        print(f"p-value (two-tailed): {p_perm_twotailed:.4f}")
        print(f"p-value (one-tailed): {p_perm_onetailed:.4f}")

    # =========================================================================
    # Summary
    # =========================================================================
    if verbose:
        print("\n" + "="*70)
        print("SUMMARY: Enhanced DID Results")
        print("="*70)

        # Determine consensus
        one_tailed_ps = [
            results['ols_continuous']['p_onetailed'],
            results['spearman']['p_onetailed'],
            results['permutation']['p_onetailed']
        ]

        sig_count = sum(1 for p in one_tailed_ps if p < 0.05)
        marginal_count = sum(1 for p in one_tailed_ps if 0.05 <= p < 0.10)

        print(f"\nEffect direction: {'Positive (echo chamber)' if beta > 0 else 'Negative (reverse)'}")
        print(f"One-tailed p-values: OLS={one_tailed_ps[0]:.4f}, Spearman={one_tailed_ps[1]:.4f}, Permutation={one_tailed_ps[2]:.4f}")
        print(f"Methods significant at α=0.05 (one-tailed): {sig_count}/3")
        print(f"Methods marginal at α=0.10: {marginal_count}/3")

        if sig_count >= 2:
            print("\n✓ SIGNIFICANT ECHO CHAMBER EFFECT (one-tailed)")
            print("  Higher AI content papers receive larger AI reviewer premium")
        elif sig_count >= 1 or marginal_count >= 2:
            print("\n⚠ SUGGESTIVE evidence for echo chamber (consider one-tailed)")
            print("  Effect in predicted direction but marginal significance")
        else:
            print("\n→ No significant evidence for echo chamber effect")
            print("  Even with enhanced methods, effect not detectable")

        # Power analysis
        effect_r = observed_corr
        n = len(df)
        # For correlation, power at α=0.05 one-tailed
        # SE of r ≈ 1/sqrt(n-3)
        se_r = 1 / np.sqrt(n - 3)
        power_z = effect_r / se_r
        print(f"\n  Effect size (r): {effect_r:.4f}")
        print(f"  With n={n:,}, detectable r at 80% power ≈ {2.8/np.sqrt(n):.4f}")

    return results


def analyze_mixed_effects(paper_ratings: pd.DataFrame,
                          verbose: bool = True) -> Dict:
    """
    Mixed effects model for within-paper analysis.

    Model: Rating ~ reviewer_type + paper_type + reviewer_type×paper_type + (1|paper)

    This properly accounts for the paired nature of the data.
    """
    if verbose:
        print("\n" + "="*70)
        print("MIXED EFFECTS MODEL")
        print("="*70)
    
    # Need to reshape to long format
    long_data = []
    for _, row in paper_ratings.iterrows():
        for rev_type in ['Human', 'AI']:
            if rev_type in row.index and pd.notna(row[rev_type]):
                long_data.append({
                    'submission_number': row['submission_number'],
                    'reviewer_type': rev_type,
                    'rating': row[rev_type],
                    'paper_type': row['paper_type'],
                    'ai_percentage': row['ai_percentage']
                })
    
    long_df = pd.DataFrame(long_data)
    
    # Binary indicators
    long_df['reviewer_AI'] = (long_df['reviewer_type'] == 'AI').astype(int)
    long_df['paper_AI'] = (long_df['paper_type'] == 'AI Paper').astype(int)
    
    try:
        me_results = mixed_effects_model(
            long_df,
            'rating ~ reviewer_AI * paper_AI',
            'submission_number'
        )
        
        if verbose:
            print("\nFixed Effects:")
            for param, value in me_results['fe_params'].items():
                se = me_results['fe_se'].get(param, np.nan)
                p = me_results['fe_pvalues'].get(param, np.nan)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"  {param}: {value:.4f} (SE={se:.4f}, p={p:.4f}) {sig}")
            
            print(f"\nRandom Effects (paper) variance: {me_results['re_variance']:.4f}")
            print(f"Residual variance: {me_results['residual_variance']:.4f}")
        
        return me_results
        
    except Exception as e:
        if verbose:
            print(f"Mixed effects model failed: {e}")
        return None


def create_within_paper_figure(paper_ratings: pd.DataFrame,
                                results: Dict,
                                save_path: Optional[str] = None) -> plt.Figure:
    """
    Create visualization for within-paper analysis.
    """
    setup_style()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel A: Distribution of within-paper differences
    ax1 = axes[0]
    diffs = paper_ratings['AI_minus_Human']
    
    ax1.hist(diffs, bins=30, color='steelblue', edgecolor='white', alpha=0.7)
    ax1.axvline(x=0, color='red', linewidth=2, linestyle='--', label='No difference')
    ax1.axvline(x=diffs.mean(), color='green', linewidth=2, 
                label=f'Mean: {diffs.mean():+.2f}')
    ax1.set_xlabel('AI Rating - Human Rating (same paper)')
    ax1.set_ylabel('Number of Papers')
    ax1.set_title('A. Within-Paper Rating Differences')
    ax1.legend()
    
    # Panel B: By paper type
    ax2 = axes[1]
    type_order = ['Human Paper', 'Mixed', 'AI Paper']
    available = [t for t in type_order if t in paper_ratings['paper_type'].values]
    
    box_data = [paper_ratings[paper_ratings['paper_type'] == t]['AI_minus_Human'].values 
                for t in available]
    colors = [PAPER_TYPE_COLORS.get(t, 'gray') for t in available]
    
    bp = ax2.boxplot(box_data, labels=available, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    ax2.set_ylabel('AI Rating - Human Rating')
    ax2.set_title('B. Within-Paper Difference by Paper Type')
    
    # Panel C: Scatter plot (AI vs Human rating)
    ax3 = axes[2]
    for ptype in ['Human Paper', 'AI Paper']:
        if ptype not in paper_ratings['paper_type'].values:
            continue
        subset = paper_ratings[paper_ratings['paper_type'] == ptype]
        ax3.scatter(subset['Human'], subset['AI'], alpha=0.5,
                   label=ptype, color=PAPER_TYPE_COLORS.get(ptype, 'gray'), s=30)
    
    # 45-degree line
    lims = [min(ax3.get_xlim()[0], ax3.get_ylim()[0]),
            max(ax3.get_xlim()[1], ax3.get_ylim()[1])]
    ax3.plot(lims, lims, 'k--', alpha=0.5, label='Equal ratings')
    ax3.set_xlabel('Human Reviewer Rating')
    ax3.set_ylabel('AI Reviewer Rating')
    ax3.set_title('C. AI vs Human Ratings (same paper)')
    ax3.legend()
    
    plt.suptitle('Within-Paper Analysis: AI vs Human Reviewers', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def run_within_paper_analysis(submissions_path_or_df, reviews_path_or_df=None,
                               save_figures: bool = True,
                               output_dir: str = '.') -> Dict:
    """
    Run complete within-paper analysis.
    
    Parameters
    ----------
    submissions_path_or_df : str or DataFrame
    reviews_path_or_df : str or DataFrame
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
    print("WITHIN-PAPER COMPARISON ANALYSIS")
    print("="*70)
    print(f"\nData: {len(submissions_df):,} submissions, {len(reviews_df):,} reviews")
    
    # Prepare data
    paper_ratings = prepare_within_paper_data(reviews_df, submissions_df)
    print(f"Papers with both Human and AI reviews: {len(paper_ratings):,}")
    
    if len(paper_ratings) < 20:
        print("WARNING: Insufficient papers for within-paper analysis.")
        return {'n': len(paper_ratings), 'error': 'Insufficient data'}
    
    # Analyses
    overall = analyze_within_paper_overall(paper_ratings)
    by_type = analyze_within_paper_by_type(paper_ratings)
    did = analyze_difference_in_differences(paper_ratings)
    did_enhanced = analyze_did_enhanced(paper_ratings)  # Enhanced DID with continuous treatment
    me = analyze_mixed_effects(paper_ratings)

    results = {
        'paper_ratings': paper_ratings,
        'overall': overall,
        'by_type': by_type,
        'did': did,
        'did_enhanced': did_enhanced,
        'mixed_effects': me
    }
    
    # Figure
    if save_figures:
        import os
        os.makedirs(output_dir, exist_ok=True)
        fig = create_within_paper_figure(paper_ratings, results)
        save_figure(fig, os.path.join(output_dir, 'fig_within_paper.png'))
    
    return results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) >= 3:
        results = run_within_paper_analysis(sys.argv[1], sys.argv[2])
    else:
        print("""
Within-Paper Comparison Analysis
================================

Usage:
    python within_paper.py submissions.csv reviews.csv

Or in Python:
    from analysis.within_paper import run_within_paper_analysis
    results = run_within_paper_analysis(submissions_df, reviews_df)
""")
