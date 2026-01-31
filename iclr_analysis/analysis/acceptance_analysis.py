"""
Acceptance Analysis Module for ICLR Papers.
============================================

Statistical tests for analyzing AI content effects on acceptance decisions.

Tests Implemented:
1. AI → Acceptance | Scores (Probit/Logit regression)
2. Threshold Discontinuity (RDD-style analysis near decision margin)
3. Presentation Tier Analysis (Ordered Probit among accepted papers)
4. Author Reputation × Acceptance Interaction
5. Selection Bounds Analysis

Research Questions:
- Does AI content predict acceptance conditional on review scores?
- Is AI content used as a tiebreaker near the decision margin?
- Are high-AI papers relegated to lower presentation tiers?
- Does author reputation insulate from AI content penalties?

METHODOLOGICAL NOTES:
- Acceptance is observed for ALL papers → no selection bias
- Author identity only observed for accepted + arXiv papers → selection issues
- Presentation tier analysis is conditional on acceptance → interpretation differs
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats as scipy_stats

import sys
sys.path.insert(0, '..')

from src.stats_utils import (
    mann_whitney_test, chi_square_test, bootstrap_ci, bootstrap_diff_ci,
    fdr_correction, cohens_d, hedges_g, cliffs_delta,
    ols_robust, permutation_test_two_groups
)
from src.constants import (
    ALPHA, N_BOOTSTRAP, N_PERMUTATIONS, RANDOM_SEED,
    DEFAULT_AI_PAPER_THRESHOLD, DEFAULT_HUMAN_PAPER_THRESHOLD
)
from src.data_loading import clean_ai_percentage, classify_papers


# =============================================================================
# CONSTANTS
# =============================================================================

# Default margin for threshold discontinuity analysis
DEFAULT_MARGIN_LOWER = 4.8
DEFAULT_MARGIN_UPPER = 5.2

# Tier encoding
TIER_ENCODING = {
    'Poster': 1,
    'Spotlight': 2,
    'Oral': 3,
    'Rejected': 0,
    'Withdrawn': -1,
    'Desk Rejected': -2
}


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_acceptance_data(df: pd.DataFrame,
                            ai_col: str = 'ai_percentage',
                            rating_col: str = 'avg_rating',
                            accepted_col: str = 'accepted',
                            tier_col: str = 'tier') -> pd.DataFrame:
    """
    Prepare data for acceptance analysis.

    Parameters
    ----------
    df : DataFrame
        Submissions data with AI percentage and acceptance status
    ai_col : str
        Column with AI content percentage
    rating_col : str
        Column with average rating
    accepted_col : str
        Column with acceptance status (boolean or 0/1)
    tier_col : str
        Column with presentation tier

    Returns
    -------
    DataFrame ready for analysis
    """
    data = df.copy()

    # Clean AI percentage
    data = clean_ai_percentage(data, ai_col)

    # Ensure acceptance is numeric
    if accepted_col in data.columns:
        data[accepted_col] = data[accepted_col].astype(float)

    # Encode tier if present
    if tier_col in data.columns and 'tier_name' in data.columns:
        data['tier_numeric'] = data['tier_name'].map(TIER_ENCODING)

    # Create binary AI indicator
    data['high_ai'] = (data[ai_col] >= DEFAULT_AI_PAPER_THRESHOLD).astype(int)
    data['low_ai'] = (data[ai_col] <= DEFAULT_HUMAN_PAPER_THRESHOLD).astype(int)

    # AI content categories
    data['ai_category'] = classify_papers(data, ai_col)

    # Standardized AI percentage (for regression)
    if data[ai_col].std() > 0:
        data['ai_percentage_std'] = (data[ai_col] - data[ai_col].mean()) / data[ai_col].std()
    else:
        data['ai_percentage_std'] = 0

    return data


# =============================================================================
# TEST 1: AI → ACCEPTANCE CONDITIONAL ON SCORES
# =============================================================================

def ai_acceptance_probit(df: pd.DataFrame,
                         controls: Optional[List[str]] = None,
                         verbose: bool = True) -> Dict:
    """
    Probit/Logit regression testing if AI content predicts acceptance
    conditional on review scores.

    Model: Pr(Accept=1) = Φ(α + β₁·AI% + β₂·avg_rating + X'γ)

    Key question: If β₁ < 0 after controlling for avg_rating,
    there's discrimination beyond the score channel.

    Parameters
    ----------
    df : DataFrame
        Must have: ai_percentage, avg_rating, accepted
    controls : list of str, optional
        Additional control variables
    verbose : bool
        Print results

    Returns
    -------
    dict with model results
    """
    try:
        from statsmodels.discrete.discrete_model import Probit, Logit
        import statsmodels.api as sm
    except ImportError:
        raise ImportError("statsmodels required for probit analysis")

    if verbose:
        print("\n" + "=" * 70)
        print("TEST 1: AI Content → Acceptance (Conditional on Scores)")
        print("=" * 70)

    # Prepare data
    required = ['ai_percentage', 'avg_rating', 'accepted']
    clean_df = df.dropna(subset=required)

    if controls:
        available_controls = [c for c in controls if c in df.columns]
        clean_df = clean_df.dropna(subset=available_controls)
    else:
        available_controls = []

    if len(clean_df) < 100:
        warnings.warn(f"Small sample size ({len(clean_df)})")

    if verbose:
        print(f"\nSample: {len(clean_df)} papers with decisions")
        print(f"Acceptance rate: {clean_df['accepted'].mean():.1%}")

    results = {'n': len(clean_df)}

    # Build feature matrix
    X_cols = ['ai_percentage', 'avg_rating'] + available_controls
    X = clean_df[X_cols].copy()
    X = sm.add_constant(X)
    y = clean_df['accepted']

    # =========================================================================
    # Model 1: Probit
    # =========================================================================
    if verbose:
        print("\n" + "-" * 50)
        print("Model 1: Probit Regression")
        print("-" * 50)

    try:
        probit_model = Probit(y, X)
        probit_result = probit_model.fit(disp=False, cov_type='HC3')

        results['probit'] = {
            'params': probit_result.params.to_dict(),
            'se': probit_result.bse.to_dict(),
            'p_values': probit_result.pvalues.to_dict(),
            'pseudo_rsquared': probit_result.prsquared,
            'aic': probit_result.aic,
            'bic': probit_result.bic,
            'model': probit_result
        }

        # Marginal effects at mean
        try:
            margeff = probit_result.get_margeff(at='mean')
            results['probit']['marginal_effects'] = {
                'params': margeff.margeff.tolist(),
                'se': margeff.margeff_se.tolist(),
                'p_values': margeff.pvalues.tolist()
            }
        except:
            pass

        if verbose:
            print(f"\nAI percentage coefficient: {probit_result.params['ai_percentage']:.6f}")
            print(f"  SE: {probit_result.bse['ai_percentage']:.6f}")
            print(f"  p-value: {probit_result.pvalues['ai_percentage']:.4e}")

            print(f"\navg_rating coefficient: {probit_result.params['avg_rating']:.4f}")
            print(f"  p-value: {probit_result.pvalues['avg_rating']:.4e}")

            print(f"\nPseudo R²: {probit_result.prsquared:.4f}")

            # Interpretation
            ai_coef = probit_result.params['ai_percentage']
            ai_p = probit_result.pvalues['ai_percentage']

            if ai_p < ALPHA:
                direction = "LOWER" if ai_coef < 0 else "HIGHER"
                print(f"\n*** SIGNIFICANT: AI content predicts {direction} acceptance")
                print(f"    controlling for review scores ***")
            else:
                print(f"\n→ AI content not significantly related to acceptance")
                print(f"  after controlling for scores")

    except Exception as e:
        if verbose:
            print(f"Probit failed: {e}")
        results['probit'] = None

    # =========================================================================
    # Model 2: Logit (for odds ratios)
    # =========================================================================
    if verbose:
        print("\n" + "-" * 50)
        print("Model 2: Logit Regression (Odds Ratios)")
        print("-" * 50)

    try:
        logit_model = Logit(y, X)
        logit_result = logit_model.fit(disp=False, cov_type='HC3')

        results['logit'] = {
            'params': logit_result.params.to_dict(),
            'se': logit_result.bse.to_dict(),
            'p_values': logit_result.pvalues.to_dict(),
            'odds_ratios': np.exp(logit_result.params).to_dict(),
            'or_ci_lower': np.exp(logit_result.conf_int()[0]).to_dict(),
            'or_ci_upper': np.exp(logit_result.conf_int()[1]).to_dict(),
            'model': logit_result
        }

        if verbose:
            ai_or = np.exp(logit_result.params['ai_percentage'])
            ai_or_ci = np.exp(logit_result.conf_int().loc['ai_percentage'])

            print(f"\nOdds Ratio per 1% AI content: {ai_or:.6f}")
            print(f"  95% CI: [{ai_or_ci[0]:.6f}, {ai_or_ci[1]:.6f}]")

            # Odds ratio per 10% increase
            or_10pct = ai_or ** 10
            print(f"\nOdds Ratio per 10% AI increase: {or_10pct:.4f}")

            # Odds ratio for 100% vs 0% AI
            or_100pct = ai_or ** 100
            print(f"Odds Ratio for 100% vs 0% AI: {or_100pct:.4f}")

    except Exception as e:
        if verbose:
            print(f"Logit failed: {e}")
        results['logit'] = None

    # =========================================================================
    # Model 3: Linear Probability Model (for interpretability)
    # =========================================================================
    if verbose:
        print("\n" + "-" * 50)
        print("Model 3: Linear Probability Model (OLS)")
        print("-" * 50)

    try:
        lpm = ols_robust(
            clean_df,
            f"accepted ~ ai_percentage + avg_rating + {' + '.join(available_controls)}" if available_controls
            else "accepted ~ ai_percentage + avg_rating"
        )
        results['lpm'] = lpm

        if verbose:
            ai_coef = lpm['params']['ai_percentage']
            ai_p = lpm['p_values']['ai_percentage']

            print(f"\nAI percentage coefficient: {ai_coef:.6f}")
            print(f"  SE: {lpm['se']['ai_percentage']:.6f}")
            print(f"  p-value: {ai_p:.4e}")
            print(f"\nInterpretation: Each 1% increase in AI content")
            print(f"  changes acceptance probability by {ai_coef:.4f} ({ai_coef*100:.2f} pp)")
            print(f"\n  100% AI vs 0% AI: {ai_coef*100:.2f} pp difference")

    except Exception as e:
        if verbose:
            print(f"LPM failed: {e}")
        results['lpm'] = None

    # =========================================================================
    # Summary
    # =========================================================================
    if verbose:
        print("\n" + "=" * 70)
        print("SUMMARY: AI → Acceptance | Scores")
        print("=" * 70)

        # Collect significance from all models
        sig_probit = (results.get('probit', {}) or {}).get('p_values', {}).get('ai_percentage', 1) < ALPHA
        sig_logit = (results.get('logit', {}) or {}).get('p_values', {}).get('ai_percentage', 1) < ALPHA
        sig_lpm = (results.get('lpm', {}) or {}).get('p_values', {}).get('ai_percentage', 1) < ALPHA

        n_sig = sum([sig_probit, sig_logit, sig_lpm])

        if n_sig >= 2:
            print("\n✓ CONSISTENT FINDING: AI content significantly predicts")
            print("  acceptance after controlling for review scores")
            print("  → Evidence of discrimination beyond the score channel")
        elif n_sig == 1:
            print("\n→ Mixed results: significance depends on model specification")
        else:
            print("\n→ No significant AI effect on acceptance after controlling for scores")
            print("  → Any AI penalty operates through the score channel, not direct discrimination")

    return results


def ai_acceptance_by_category(df: pd.DataFrame,
                              verbose: bool = True) -> Dict:
    """
    Compare acceptance rates by AI content category.

    Non-parametric comparison of acceptance rates for:
    - High AI papers (≥75%)
    - Low AI papers (≤25%)
    - Medium AI papers (25-75%)

    Parameters
    ----------
    df : DataFrame
        Must have: ai_percentage, accepted

    Returns
    -------
    dict with comparison results
    """
    if verbose:
        print("\n" + "=" * 70)
        print("AI Category → Acceptance (Unadjusted)")
        print("=" * 70)

    clean_df = df.dropna(subset=['ai_percentage', 'accepted'])

    # Create categories
    high_ai = clean_df[clean_df['ai_percentage'] >= DEFAULT_AI_PAPER_THRESHOLD]
    low_ai = clean_df[clean_df['ai_percentage'] <= DEFAULT_HUMAN_PAPER_THRESHOLD]
    medium_ai = clean_df[(clean_df['ai_percentage'] > DEFAULT_HUMAN_PAPER_THRESHOLD) &
                         (clean_df['ai_percentage'] < DEFAULT_AI_PAPER_THRESHOLD)]

    results = {
        'high_ai': {'n': len(high_ai), 'acceptance_rate': high_ai['accepted'].mean()},
        'low_ai': {'n': len(low_ai), 'acceptance_rate': low_ai['accepted'].mean()},
        'medium_ai': {'n': len(medium_ai), 'acceptance_rate': medium_ai['accepted'].mean()},
    }

    if verbose:
        print(f"\nHigh AI (≥{DEFAULT_AI_PAPER_THRESHOLD}%): {results['high_ai']['acceptance_rate']:.1%} "
              f"(n={results['high_ai']['n']})")
        print(f"Low AI (≤{DEFAULT_HUMAN_PAPER_THRESHOLD}%): {results['low_ai']['acceptance_rate']:.1%} "
              f"(n={results['low_ai']['n']})")
        print(f"Medium AI: {results['medium_ai']['acceptance_rate']:.1%} "
              f"(n={results['medium_ai']['n']})")

    # Chi-square test: High vs Low
    if len(high_ai) >= 5 and len(low_ai) >= 5:
        contingency = pd.crosstab(
            pd.concat([high_ai.assign(group='High AI'),
                       low_ai.assign(group='Low AI')])['group'],
            pd.concat([high_ai.assign(group='High AI'),
                       low_ai.assign(group='Low AI')])['accepted']
        )

        chi2_result = chi_square_test(contingency)
        results['chi2_high_vs_low'] = chi2_result

        if verbose:
            print(f"\nHigh AI vs Low AI:")
            print(f"  Difference: {results['high_ai']['acceptance_rate'] - results['low_ai']['acceptance_rate']:+.1%}")
            print(f"  χ² = {chi2_result['chi2']:.2f}, p = {chi2_result['p_value']:.4e}")
            print(f"  Cramér's V = {chi2_result['cramers_v']:.4f}")

    return results


# =============================================================================
# TEST 2: THRESHOLD DISCONTINUITY
# =============================================================================

def threshold_discontinuity(df: pd.DataFrame,
                            margin_lower: float = DEFAULT_MARGIN_LOWER,
                            margin_upper: float = DEFAULT_MARGIN_UPPER,
                            ai_threshold: float = None,
                            verbose: bool = True) -> Dict:
    """
    Analyze acceptance patterns near the decision margin.

    Among papers with scores in a narrow band (e.g., avg_rating ∈ [4.8, 5.2]):
    - What's the acceptance rate for high vs. low AI content?
    - Is AI content a tiebreaker?

    This is cleaner identification than full-sample regression.

    Parameters
    ----------
    df : DataFrame
        Must have: ai_percentage, avg_rating, accepted
    margin_lower : float
        Lower bound of margin window
    margin_upper : float
        Upper bound of margin window
    ai_threshold : float, optional
        Threshold for high/low AI (default: median in margin sample)
    verbose : bool
        Print results

    Returns
    -------
    dict with discontinuity analysis results
    """
    if verbose:
        print("\n" + "=" * 70)
        print("TEST 2: Threshold Discontinuity Analysis")
        print("=" * 70)
        print(f"Score window: [{margin_lower}, {margin_upper}]")

    # Filter to margin sample
    margin_df = df[
        (df['avg_rating'] >= margin_lower) &
        (df['avg_rating'] <= margin_upper) &
        (df['accepted'].notna()) &
        (df['ai_percentage'].notna())
    ].copy()

    if verbose:
        print(f"Papers in margin: {len(margin_df)}")
        print(f"Overall acceptance rate in margin: {margin_df['accepted'].mean():.1%}")

    if len(margin_df) < 20:
        warnings.warn(f"Small margin sample ({len(margin_df)}). Consider widening the window.")

    results = {
        'n_margin': len(margin_df),
        'margin_bounds': (margin_lower, margin_upper),
        'margin_acceptance_rate': margin_df['accepted'].mean(),
        'margin_mean_rating': margin_df['avg_rating'].mean(),
    }

    # AI threshold
    if ai_threshold is None:
        ai_threshold = margin_df['ai_percentage'].median()
    results['ai_threshold'] = ai_threshold

    # Split by AI content
    high_ai_margin = margin_df[margin_df['ai_percentage'] >= ai_threshold]
    low_ai_margin = margin_df[margin_df['ai_percentage'] < ai_threshold]

    results['high_ai'] = {
        'n': len(high_ai_margin),
        'mean_ai': high_ai_margin['ai_percentage'].mean(),
        'acceptance_rate': high_ai_margin['accepted'].mean(),
        'mean_rating': high_ai_margin['avg_rating'].mean()
    }
    results['low_ai'] = {
        'n': len(low_ai_margin),
        'mean_ai': low_ai_margin['ai_percentage'].mean(),
        'acceptance_rate': low_ai_margin['accepted'].mean(),
        'mean_rating': low_ai_margin['avg_rating'].mean()
    }

    if verbose:
        print(f"\nAI threshold: {ai_threshold:.1f}%")
        print(f"\nHigh AI (≥{ai_threshold:.0f}%): n={results['high_ai']['n']}")
        print(f"  Acceptance rate: {results['high_ai']['acceptance_rate']:.1%}")
        print(f"  Mean rating: {results['high_ai']['mean_rating']:.3f}")
        print(f"\nLow AI (<{ai_threshold:.0f}%): n={results['low_ai']['n']}")
        print(f"  Acceptance rate: {results['low_ai']['acceptance_rate']:.1%}")
        print(f"  Mean rating: {results['low_ai']['mean_rating']:.3f}")

    # Difference in acceptance rates
    acc_diff = results['high_ai']['acceptance_rate'] - results['low_ai']['acceptance_rate']
    results['acceptance_difference'] = acc_diff

    if verbose:
        print(f"\nDifference: {acc_diff:+.1%}")

    # Statistical tests
    if len(high_ai_margin) >= 5 and len(low_ai_margin) >= 5:
        # Chi-square test
        contingency = pd.crosstab(
            margin_df['ai_percentage'] >= ai_threshold,
            margin_df['accepted']
        )
        chi2_result = chi_square_test(contingency)
        results['chi2'] = chi2_result

        # Fisher's exact test (better for small samples)
        try:
            odds_ratio, fisher_p = scipy_stats.fisher_exact(contingency.values)
            results['fisher'] = {'odds_ratio': odds_ratio, 'p_value': fisher_p}
        except:
            results['fisher'] = None

        # Permutation test
        perm_result = permutation_test_two_groups(
            high_ai_margin['accepted'].values,
            low_ai_margin['accepted'].values,
            n_permutations=N_PERMUTATIONS
        )
        results['permutation'] = perm_result

        if verbose:
            print(f"\nStatistical Tests:")
            print(f"  χ² = {chi2_result['chi2']:.3f}, p = {chi2_result['p_value']:.4f}")
            if results['fisher']:
                print(f"  Fisher's exact: OR = {results['fisher']['odds_ratio']:.3f}, "
                      f"p = {results['fisher']['p_value']:.4f}")
            print(f"  Permutation p = {perm_result['p_value']:.4f}")

    # Rating balance check
    if verbose:
        print(f"\nBalance Check:")
        rating_diff = results['high_ai']['mean_rating'] - results['low_ai']['mean_rating']
        print(f"  Rating difference (High - Low AI): {rating_diff:+.4f}")
        if abs(rating_diff) > 0.05:
            print(f"  ⚠ Groups not perfectly balanced on rating")
        else:
            print(f"  ✓ Groups well-balanced on rating")

    # Interpretation
    if verbose:
        print("\n" + "-" * 50)
        print("INTERPRETATION")
        print("-" * 50)

        if results.get('chi2', {}).get('p_value', 1) < ALPHA:
            direction = "HIGHER" if acc_diff > 0 else "LOWER"
            print(f"\n✓ SIGNIFICANT: High-AI papers have {direction} acceptance")
            print(f"  rates near the margin, suggesting AI content is")
            print(f"  used as a TIEBREAKER by area chairs.")
        else:
            print(f"\n→ No significant difference in acceptance rates")
            print(f"  near the margin. AI content does not appear to")
            print(f"  be used as a tiebreaker.")

    return results


def optimal_bandwidth_selection(df: pd.DataFrame,
                                bandwidths: List[Tuple[float, float]] = None,
                                verbose: bool = True) -> pd.DataFrame:
    """
    Test threshold discontinuity across multiple bandwidth specifications.

    Parameters
    ----------
    df : DataFrame
    bandwidths : list of (lower, upper) tuples
        Different margin windows to test
    verbose : bool

    Returns
    -------
    DataFrame with results for each bandwidth
    """
    if bandwidths is None:
        # Default: range of bandwidths around typical ICLR threshold
        bandwidths = [
            (4.5, 5.5),  # Wide
            (4.6, 5.4),
            (4.7, 5.3),
            (4.8, 5.2),  # Narrow
            (4.9, 5.1),  # Very narrow
            (5.0, 5.5),  # Above threshold
            (4.5, 5.0),  # Below threshold
        ]

    if verbose:
        print("\n" + "=" * 70)
        print("Bandwidth Sensitivity Analysis")
        print("=" * 70)

    results = []
    for lower, upper in bandwidths:
        try:
            disc = threshold_discontinuity(df, lower, upper, verbose=False)

            results.append({
                'bandwidth': f"[{lower}, {upper}]",
                'lower': lower,
                'upper': upper,
                'n_margin': disc['n_margin'],
                'high_ai_rate': disc['high_ai']['acceptance_rate'],
                'low_ai_rate': disc['low_ai']['acceptance_rate'],
                'difference': disc['acceptance_difference'],
                'chi2_p': disc.get('chi2', {}).get('p_value', np.nan),
                'fisher_p': disc.get('fisher', {}).get('p_value', np.nan) if disc.get('fisher') else np.nan,
                'perm_p': disc.get('permutation', {}).get('p_value', np.nan),
                'significant': disc.get('chi2', {}).get('p_value', 1) < ALPHA
            })
        except Exception as e:
            if verbose:
                print(f"Failed for bandwidth [{lower}, {upper}]: {e}")

    results_df = pd.DataFrame(results)

    if verbose and len(results_df) > 0:
        print(results_df.to_string(index=False))

        n_sig = results_df['significant'].sum()
        print(f"\nSignificant in {n_sig}/{len(results_df)} bandwidth specifications")

    return results_df


# =============================================================================
# TEST 3: PRESENTATION TIER ANALYSIS
# =============================================================================

def presentation_tier_analysis(df: pd.DataFrame,
                               verbose: bool = True) -> Dict:
    """
    Ordered probit/logit for presentation tier among accepted papers.

    Model: Pr(Tier = k) = f(α_k + β·AI% + X'γ)

    If β < 0, high-AI papers are relegated to lower tiers (poster)
    despite comparable scores.

    Parameters
    ----------
    df : DataFrame
        Must have: ai_percentage, avg_rating, accepted, tier or tier_name

    Returns
    -------
    dict with ordinal regression results
    """
    try:
        from statsmodels.miscmodels.ordinal_model import OrderedModel
    except ImportError:
        raise ImportError("statsmodels required for ordered probit")

    if verbose:
        print("\n" + "=" * 70)
        print("TEST 3: Presentation Tier Analysis (Accepted Papers)")
        print("=" * 70)

    # Filter to accepted papers only
    accepted_df = df[df['accepted'] == 1].copy()

    # Ensure tier encoding
    if 'tier_numeric' not in accepted_df.columns:
        if 'tier_name' in accepted_df.columns:
            accepted_df['tier_numeric'] = accepted_df['tier_name'].map(TIER_ENCODING)
        elif 'tier' in accepted_df.columns:
            accepted_df['tier_numeric'] = accepted_df['tier']
        else:
            raise ValueError("No tier column found")

    # Filter to valid tiers (poster=1, spotlight=2, oral=3)
    tier_df = accepted_df[accepted_df['tier_numeric'].isin([1, 2, 3])].copy()
    tier_df = tier_df.dropna(subset=['ai_percentage', 'avg_rating', 'tier_numeric'])

    if verbose:
        print(f"\nAccepted papers with tier info: {len(tier_df)}")
        print(f"\nTier distribution:")
        tier_counts = tier_df['tier_numeric'].value_counts().sort_index()
        for tier, count in tier_counts.items():
            tier_name = {1: 'Poster', 2: 'Spotlight', 3: 'Oral'}.get(tier, str(tier))
            print(f"  {tier_name}: {count} ({100*count/len(tier_df):.1f}%)")

    if len(tier_df) < 50:
        warnings.warn(f"Small sample ({len(tier_df)}). Results may be unreliable.")

    results = {'n': len(tier_df)}

    # =========================================================================
    # Ordered Probit
    # =========================================================================
    if verbose:
        print("\n" + "-" * 50)
        print("Ordered Probit Model")
        print("-" * 50)

    try:
        X = tier_df[['ai_percentage', 'avg_rating']]
        y = tier_df['tier_numeric']

        ord_model = OrderedModel(y, X, distr='probit')
        ord_result = ord_model.fit(method='bfgs', disp=False)

        # Extract coefficients (not thresholds)
        params = {}
        se = {}
        p_values = {}
        for var in ['ai_percentage', 'avg_rating']:
            if var in ord_result.params.index:
                params[var] = ord_result.params[var]
                se[var] = ord_result.bse[var]
                p_values[var] = ord_result.pvalues[var]

        results['ordered_probit'] = {
            'params': params,
            'se': se,
            'p_values': p_values,
            'pseudo_rsquared': ord_result.prsquared,
            'model': ord_result
        }

        if verbose:
            print(f"\nAI percentage coefficient: {params.get('ai_percentage', np.nan):.6f}")
            print(f"  SE: {se.get('ai_percentage', np.nan):.6f}")
            print(f"  p-value: {p_values.get('ai_percentage', np.nan):.4e}")

            print(f"\navg_rating coefficient: {params.get('avg_rating', np.nan):.4f}")
            print(f"  p-value: {p_values.get('avg_rating', np.nan):.4e}")

    except Exception as e:
        if verbose:
            print(f"Ordered probit failed: {e}")
        results['ordered_probit'] = None

    # =========================================================================
    # Ordered Logit
    # =========================================================================
    if verbose:
        print("\n" + "-" * 50)
        print("Ordered Logit Model")
        print("-" * 50)

    try:
        ord_model_logit = OrderedModel(y, X, distr='logit')
        ord_result_logit = ord_model_logit.fit(method='bfgs', disp=False)

        params_logit = {}
        se_logit = {}
        p_values_logit = {}
        for var in ['ai_percentage', 'avg_rating']:
            if var in ord_result_logit.params.index:
                params_logit[var] = ord_result_logit.params[var]
                se_logit[var] = ord_result_logit.bse[var]
                p_values_logit[var] = ord_result_logit.pvalues[var]

        results['ordered_logit'] = {
            'params': params_logit,
            'se': se_logit,
            'p_values': p_values_logit,
            'odds_ratios': {k: np.exp(v) for k, v in params_logit.items()},
            'pseudo_rsquared': ord_result_logit.prsquared,
            'model': ord_result_logit
        }

        if verbose and 'ai_percentage' in params_logit:
            or_ai = np.exp(params_logit['ai_percentage'])
            print(f"\nOdds ratio per 1% AI: {or_ai:.6f}")
            print(f"Odds ratio per 10% AI: {or_ai**10:.4f}")
            print(f"Odds ratio for 100% vs 0% AI: {or_ai**100:.4f}")

    except Exception as e:
        if verbose:
            print(f"Ordered logit failed: {e}")
        results['ordered_logit'] = None

    # =========================================================================
    # Non-parametric comparison
    # =========================================================================
    if verbose:
        print("\n" + "-" * 50)
        print("Non-parametric Comparison")
        print("-" * 50)

    # Mean AI content by tier
    ai_by_tier = tier_df.groupby('tier_numeric')['ai_percentage'].agg(['mean', 'median', 'std', 'count'])
    results['ai_by_tier'] = ai_by_tier.to_dict()

    if verbose:
        print("\nAI content by tier:")
        for tier in [1, 2, 3]:
            if tier in ai_by_tier.index:
                tier_name = {1: 'Poster', 2: 'Spotlight', 3: 'Oral'}.get(tier)
                row = ai_by_tier.loc[tier]
                print(f"  {tier_name}: mean={row['mean']:.1f}%, "
                      f"median={row['median']:.1f}%, n={int(row['count'])}")

    # Kruskal-Wallis test
    groups = [tier_df[tier_df['tier_numeric'] == t]['ai_percentage'].values for t in [1, 2, 3]]
    groups = [g for g in groups if len(g) > 0]

    if len(groups) >= 2:
        h_stat, kw_p = scipy_stats.kruskal(*groups)
        results['kruskal_wallis'] = {'h_stat': h_stat, 'p_value': kw_p}

        if verbose:
            print(f"\nKruskal-Wallis test: H = {h_stat:.2f}, p = {kw_p:.4f}")

    # Interpretation
    if verbose:
        print("\n" + "-" * 50)
        print("INTERPRETATION")
        print("-" * 50)

        op_p = results.get('ordered_probit', {})
        if op_p:
            op_p = op_p.get('p_values', {}).get('ai_percentage', 1)
        else:
            op_p = 1

        if op_p < ALPHA:
            ai_coef = results['ordered_probit']['params']['ai_percentage']
            direction = "LOWER" if ai_coef < 0 else "HIGHER"
            print(f"\n✓ SIGNIFICANT: High-AI papers assigned to {direction} tiers")
            print(f"  → Evidence of tier discrimination based on AI content")
        else:
            print(f"\n→ No significant relationship between AI content and tier")
            print(f"  → Tier assignment appears unrelated to AI content")

    return results


# =============================================================================
# TEST 4: AUTHOR REPUTATION × ACCEPTANCE INTERACTION
# =============================================================================

def reputation_acceptance_interaction(df: pd.DataFrame,
                                      reputation_col: str = 'first_author_h_index',
                                      verbose: bool = True) -> Dict:
    """
    Test if author reputation insulates from AI content penalty at acceptance.

    Model: Pr(Accept) = Φ(α + β₁·AI + β₂·Reputation + β₃·(AI×Reputation) + ...)

    If β₃ > 0, reputation protects against the AI penalty.

    Parameters
    ----------
    df : DataFrame
        Must have: ai_percentage, avg_rating, accepted, and reputation column
    reputation_col : str
        Column with author reputation metric
    verbose : bool

    Returns
    -------
    dict with interaction analysis results
    """
    try:
        from statsmodels.discrete.discrete_model import Probit, Logit
        import statsmodels.api as sm
    except ImportError:
        raise ImportError("statsmodels required")

    if verbose:
        print("\n" + "=" * 70)
        print("TEST 4: Author Reputation × AI Content Interaction")
        print("=" * 70)
        print(f"Reputation measure: {reputation_col}")

    # Check for required columns
    required = ['ai_percentage', 'avg_rating', 'accepted', reputation_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Clean data
    clean_df = df.dropna(subset=required).copy()

    if verbose:
        print(f"\nSample with author data: {len(clean_df)}")
        print(f"Acceptance rate: {clean_df['accepted'].mean():.1%}")
        print(f"Mean {reputation_col}: {clean_df[reputation_col].mean():.1f}")

    if len(clean_df) < 100:
        warnings.warn(f"Small sample ({len(clean_df)}). Results may be unreliable.")

    results = {'n': len(clean_df), 'reputation_col': reputation_col}

    # Create interaction term
    clean_df['ai_x_reputation'] = clean_df['ai_percentage'] * clean_df[reputation_col]

    # Standardize for interpretation
    clean_df['ai_std'] = (clean_df['ai_percentage'] - clean_df['ai_percentage'].mean()) / clean_df['ai_percentage'].std()
    clean_df['rep_std'] = (clean_df[reputation_col] - clean_df[reputation_col].mean()) / clean_df[reputation_col].std()
    clean_df['ai_x_rep_std'] = clean_df['ai_std'] * clean_df['rep_std']

    # Build feature matrices
    X_interact = clean_df[['ai_percentage', 'avg_rating', reputation_col, 'ai_x_reputation']].copy()
    X_interact = sm.add_constant(X_interact)

    X_no_interact = clean_df[['ai_percentage', 'avg_rating', reputation_col]].copy()
    X_no_interact = sm.add_constant(X_no_interact)

    y = clean_df['accepted']

    # =========================================================================
    # Model without interaction
    # =========================================================================
    if verbose:
        print("\n" + "-" * 50)
        print("Model 1: Main Effects Only")
        print("-" * 50)

    try:
        probit_main = Probit(y, X_no_interact)
        result_main = probit_main.fit(disp=False, cov_type='HC3')

        results['main_effects'] = {
            'params': result_main.params.to_dict(),
            'se': result_main.bse.to_dict(),
            'p_values': result_main.pvalues.to_dict(),
            'pseudo_rsquared': result_main.prsquared,
            'll': result_main.llf
        }

        if verbose:
            print(f"\nAI coefficient: {result_main.params['ai_percentage']:.6f} "
                  f"(p = {result_main.pvalues['ai_percentage']:.4e})")
            print(f"Reputation coefficient: {result_main.params[reputation_col]:.6f} "
                  f"(p = {result_main.pvalues[reputation_col]:.4e})")

    except Exception as e:
        if verbose:
            print(f"Main effects model failed: {e}")
        results['main_effects'] = None

    # =========================================================================
    # Model with interaction
    # =========================================================================
    if verbose:
        print("\n" + "-" * 50)
        print("Model 2: With Interaction")
        print("-" * 50)

    try:
        probit_interact = Probit(y, X_interact)
        result_interact = probit_interact.fit(disp=False, cov_type='HC3')

        results['interaction_model'] = {
            'params': result_interact.params.to_dict(),
            'se': result_interact.bse.to_dict(),
            'p_values': result_interact.pvalues.to_dict(),
            'pseudo_rsquared': result_interact.prsquared,
            'll': result_interact.llf
        }

        if verbose:
            print(f"\nAI coefficient: {result_interact.params['ai_percentage']:.6f} "
                  f"(p = {result_interact.pvalues['ai_percentage']:.4e})")
            print(f"Reputation coefficient: {result_interact.params[reputation_col]:.6f} "
                  f"(p = {result_interact.pvalues[reputation_col]:.4e})")
            print(f"\n*** Interaction (AI × Reputation): {result_interact.params['ai_x_reputation']:.6f} ***")
            print(f"    SE: {result_interact.bse['ai_x_reputation']:.6f}")
            print(f"    p-value: {result_interact.pvalues['ai_x_reputation']:.4e}")

    except Exception as e:
        if verbose:
            print(f"Interaction model failed: {e}")
        results['interaction_model'] = None

    # =========================================================================
    # Likelihood ratio test
    # =========================================================================
    if results.get('main_effects') and results.get('interaction_model'):
        ll_main = results['main_effects']['ll']
        ll_interact = results['interaction_model']['ll']
        lr_stat = 2 * (ll_interact - ll_main)
        lr_p = 1 - scipy_stats.chi2.cdf(lr_stat, df=1)

        results['lr_test'] = {'statistic': lr_stat, 'p_value': lr_p}

        if verbose:
            print(f"\nLikelihood Ratio Test (interaction vs main effects):")
            print(f"  LR statistic: {lr_stat:.3f}")
            print(f"  p-value: {lr_p:.4f}")

    # =========================================================================
    # Stratified analysis
    # =========================================================================
    if verbose:
        print("\n" + "-" * 50)
        print("Stratified Analysis")
        print("-" * 50)

    # Split by reputation
    rep_median = clean_df[reputation_col].median()
    high_rep = clean_df[clean_df[reputation_col] >= rep_median]
    low_rep = clean_df[clean_df[reputation_col] < rep_median]

    results['stratified'] = {}

    for label, subset in [('high_reputation', high_rep), ('low_reputation', low_rep)]:
        if len(subset) >= 30:
            # Simple logit for each stratum
            X_sub = subset[['ai_percentage', 'avg_rating']].copy()
            X_sub = sm.add_constant(X_sub)
            y_sub = subset['accepted']

            try:
                logit_sub = Logit(y_sub, X_sub).fit(disp=False, cov_type='HC3')
                results['stratified'][label] = {
                    'n': len(subset),
                    'acceptance_rate': subset['accepted'].mean(),
                    'ai_coef': logit_sub.params['ai_percentage'],
                    'ai_se': logit_sub.bse['ai_percentage'],
                    'ai_p': logit_sub.pvalues['ai_percentage'],
                    'ai_or': np.exp(logit_sub.params['ai_percentage'])
                }

                if verbose:
                    print(f"\n{label.replace('_', ' ').title()} (n={len(subset)}):")
                    print(f"  Acceptance rate: {subset['accepted'].mean():.1%}")
                    print(f"  AI coefficient: {logit_sub.params['ai_percentage']:.6f} "
                          f"(p = {logit_sub.pvalues['ai_percentage']:.4e})")

            except Exception as e:
                if verbose:
                    print(f"\n{label}: Analysis failed - {e}")

    # =========================================================================
    # Interpretation
    # =========================================================================
    if verbose:
        print("\n" + "=" * 70)
        print("INTERPRETATION")
        print("=" * 70)

        interact_p = results.get('interaction_model', {})
        if interact_p:
            interact_p = interact_p.get('p_values', {}).get('ai_x_reputation', 1)
        else:
            interact_p = 1

        if interact_p < ALPHA:
            interact_coef = results['interaction_model']['params']['ai_x_reputation']
            if interact_coef > 0:
                print("\n✓ SIGNIFICANT POSITIVE INTERACTION")
                print("  High-reputation authors are INSULATED from the AI penalty")
                print("  → Reputation protects against AI content discrimination")
            else:
                print("\n✓ SIGNIFICANT NEGATIVE INTERACTION")
                print("  High-reputation authors face STRONGER AI penalty")
                print("  → Perhaps held to higher standards")
        else:
            print("\n→ No significant interaction")
            print("  AI penalty (if any) applies equally across reputation levels")

    return results


# =============================================================================
# TEST 5: SELECTION BOUNDS
# =============================================================================

def selection_bounds_analysis(df: pd.DataFrame,
                              observed_col: str = 'first_author_h_index',
                              verbose: bool = True) -> Dict:
    """
    Compute bounds for author-level analyses given selection on observing authors.

    For rejected papers without arXiv posting, we don't observe authors.
    This computes best-case and worst-case bounds.

    Parameters
    ----------
    df : DataFrame
        Must have: accepted, and observed_col
    observed_col : str
        Column that's only observed for some papers
    verbose : bool

    Returns
    -------
    dict with bounds analysis
    """
    if verbose:
        print("\n" + "=" * 70)
        print("TEST 5: Selection Bounds Analysis")
        print("=" * 70)

    results = {}

    # Identify observed vs unobserved
    observed = df[df[observed_col].notna()].copy()
    unobserved = df[df[observed_col].isna()].copy()

    n_total = len(df)
    n_observed = len(observed)
    n_unobserved = len(unobserved)

    results['n_total'] = n_total
    results['n_observed'] = n_observed
    results['n_unobserved'] = n_unobserved
    results['pct_observed'] = n_observed / n_total

    if verbose:
        print(f"\nTotal papers: {n_total}")
        print(f"Papers with {observed_col}: {n_observed} ({100*n_observed/n_total:.1f}%)")
        print(f"Papers without {observed_col}: {n_unobserved} ({100*n_unobserved/n_total:.1f}%)")

    # Selection by acceptance status
    if 'accepted' in df.columns:
        observed_by_status = observed.groupby('accepted').size()
        total_by_status = df.groupby('accepted').size()

        results['observation_by_status'] = {
            'accepted': {
                'observed': observed_by_status.get(1, observed_by_status.get(True, 0)),
                'total': total_by_status.get(1, total_by_status.get(True, 0))
            },
            'rejected': {
                'observed': observed_by_status.get(0, observed_by_status.get(False, 0)),
                'total': total_by_status.get(0, total_by_status.get(False, 0))
            }
        }

        if verbose:
            acc_obs = results['observation_by_status']['accepted']
            rej_obs = results['observation_by_status']['rejected']

            print(f"\nObservation by acceptance status:")
            print(f"  Accepted: {acc_obs['observed']}/{acc_obs['total']} "
                  f"({100*acc_obs['observed']/max(acc_obs['total'],1):.1f}%)")
            print(f"  Rejected: {rej_obs['observed']}/{rej_obs['total']} "
                  f"({100*rej_obs['observed']/max(rej_obs['total'],1):.1f}%)")

    # Bounds on author characteristics
    if verbose:
        print("\n" + "-" * 50)
        print("Bounds on Unobserved Papers")
        print("-" * 50)

    observed_mean = observed[observed_col].mean()
    observed_min = observed[observed_col].min()
    observed_max = observed[observed_col].max()

    results['observed_stats'] = {
        'mean': observed_mean,
        'min': observed_min,
        'max': observed_max,
        'std': observed[observed_col].std()
    }

    if verbose:
        print(f"\nObserved {observed_col}:")
        print(f"  Mean: {observed_mean:.2f}")
        print(f"  Range: [{observed_min:.0f}, {observed_max:.0f}]")

    # Worst-case bounds
    # If all unobserved are at the min
    worst_case_low = (observed_mean * n_observed + observed_min * n_unobserved) / n_total
    # If all unobserved are at the max
    worst_case_high = (observed_mean * n_observed + observed_max * n_unobserved) / n_total

    results['bounds'] = {
        'lower': worst_case_low,
        'upper': worst_case_high
    }

    if verbose:
        print(f"\nBounds on true population mean:")
        print(f"  Best case (unobserved = max): {worst_case_high:.2f}")
        print(f"  Worst case (unobserved = min): {worst_case_low:.2f}")
        print(f"  Width of identified set: {worst_case_high - worst_case_low:.2f}")

    # Recommendation
    if verbose:
        print("\n" + "-" * 50)
        print("RECOMMENDATIONS")
        print("-" * 50)

        if results['pct_observed'] > 0.8:
            print("\n✓ High observation rate (>80%)")
            print("  Selection bias likely limited")
        elif results['pct_observed'] > 0.5:
            print("\n⚠ Moderate observation rate (50-80%)")
            print("  Consider reporting bounds alongside point estimates")
        else:
            print("\n⚠ Low observation rate (<50%)")
            print("  Selection bias may be substantial")
            print("  Recommend focusing on accepted-paper analyses")

    return results


# =============================================================================
# SCORE-ACCEPTANCE RESIDUAL ANALYSIS
# =============================================================================

def score_acceptance_residual(df: pd.DataFrame,
                              verbose: bool = True) -> Dict:
    """
    Analyze whether high-AI papers have lower acceptance rates than
    predicted by their scores alone.

    Steps:
    1. Fit acceptance ~ scores only (no AI)
    2. Compare residuals by AI content

    This shows if ACs penalize AI content beyond what scores predict.

    Parameters
    ----------
    df : DataFrame
        Must have: ai_percentage, avg_rating, accepted

    Returns
    -------
    dict with residual analysis results
    """
    try:
        from statsmodels.discrete.discrete_model import Probit
        import statsmodels.api as sm
    except ImportError:
        raise ImportError("statsmodels required")

    if verbose:
        print("\n" + "=" * 70)
        print("Score-Acceptance Residual Analysis")
        print("=" * 70)

    clean_df = df.dropna(subset=['ai_percentage', 'avg_rating', 'accepted']).copy()

    if verbose:
        print(f"\nSample: {len(clean_df)} papers")

    results = {'n': len(clean_df)}

    # Step 1: Fit baseline model (scores only)
    X_base = clean_df[['avg_rating']].copy()
    X_base = sm.add_constant(X_base)
    y = clean_df['accepted']

    try:
        baseline = Probit(y, X_base).fit(disp=False)
        clean_df['predicted_prob'] = baseline.predict()
        clean_df['residual'] = clean_df['accepted'] - clean_df['predicted_prob']

        results['baseline_model'] = {
            'params': baseline.params.to_dict(),
            'pseudo_rsquared': baseline.prsquared
        }

        if verbose:
            print(f"\nBaseline model (scores only):")
            print(f"  Pseudo R²: {baseline.prsquared:.4f}")

    except Exception as e:
        if verbose:
            print(f"Baseline model failed: {e}")
        return results

    # Step 2: Compare residuals by AI content
    if verbose:
        print("\n" + "-" * 50)
        print("Residuals by AI Content")
        print("-" * 50)

    # Categorize
    high_ai = clean_df[clean_df['ai_percentage'] >= DEFAULT_AI_PAPER_THRESHOLD]
    low_ai = clean_df[clean_df['ai_percentage'] <= DEFAULT_HUMAN_PAPER_THRESHOLD]
    medium_ai = clean_df[(clean_df['ai_percentage'] > DEFAULT_HUMAN_PAPER_THRESHOLD) &
                         (clean_df['ai_percentage'] < DEFAULT_AI_PAPER_THRESHOLD)]

    results['residuals'] = {
        'high_ai': {
            'n': len(high_ai),
            'mean_residual': high_ai['residual'].mean(),
            'se_residual': high_ai['residual'].std() / np.sqrt(len(high_ai)),
            'actual_rate': high_ai['accepted'].mean(),
            'predicted_rate': high_ai['predicted_prob'].mean()
        },
        'low_ai': {
            'n': len(low_ai),
            'mean_residual': low_ai['residual'].mean(),
            'se_residual': low_ai['residual'].std() / np.sqrt(len(low_ai)),
            'actual_rate': low_ai['accepted'].mean(),
            'predicted_rate': low_ai['predicted_prob'].mean()
        },
        'medium_ai': {
            'n': len(medium_ai),
            'mean_residual': medium_ai['residual'].mean(),
            'se_residual': medium_ai['residual'].std() / np.sqrt(len(medium_ai)) if len(medium_ai) > 0 else np.nan,
            'actual_rate': medium_ai['accepted'].mean() if len(medium_ai) > 0 else np.nan,
            'predicted_rate': medium_ai['predicted_prob'].mean() if len(medium_ai) > 0 else np.nan
        }
    }

    if verbose:
        for cat, data in results['residuals'].items():
            print(f"\n{cat.replace('_', ' ').title()} (n={data['n']}):")
            print(f"  Predicted acceptance: {data['predicted_rate']:.1%}")
            print(f"  Actual acceptance: {data['actual_rate']:.1%}")
            print(f"  Residual: {data['mean_residual']:+.3f} (SE: {data['se_residual']:.3f})")

    # Test difference in residuals
    if len(high_ai) >= 10 and len(low_ai) >= 10:
        mw = mann_whitney_test(high_ai['residual'].values, low_ai['residual'].values)
        results['residual_test'] = mw

        if verbose:
            print(f"\nHigh AI vs Low AI residuals:")
            print(f"  Difference: {results['residuals']['high_ai']['mean_residual'] - results['residuals']['low_ai']['mean_residual']:+.4f}")
            print(f"  Mann-Whitney p = {mw['p_value']:.4f}")

    # Interpretation
    if verbose:
        print("\n" + "-" * 50)
        print("INTERPRETATION")
        print("-" * 50)

        high_resid = results['residuals']['high_ai']['mean_residual']
        low_resid = results['residuals']['low_ai']['mean_residual']

        if results.get('residual_test', {}).get('p_value', 1) < ALPHA:
            if high_resid < low_resid:
                print("\n✓ High-AI papers UNDERPERFORM their score-predicted acceptance")
                print("  → Area chairs penalize AI content beyond reviewer scores")
            else:
                print("\n✓ High-AI papers OUTPERFORM their score-predicted acceptance")
                print("  → Area chairs may favor AI content")
        else:
            print("\n→ No significant difference in residuals")
            print("  → Acceptance tracks scores regardless of AI content")

    return results


# =============================================================================
# COMPREHENSIVE ACCEPTANCE ANALYSIS
# =============================================================================

def run_acceptance_analysis(df: pd.DataFrame,
                            controls: Optional[List[str]] = None,
                            run_all: bool = True,
                            verbose: bool = True) -> Dict:
    """
    Run all acceptance analysis tests.

    Parameters
    ----------
    df : DataFrame
        Submissions data with acceptance decisions
    controls : list of str, optional
        Additional control variables for regressions
    run_all : bool
        If True, run all tests. If False, only run main tests.
    verbose : bool

    Returns
    -------
    dict with all test results
    """
    if verbose:
        print("\n" + "=" * 70)
        print("COMPREHENSIVE ACCEPTANCE ANALYSIS")
        print("=" * 70)
        print(f"\nTotal papers: {len(df)}")

        if 'accepted' in df.columns:
            acc_rate = df['accepted'].mean()
            print(f"Acceptance rate: {acc_rate:.1%}")

    # Prepare data
    data = prepare_acceptance_data(df)

    results = {}

    # Test 1: AI → Acceptance | Scores
    try:
        results['ai_acceptance_probit'] = ai_acceptance_probit(data, controls=controls, verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"\nTest 1 failed: {e}")
        results['ai_acceptance_probit'] = None

    # Test 1b: Unadjusted comparison
    try:
        results['ai_acceptance_by_category'] = ai_acceptance_by_category(data, verbose=verbose)
    except Exception as e:
        results['ai_acceptance_by_category'] = None

    # Test 2: Threshold Discontinuity
    try:
        results['threshold_discontinuity'] = threshold_discontinuity(data, verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"\nTest 2 failed: {e}")
        results['threshold_discontinuity'] = None

    if run_all:
        # Bandwidth sensitivity
        try:
            results['bandwidth_sensitivity'] = optimal_bandwidth_selection(data, verbose=verbose)
        except Exception as e:
            results['bandwidth_sensitivity'] = None

    # Test 3: Presentation Tier
    if 'tier' in data.columns or 'tier_name' in data.columns:
        try:
            results['presentation_tier'] = presentation_tier_analysis(data, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"\nTest 3 failed: {e}")
            results['presentation_tier'] = None

    # Test 4: Reputation Interaction
    reputation_cols = ['first_author_h_index', 'max_author_h_index', 'mean_author_h_index']
    available_rep = [c for c in reputation_cols if c in data.columns and data[c].notna().sum() >= 100]

    if available_rep and run_all:
        try:
            results['reputation_interaction'] = reputation_acceptance_interaction(
                data, reputation_col=available_rep[0], verbose=verbose
            )
        except Exception as e:
            if verbose:
                print(f"\nTest 4 failed: {e}")
            results['reputation_interaction'] = None

    # Test 5: Selection Bounds
    if available_rep and run_all:
        try:
            results['selection_bounds'] = selection_bounds_analysis(
                data, observed_col=available_rep[0], verbose=verbose
            )
        except Exception as e:
            results['selection_bounds'] = None

    # Score-Acceptance Residuals
    try:
        results['residual_analysis'] = score_acceptance_residual(data, verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"\nResidual analysis failed: {e}")
        results['residual_analysis'] = None

    # Summary
    if verbose:
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)

        completed = [k for k, v in results.items() if v is not None]
        print(f"\nCompleted tests: {len(completed)}/{len(results)}")
        for test in completed:
            print(f"  ✓ {test}")

    return results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print("""
Acceptance Analysis Module
==========================

Usage in Python:
    from analysis.acceptance_analysis import run_acceptance_analysis

    # Load data with acceptance decisions
    df = pd.read_csv('data/iclr_submissions_with_decisions.csv')

    # Run all tests
    results = run_acceptance_analysis(df)

Or run individual tests:
    from analysis.acceptance_analysis import (
        ai_acceptance_probit,
        threshold_discontinuity,
        presentation_tier_analysis,
        reputation_acceptance_interaction
    )
""")
