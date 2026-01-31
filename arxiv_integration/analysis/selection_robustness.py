"""
Selection Robustness Analysis: Author Ability Controls
======================================================
Addresses referee critique: "Selection, not treatment - low-ability
researchers simply overuse AI"

This module tests whether the AI-quality relationship persists after
controlling for author ability (h-index, citations).

Key Tests:
1. Stratified DiD by author reputation tier
2. Interaction models: AI × h_index
3. Within high-reputation author analysis
4. Component-specific effects (Soundness vs Presentation)

If the pattern holds among high-h-index authors, selection is ruled out.

Statistical Methods:
- Clustered standard errors by paper (multiple reviews per paper)
- Benjamini-Hochberg FDR correction for multiple testing
- Effect sizes (Cohen's d, percentage change)
- Bootstrap confidence intervals
- Winsorization for outlier robustness
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple, Union
from dataclasses import dataclass, field
import warnings
from scipy import stats as scipy_stats
import matplotlib.pyplot as plt

# Try to import statsmodels
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    warnings.warn("statsmodels not installed - some features may be limited")

# Constants
ALPHA = 0.05
BOOTSTRAP_ITERATIONS = 2000
WINSORIZE_LIMITS = (0.01, 0.01)  # 1% on each tail


# =============================================================================
# STATISTICAL UTILITIES (Bulletproof implementations)
# =============================================================================

def ols_with_clustered_se(
    data: pd.DataFrame,
    formula: str,
    cluster_col: str = 'submission_number',
    cov_type: str = 'cluster'
) -> Dict:
    """
    OLS regression with clustered standard errors.

    CRITICAL: Uses cluster-robust SEs to account for multiple reviews per paper.
    This is essential because reviews of the same paper are not independent.

    Parameters
    ----------
    data : DataFrame
        Data for regression
    formula : str
        Patsy formula (e.g., 'y ~ x1 + x2')
    cluster_col : str
        Column to cluster on (default: submission_number)
    cov_type : str
        'cluster' for clustered SEs, 'HC3' for heteroskedasticity-robust

    Returns
    -------
    dict with regression results
    """
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels required for regression analysis")

    # Fit model
    model = smf.ols(formula, data=data)

    # Choose covariance type
    if cov_type == 'cluster' and cluster_col in data.columns:
        # Clustered standard errors - CRITICAL for multiple reviews per paper
        results = model.fit(
            cov_type='cluster',
            cov_kwds={'groups': data[cluster_col]}
        )
    else:
        # Fall back to HC3 (heteroskedasticity-robust)
        results = model.fit(cov_type='HC3')

    return {
        'params': results.params.to_dict(),
        'se': results.bse.to_dict(),
        'p_values': results.pvalues.to_dict(),
        'ci_lower': results.conf_int()[0].to_dict(),
        'ci_upper': results.conf_int()[1].to_dict(),
        'rsquared': results.rsquared,
        'rsquared_adj': results.rsquared_adj,
        'nobs': int(results.nobs),
        'df_resid': results.df_resid,
        'fvalue': results.fvalue,
        'f_pvalue': results.f_pvalue,
        'model': results
    }


def bootstrap_diff_ci(
    group1: np.ndarray,
    group2: np.ndarray,
    n_boot: int = BOOTSTRAP_ITERATIONS,
    alpha: float = ALPHA,
    statistic: str = 'mean'
) -> Dict:
    """
    Bootstrap confidence interval for difference between groups.

    Parameters
    ----------
    group1, group2 : array-like
        Two groups to compare
    n_boot : int
        Number of bootstrap iterations
    alpha : float
        Significance level for CI
    statistic : str
        'mean' or 'median'

    Returns
    -------
    dict with estimate, CI, and p-value
    """
    group1 = np.array(group1)
    group2 = np.array(group2)

    # Remove NaN
    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]

    if len(group1) < 5 or len(group2) < 5:
        return {'estimate': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan, 'p_value': np.nan}

    stat_func = np.mean if statistic == 'mean' else np.median
    observed_diff = stat_func(group1) - stat_func(group2)

    # Bootstrap
    np.random.seed(42)  # For reproducibility
    boot_diffs = []
    for _ in range(n_boot):
        s1 = np.random.choice(group1, size=len(group1), replace=True)
        s2 = np.random.choice(group2, size=len(group2), replace=True)
        boot_diffs.append(stat_func(s1) - stat_func(s2))

    boot_diffs = np.array(boot_diffs)
    ci_lower = np.percentile(boot_diffs, 100 * alpha / 2)
    ci_upper = np.percentile(boot_diffs, 100 * (1 - alpha / 2))

    # Two-tailed p-value (proportion of bootstrap samples on opposite side of zero)
    if observed_diff > 0:
        p_value = 2 * np.mean(boot_diffs <= 0)
    else:
        p_value = 2 * np.mean(boot_diffs >= 0)
    p_value = min(p_value, 1.0)

    return {
        'estimate': observed_diff,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_value,
        'se': np.std(boot_diffs)
    }


def mann_whitney_test(group1: np.ndarray, group2: np.ndarray) -> Dict:
    """
    Mann-Whitney U test with effect size (rank-biserial correlation).

    Parameters
    ----------
    group1, group2 : array-like
        Two groups to compare

    Returns
    -------
    dict with statistic, p-value, and effect size
    """
    group1 = np.array(group1)
    group2 = np.array(group2)

    # Remove NaN
    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]

    if len(group1) < 5 or len(group2) < 5:
        return {'statistic': np.nan, 'p_value': np.nan, 'effect_size': np.nan}

    stat, p = scipy_stats.mannwhitneyu(group1, group2, alternative='two-sided')

    # Rank-biserial correlation (effect size)
    # r = 1 - (2U / (n1 * n2))
    n1, n2 = len(group1), len(group2)
    r = 1 - (2 * stat) / (n1 * n2)

    return {
        'statistic': stat,
        'p_value': p,
        'effect_size': r,
        'n1': n1,
        'n2': n2
    }


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Cohen's d effect size with Hedges' correction for small samples.

    Parameters
    ----------
    group1, group2 : array-like
        Two groups to compare

    Returns
    -------
    float: Cohen's d (positive means group1 > group2)
    """
    group1 = np.array(group1)
    group2 = np.array(group2)

    # Remove NaN
    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]

    n1, n2 = len(group1), len(group2)

    if n1 < 2 or n2 < 2:
        return np.nan

    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    d = (np.mean(group1) - np.mean(group2)) / pooled_std

    # Hedges' correction for small samples
    correction = 1 - (3 / (4 * (n1 + n2) - 9))

    return d * correction


def fdr_correction(p_values: List[float], alpha: float = ALPHA) -> Dict:
    """
    Benjamini-Hochberg FDR correction for multiple testing.

    Parameters
    ----------
    p_values : list of float
        Uncorrected p-values
    alpha : float
        Desired FDR level

    Returns
    -------
    dict with corrected p-values and significant indicators
    """
    p_values = np.array(p_values)
    n = len(p_values)

    if n == 0:
        return {'p_corrected': [], 'significant': [], 'n_significant': 0}

    # Handle NaN
    valid_mask = ~np.isnan(p_values)
    valid_p = p_values[valid_mask]

    if len(valid_p) == 0:
        return {'p_corrected': p_values, 'significant': [False] * n, 'n_significant': 0}

    if HAS_STATSMODELS:
        rejected, p_corrected_valid, _, _ = multipletests(valid_p, alpha=alpha, method='fdr_bh')
    else:
        # Manual BH implementation
        sorted_idx = np.argsort(valid_p)
        sorted_p = valid_p[sorted_idx]
        m = len(sorted_p)

        # BH adjusted p-values
        adjusted = np.zeros(m)
        adjusted[-1] = sorted_p[-1]
        for i in range(m - 2, -1, -1):
            adjusted[i] = min(adjusted[i + 1], sorted_p[i] * m / (i + 1))

        p_corrected_valid = np.zeros(m)
        p_corrected_valid[sorted_idx] = np.minimum(1, adjusted)
        rejected = p_corrected_valid < alpha

    # Reconstruct full arrays
    p_corrected = np.full(n, np.nan)
    p_corrected[valid_mask] = p_corrected_valid

    significant = np.full(n, False)
    significant[valid_mask] = rejected

    return {
        'p_corrected': p_corrected.tolist(),
        'significant': significant.tolist(),
        'n_significant': int(np.sum(rejected)),
        'n_tests': n
    }


def winsorize(x: np.ndarray, limits: Tuple[float, float] = WINSORIZE_LIMITS) -> np.ndarray:
    """
    Winsorize data to reduce outlier influence.

    Parameters
    ----------
    x : array-like
        Data to winsorize
    limits : tuple
        (lower, upper) quantile limits

    Returns
    -------
    array: Winsorized data
    """
    x = np.array(x)
    valid = ~np.isnan(x)

    if np.sum(valid) < 10:
        return x

    lower = np.percentile(x[valid], limits[0] * 100)
    upper = np.percentile(x[valid], (1 - limits[1]) * 100)

    result = x.copy()
    result[valid & (x < lower)] = lower
    result[valid & (x > upper)] = upper

    return result


# =============================================================================
# DATA PREPARATION
# =============================================================================

def merge_author_data(
    reviews_df: pd.DataFrame,
    submissions_df: pd.DataFrame,
    enriched_df: pd.DataFrame,
    submission_col: str = 'submission_number'
) -> pd.DataFrame:
    """
    Merge author reputation data with reviews.

    IMPORTANT: Deduplicates enriched_df to prevent row explosion during merge.
    """
    # Select author columns from enriched data
    author_cols = [
        submission_col,
        'first_author_name', 'first_author_country', 'first_author_university',
        'first_author_h_index', 'first_author_citation_count', 'first_author_paper_count',
        'first_author_top_university',
        'last_author_name', 'last_author_country', 'last_author_university',
        'last_author_h_index', 'last_author_citation_count', 'last_author_paper_count',
        'openalex_match_score'
    ]

    author_cols = [c for c in author_cols if c in enriched_df.columns]
    author_data = enriched_df[author_cols].copy()

    # CRITICAL: Deduplicate enriched data to prevent row explosion
    # Keep first occurrence (or could aggregate if needed)
    n_before = len(author_data)
    author_data = author_data.drop_duplicates(subset=[submission_col], keep='first')
    n_after = len(author_data)

    if n_before != n_after:
        warnings.warn(
            f"Removed {n_before - n_after} duplicate submission entries from enriched data "
            f"({n_before} → {n_after} unique papers)"
        )

    # Merge with submissions first
    merged = submissions_df.merge(author_data, on=submission_col, how='left')

    # Sanity check: merged should not have more rows than submissions
    if len(merged) != len(submissions_df):
        warnings.warn(
            f"Merge row count mismatch: {len(merged)} vs {len(submissions_df)} submissions. "
            f"Check for duplicate keys."
        )

    # Then merge reviews with submissions
    if reviews_df is not None:
        result = reviews_df.merge(merged, on=submission_col, how='left')
    else:
        result = merged

    return result


def create_reputation_tiers(
    df: pd.DataFrame,
    h_index_col: str = 'first_author_h_index',
    method: str = 'terciles'
) -> pd.DataFrame:
    """
    Create author reputation tiers based on h-index.
    """
    df = df.copy()
    h_values = df[h_index_col].dropna()

    if len(h_values) < 30:
        warnings.warn("Insufficient h-index data for stratification")
        df['reputation_tier'] = 'Unknown'
        return df

    if method == 'terciles':
        # Use rank to handle ties properly
        df['reputation_tier'] = pd.qcut(
            df[h_index_col].rank(method='first', na_option='keep'),
            q=3,
            labels=['Emerging', 'Established', 'Senior']
        )
    elif method == 'absolute':
        conditions = [
            df[h_index_col] < 10,
            (df[h_index_col] >= 10) & (df[h_index_col] < 30),
            (df[h_index_col] >= 30) & (df[h_index_col] < 60),
            df[h_index_col] >= 60
        ]
        choices = ['Emerging', 'Established', 'Senior', 'Highly Cited']
        df['reputation_tier'] = np.select(conditions, choices, default='Unknown')

    return df


def create_analysis_variables(df: pd.DataFrame, winsorize_h: bool = True) -> pd.DataFrame:
    """
    Create standardized analysis variables with optional winsorization.
    """
    df = df.copy()

    # Process AI percentage
    if 'ai_percentage' in df.columns:
        ai_pct = pd.to_numeric(df['ai_percentage'], errors='coerce')
        df['ai_percentage_clean'] = ai_pct
        df['ai_percentage_std'] = (ai_pct - ai_pct.mean()) / ai_pct.std()
        df['paper_AI'] = (ai_pct > 50).astype(int)

    # Process h-index with optional winsorization
    for col in ['first_author_h_index', 'last_author_h_index']:
        if col in df.columns:
            h = df[col].copy()

            # Winsorize to reduce outlier influence
            if winsorize_h:
                h_winsor = winsorize(h.values)
                df[f'{col}_winsor'] = h_winsor
                df[f'{col}_std'] = (h_winsor - np.nanmean(h_winsor)) / np.nanstd(h_winsor)
            else:
                df[f'{col}_std'] = (h - h.mean()) / h.std()

    # Create max h-index (robustness check)
    if 'first_author_h_index' in df.columns and 'last_author_h_index' in df.columns:
        df['max_author_h_index'] = df[['first_author_h_index', 'last_author_h_index']].max(axis=1)
        df['mean_author_h_index'] = df[['first_author_h_index', 'last_author_h_index']].mean(axis=1)

    # Log-transformed variables
    for col in ['first_author_citation_count', 'last_author_citation_count']:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col])

    return df


# =============================================================================
# STRATIFIED ANALYSIS
# =============================================================================

def stratified_did_analysis(
    df: pd.DataFrame,
    outcome: str = 'rating',
    strata_col: str = 'reputation_tier',
    paper_type_col: str = 'paper_AI',
    cluster_col: str = 'submission_number',
    verbose: bool = True
) -> Dict:
    """
    Run stratified analysis by reputation tier with clustered SEs.
    """
    results = {
        'by_stratum': {},
        'n_total': len(df),
        'outcome': outcome,
        'all_p_values': []  # For FDR correction
    }

    if verbose:
        print("\n" + "=" * 70)
        print(f"STRATIFIED ANALYSIS: {outcome.upper()}")
        print("=" * 70)
        print(f"Clustering standard errors by: {cluster_col}")

    strata = df[strata_col].dropna().unique()

    for stratum in sorted(strata):
        subset = df[df[strata_col] == stratum].copy()
        n_subset = len(subset)
        n_clusters = subset[cluster_col].nunique() if cluster_col in subset.columns else n_subset

        if n_subset < 50 or n_clusters < 20:
            if verbose:
                print(f"\n{stratum}: Skipped (n={n_subset}, clusters={n_clusters})")
            continue

        if subset[paper_type_col].nunique() < 2:
            if verbose:
                print(f"\n{stratum}: Skipped (no variation in paper type)")
            continue

        # Get AI and Human paper scores
        ai_papers = subset[subset[paper_type_col] == 1][outcome].dropna()
        human_papers = subset[subset[paper_type_col] == 0][outcome].dropna()

        if len(ai_papers) < 10 or len(human_papers) < 10:
            continue

        # Multiple statistical tests
        # 1. Bootstrap CI
        boot_result = bootstrap_diff_ci(ai_papers.values, human_papers.values)

        # 2. Mann-Whitney U test
        mw_result = mann_whitney_test(ai_papers.values, human_papers.values)

        # 3. Cohen's d effect size
        d = cohens_d(ai_papers.values, human_papers.values)

        # 4. OLS with clustered SEs (if enough clusters)
        if n_clusters >= 20:
            try:
                reg_result = ols_with_clustered_se(
                    subset.dropna(subset=[outcome, paper_type_col]),
                    f'{outcome} ~ {paper_type_col}',
                    cluster_col
                )
                ols_p = reg_result['p_values'].get(paper_type_col, np.nan)
                ols_se = reg_result['se'].get(paper_type_col, np.nan)
            except Exception:
                ols_p = np.nan
                ols_se = np.nan
        else:
            ols_p = np.nan
            ols_se = np.nan

        results['by_stratum'][stratum] = {
            'n': n_subset,
            'n_clusters': n_clusters,
            'n_ai': len(ai_papers),
            'n_human': len(human_papers),
            'mean_diff': boot_result['estimate'],
            'se_clustered': ols_se,
            'se_bootstrap': boot_result['se'],
            'ci_lower': boot_result['ci_lower'],
            'ci_upper': boot_result['ci_upper'],
            'p_value': mw_result['p_value'],  # Use Mann-Whitney as primary
            'p_value_ols': ols_p,
            'p_value_bootstrap': boot_result['p_value'],
            'effect_size_d': d,
            'effect_size_r': mw_result['effect_size'],
            'ai_mean': ai_papers.mean(),
            'human_mean': human_papers.mean(),
        }

        results['all_p_values'].append(mw_result['p_value'])

        if verbose:
            res = results['by_stratum'][stratum]
            print(f"\n{stratum} (n={n_subset}, clusters={n_clusters}):")
            print(f"  AI-Human difference: {res['mean_diff']:+.4f} [{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]")
            print(f"  p-value (Mann-Whitney): {res['p_value']:.4f}")
            print(f"  p-value (OLS clustered): {res['p_value_ols']:.4f}" if not np.isnan(ols_p) else "")
            print(f"  Effect size (Cohen's d): {res['effect_size_d']:+.3f}")

    # Apply FDR correction
    if results['all_p_values']:
        fdr_result = fdr_correction(results['all_p_values'])
        results['fdr_corrected'] = fdr_result

        # Add corrected p-values to results
        for i, stratum in enumerate(results['by_stratum'].keys()):
            if i < len(fdr_result['p_corrected']):
                results['by_stratum'][stratum]['p_value_fdr'] = fdr_result['p_corrected'][i]

        if verbose:
            print(f"\nFDR-corrected significant effects: {fdr_result['n_significant']}/{fdr_result['n_tests']}")

    return results


def stratified_component_analysis(
    df: pd.DataFrame,
    components: List[str] = ['soundness', 'presentation', 'contribution'],
    strata_col: str = 'reputation_tier',
    paper_type_col: str = 'paper_AI',
    cluster_col: str = 'submission_number',
    verbose: bool = True
) -> Dict:
    """
    Analyze effect on different review components by reputation tier.
    """
    results = {'by_stratum': {}, 'all_p_values': []}
    available_components = [c for c in components if c in df.columns]

    if not available_components:
        warnings.warn("No component columns found in data")
        return results

    if verbose:
        print("\n" + "=" * 70)
        print("COMPONENT-SPECIFIC EFFECTS BY REPUTATION TIER")
        print("=" * 70)

    for stratum in df[strata_col].dropna().unique():
        subset = df[df[strata_col] == stratum]

        if len(subset) < 50:
            continue

        stratum_results = {}

        for component in available_components:
            clean = subset.dropna(subset=[component, paper_type_col])

            if len(clean) < 30:
                continue

            ai_scores = clean[clean[paper_type_col] == 1][component]
            human_scores = clean[clean[paper_type_col] == 0][component]

            if len(ai_scores) < 10 or len(human_scores) < 10:
                continue

            diff = ai_scores.mean() - human_scores.mean()
            pct_diff = 100 * diff / human_scores.mean() if human_scores.mean() != 0 else np.nan

            boot = bootstrap_diff_ci(ai_scores.values, human_scores.values)
            mw = mann_whitney_test(ai_scores.values, human_scores.values)
            d = cohens_d(ai_scores.values, human_scores.values)

            stratum_results[component] = {
                'mean_diff': diff,
                'pct_diff': pct_diff,
                'ci_lower': boot['ci_lower'],
                'ci_upper': boot['ci_upper'],
                'p_value': mw['p_value'],
                'effect_size_d': d,
                'effect_size_r': mw['effect_size'],
                'ai_mean': ai_scores.mean(),
                'human_mean': human_scores.mean(),
            }

            results['all_p_values'].append(mw['p_value'])

        if stratum_results:
            results['by_stratum'][stratum] = stratum_results

            if verbose:
                print(f"\n{stratum}:")
                for comp, res in stratum_results.items():
                    sig = '*' if res['p_value'] < 0.05 else ''
                    print(f"  {comp}: {res['pct_diff']:+.1f}% (d={res['effect_size_d']:+.2f}, p={res['p_value']:.3f}){sig}")

    # FDR correction
    if results['all_p_values']:
        fdr_result = fdr_correction(results['all_p_values'])
        results['fdr_corrected'] = fdr_result
        if verbose:
            print(f"\nFDR-corrected significant: {fdr_result['n_significant']}/{fdr_result['n_tests']}")

    # Differential decline analysis
    if verbose:
        print("\n" + "-" * 50)
        print("Differential decline (Soundness vs Presentation):")
        for stratum, comps in results['by_stratum'].items():
            if 'soundness' in comps and 'presentation' in comps:
                diff_decline = comps['soundness']['pct_diff'] - comps['presentation']['pct_diff']
                print(f"  {stratum}: {diff_decline:+.1f}pp")

    return results


# =============================================================================
# INTERACTION MODELS
# =============================================================================

def interaction_model_h_index(
    df: pd.DataFrame,
    outcome: str = 'avg_rating',
    ai_col: str = 'ai_percentage_std',
    h_index_col: str = 'first_author_h_index_std',
    cluster_col: str = 'submission_number',
    controls: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict:
    """
    Estimate interaction model with clustered standard errors.
    """
    # Build formula
    formula_parts = [f'{ai_col}', f'{h_index_col}', f'{ai_col}:{h_index_col}']
    if controls:
        formula_parts.extend(controls)
    formula = f'{outcome} ~ ' + ' + '.join(formula_parts)

    # Clean data
    required_cols = [outcome, ai_col, h_index_col, cluster_col]
    if controls:
        required_cols.extend(controls)
    required_cols = [c for c in required_cols if c in df.columns]

    clean = df.dropna(subset=required_cols).copy()
    n_clusters = clean[cluster_col].nunique()

    if len(clean) < 100:
        warnings.warn(f"Small sample size: {len(clean)}")

    if verbose:
        print("\n" + "=" * 70)
        print("INTERACTION MODEL: AI × Author Reputation")
        print("=" * 70)
        print(f"\nFormula: {formula}")
        print(f"Sample size: {len(clean)}")
        print(f"Number of clusters: {n_clusters}")

    # Fit model with clustered SEs
    reg = ols_with_clustered_se(clean, formula, cluster_col)

    results = {
        'formula': formula,
        'n': len(clean),
        'n_clusters': n_clusters,
        'params': reg['params'],
        'se': reg['se'],
        'p_values': reg['p_values'],
        'ci_lower': reg['ci_lower'],
        'ci_upper': reg['ci_upper'],
        'rsquared': reg['rsquared'],
        'rsquared_adj': reg['rsquared_adj'],
    }

    # Extract key coefficients
    interaction_term = f'{ai_col}:{h_index_col}'

    for key, var in [('ai_effect', ai_col), ('h_index_effect', h_index_col), ('interaction', interaction_term)]:
        results[key] = {
            'coef': reg['params'].get(var, np.nan),
            'se': reg['se'].get(var, np.nan),
            'p': reg['p_values'].get(var, np.nan),
            'ci_lower': reg['ci_lower'].get(var, np.nan),
            'ci_upper': reg['ci_upper'].get(var, np.nan),
        }

    if verbose:
        print(f"\n{'Variable':<45} {'Coef':>10} {'SE':>10} {'p':>10}")
        print("-" * 77)
        for var in [ai_col, h_index_col, interaction_term, 'Intercept']:
            if var in reg['params']:
                ci_l = reg['ci_lower'].get(var, np.nan)
                ci_u = reg['ci_upper'].get(var, np.nan)
                print(f"{var:<45} {reg['params'][var]:>10.4f} {reg['se'][var]:>10.4f} {reg['p_values'][var]:>10.4f}")

        print(f"\nR²: {reg['rsquared']:.4f} (adj: {reg['rsquared_adj']:.4f})")
        print(f"Clustered by: {cluster_col} ({n_clusters} clusters)")

        # Interpretation
        print("\n" + "-" * 50)
        print("INTERPRETATION:")
        ai_p = results['ai_effect']['p']
        ai_coef = results['ai_effect']['coef']
        int_p = results['interaction']['p']
        int_coef = results['interaction']['coef']

        if ai_p < 0.05:
            direction = "negative" if ai_coef < 0 else "positive"
            print(f"  AI effect is significantly {direction} (p={ai_p:.4f})")
        else:
            print(f"  AI effect is not significant at mean h-index (p={ai_p:.4f})")

        if int_p < 0.05:
            if int_coef > 0:
                print(f"  Interaction is positive (p={int_p:.4f})")
                print(f"  → High-h-index authors show LESS negative AI effect")
            else:
                print(f"  Interaction is negative (p={int_p:.4f})")
                print(f"  → High-h-index authors show MORE negative AI effect")
        else:
            print(f"  No significant interaction (p={int_p:.4f})")
            print(f"  → Effect is UNIFORM across ability levels")

    return results


def full_selection_model(
    df: pd.DataFrame,
    outcome: str = 'avg_rating',
    cluster_col: str = 'submission_number',
    verbose: bool = True
) -> Dict:
    """
    Full model with all author controls to address selection.
    """
    required = [outcome, 'ai_percentage_clean', 'first_author_h_index', 'last_author_h_index']
    clean = df.dropna(subset=required).copy()

    # Create top university indicator if not present
    if 'first_author_top_university' not in clean.columns:
        clean['first_author_top_university'] = 0

    # Standardize
    for col in ['ai_percentage_clean', 'first_author_h_index', 'last_author_h_index']:
        if col in clean.columns:
            clean[f'{col}_std'] = (clean[col] - clean[col].mean()) / clean[col].std()

    formula = (f'{outcome} ~ ai_percentage_clean_std + '
               'first_author_h_index_std + last_author_h_index_std + '
               'first_author_top_university + '
               'ai_percentage_clean_std:first_author_h_index_std + '
               'ai_percentage_clean_std:first_author_top_university')

    if verbose:
        print("\n" + "=" * 70)
        print("FULL SELECTION CONTROL MODEL")
        print("=" * 70)
        print(f"\nFormula: {formula}")
        print(f"Sample size: {len(clean)}")

    reg = ols_with_clustered_se(clean, formula, cluster_col)

    if verbose:
        print(f"\n{'Variable':<50} {'Coef':>10} {'p':>10}")
        print("-" * 72)
        for var, coef in sorted(reg['params'].items()):
            p = reg['p_values'].get(var, np.nan)
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
            print(f"{var:<50} {coef:>10.4f} {p:>10.4f} {sig}")

        print(f"\nR²: {reg['rsquared']:.4f}")

    return {
        'formula': formula,
        'n': len(clean),
        'params': reg['params'],
        'se': reg['se'],
        'p_values': reg['p_values'],
        'rsquared': reg['rsquared'],
    }


# =============================================================================
# SUMMARY TABLE GENERATION
# =============================================================================

def generate_selection_robustness_table(
    stratified_results: Dict,
    interaction_results: Dict,
    component_results: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Generate publication-ready table summarizing selection robustness tests.
    """
    rows = []

    # Panel A: Stratified Analysis
    rows.append({
        'Panel': 'A. Stratified by Author Reputation',
        'Stratum': '', 'Estimate': '', 'SE': '', 'p-value': '', 'p-FDR': '', 'N': ''
    })

    for stratum, res in stratified_results.get('by_stratum', {}).items():
        rows.append({
            'Panel': '',
            'Stratum': stratum,
            'Estimate': f"{res['mean_diff']:.4f}",
            'SE': f"{res.get('se_clustered', res.get('se_bootstrap', np.nan)):.4f}",
            'p-value': f"{res['p_value']:.4f}",
            'p-FDR': f"{res.get('p_value_fdr', np.nan):.4f}",
            'N': str(res['n'])
        })

    # Panel B: Interaction Model
    rows.append({
        'Panel': 'B. Interaction Model (Clustered SEs)',
        'Stratum': '', 'Estimate': '', 'SE': '', 'p-value': '', 'p-FDR': '', 'N': ''
    })

    for key, label in [('ai_effect', 'AI percentage'),
                       ('h_index_effect', 'H-index'),
                       ('interaction', 'AI × H-index')]:
        if key in interaction_results:
            eff = interaction_results[key]
            rows.append({
                'Panel': '',
                'Stratum': label,
                'Estimate': f"{eff['coef']:.4f}",
                'SE': f"{eff['se']:.4f}",
                'p-value': f"{eff['p']:.4f}",
                'p-FDR': '',
                'N': str(interaction_results.get('n', ''))
            })

    # Panel C: Component Effects
    if component_results and 'by_stratum' in component_results:
        rows.append({
            'Panel': 'C. Component Effects by Reputation',
            'Stratum': '', 'Estimate': '', 'SE': '', 'p-value': '', 'p-FDR': '', 'N': ''
        })

        for stratum, comps in component_results['by_stratum'].items():
            for comp, res in comps.items():
                rows.append({
                    'Panel': '',
                    'Stratum': f"{stratum}: {comp}",
                    'Estimate': f"{res['pct_diff']:.1f}%",
                    'SE': '',
                    'p-value': f"{res['p_value']:.4f}",
                    'p-FDR': '',
                    'N': ''
                })

    return pd.DataFrame(rows)


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def run_selection_robustness_analysis(
    reviews_df: pd.DataFrame,
    submissions_df: pd.DataFrame,
    enriched_df: pd.DataFrame,
    components: List[str] = ['soundness', 'presentation', 'contribution'],
    verbose: bool = True,
    save_tables: bool = False,
    output_dir: str = '.'
) -> Dict:
    """
    Run complete selection robustness analysis with proper statistical methods.

    Statistical Methods:
    - Clustered standard errors (by paper) for all regressions
    - Benjamini-Hochberg FDR correction for multiple testing
    - Effect sizes (Cohen's d, rank-biserial correlation)
    - Bootstrap confidence intervals
    - Winsorized h-index for outlier robustness
    """
    if verbose:
        print("\n" + "=" * 70)
        print("SELECTION ROBUSTNESS ANALYSIS")
        print("Addressing referee critique: Selection vs Treatment")
        print("=" * 70)
        print("\nStatistical Methods:")
        print("  - Clustered SEs by paper (accounts for multiple reviews)")
        print("  - FDR correction for multiple testing")
        print("  - Bootstrap CIs (2000 iterations)")
        print("  - Winsorized h-index (1% tails)")

    # 1. Merge data
    if verbose:
        print("\n1. Merging author reputation data...")
        print(f"   Input: {len(submissions_df)} submissions, {len(enriched_df)} enriched records")

    merged = merge_author_data(reviews_df, submissions_df, enriched_df)
    h_available = merged['first_author_h_index'].notna().sum()

    # Calculate unique papers with h-index (in case reviews created duplicates)
    unique_papers_with_h = merged.drop_duplicates('submission_number')['first_author_h_index'].notna().sum()

    if verbose:
        print(f"   Merged rows: {len(merged)} (reviews × submissions)")
        print(f"   Unique papers with h-index: {unique_papers_with_h} / {len(submissions_df)} ({100*unique_papers_with_h/len(submissions_df):.1f}%)")

    # 2. Create analysis variables
    merged = create_analysis_variables(merged, winsorize_h=True)
    merged = create_reputation_tiers(merged)

    if verbose:
        print("\n   Reputation tier distribution:")
        print(merged.groupby('reputation_tier').size())

    results = {
        'n_total': len(merged),
        'n_unique_papers': len(submissions_df),
        'n_with_h_index': unique_papers_with_h,
        'coverage_rate': unique_papers_with_h / len(submissions_df) if len(submissions_df) > 0 else 0
    }

    # 3. Stratified analysis
    if verbose:
        print("\n2. Running stratified analysis...")

    results['stratified'] = stratified_did_analysis(
        merged, outcome='rating', verbose=verbose
    )

    # 4. Component analysis
    if verbose:
        print("\n3. Running component-specific analysis...")

    results['components'] = stratified_component_analysis(
        merged, components=components, verbose=verbose
    )

    # 5. Interaction model (submission-level)
    if verbose:
        print("\n4. Running interaction models...")

    submission_level = merged.drop_duplicates(subset='submission_number')
    submission_level = create_analysis_variables(submission_level, winsorize_h=True)

    results['interaction'] = interaction_model_h_index(
        submission_level, outcome='avg_rating', verbose=verbose
    )

    # 6. Full model
    if verbose:
        print("\n5. Running full selection control model...")

    results['full_model'] = full_selection_model(
        submission_level, outcome='avg_rating', verbose=verbose
    )

    # 7. Generate summary table
    results['summary_table'] = generate_selection_robustness_table(
        results['stratified'],
        results['interaction'],
        results['components']
    )

    if save_tables:
        results['summary_table'].to_csv(f'{output_dir}/selection_robustness_table.csv', index=False)
        if verbose:
            print(f"\n   Saved table to {output_dir}/selection_robustness_table.csv")

    # 8. Summary
    if verbose:
        print("\n" + "=" * 70)
        print("SUMMARY: Selection Robustness Tests")
        print("=" * 70)

        senior_results = results['stratified']['by_stratum'].get('Senior', {})
        if senior_results:
            effect = senior_results.get('mean_diff', np.nan)
            p = senior_results.get('p_value', np.nan)
            p_fdr = senior_results.get('p_value_fdr', np.nan)
            d = senior_results.get('effect_size_d', np.nan)

            print(f"\nAmong SENIOR authors (high h-index):")
            print(f"  Effect estimate: {effect:+.4f}")
            print(f"  p-value: {p:.4f} (FDR-corrected: {p_fdr:.4f})")
            print(f"  Effect size (Cohen's d): {d:+.3f}")

            if p < 0.05 and effect < 0:
                print("\n  → SELECTION RULED OUT: Even high-reputation authors")
                print("    show quality decline when using AI.")

        int_p = results['interaction']['interaction']['p']
        int_coef = results['interaction']['interaction']['coef']
        print(f"\nAI × H-index interaction: p = {int_p:.4f}")
        if int_p >= 0.05:
            print("  → Effect is UNIFORM across ability levels")
        else:
            print(f"  → Effect varies by ability (coefficient: {int_coef:+.4f})")

    return results


# =============================================================================
# COMPREHENSIVE INTERACTION ANALYSIS
# =============================================================================

def run_component_interaction_analysis(
    df: pd.DataFrame,
    components: List[str] = ['soundness', 'presentation', 'contribution'],
    h_index_cols: List[str] = ['first_author_h_index', 'last_author_h_index', 'mean_author_h_index'],
    ai_col: str = 'ai_percentage',
    cluster_col: str = 'submission_number',
    verbose: bool = True
) -> Dict:
    """
    Run interaction models across all components and h-index measures.

    This addresses the question: Does author experience moderate the AI effect
    differently for Soundness vs Presentation?

    Parameters
    ----------
    df : DataFrame
        Data with reviews merged with author data
    components : list
        Review components to analyze (soundness, presentation, contribution)
    h_index_cols : list
        H-index measures to test (first_author, last_author, mean_author)
    ai_col : str
        AI percentage column
    cluster_col : str
        Column to cluster standard errors on
    verbose : bool
        Print results

    Returns
    -------
    dict with interaction results for each component × h-index combination
    """
    results = {
        'by_component': {},
        'by_h_index': {},
        'summary_table': None
    }

    if verbose:
        print("\n" + "=" * 70)
        print("COMPONENT × H-INDEX INTERACTION ANALYSIS")
        print("=" * 70)
        print("\nDoes author experience moderate the AI effect differently")
        print("for Soundness (judgment) vs Presentation (polish)?")
        print("=" * 70)

    # Prepare data
    df = df.copy()

    # Standardize AI percentage
    if ai_col in df.columns:
        ai_clean = pd.to_numeric(df[ai_col], errors='coerce')
        df['ai_std'] = (ai_clean - ai_clean.mean()) / ai_clean.std()

    # Standardize h-index columns
    for h_col in h_index_cols:
        if h_col in df.columns:
            h_clean = pd.to_numeric(df[h_col], errors='coerce')
            df[f'{h_col}_std'] = (h_clean - np.nanmean(h_clean)) / np.nanstd(h_clean)

    # Available components
    available_components = [c for c in components if c in df.columns]
    available_h_cols = [h for h in h_index_cols if h in df.columns]

    if not available_components:
        warnings.warn("No component columns found")
        return results

    if not available_h_cols:
        warnings.warn("No h-index columns found")
        return results

    # Run interaction models
    summary_rows = []

    for component in available_components:
        results['by_component'][component] = {}

        if verbose:
            print(f"\n{'─' * 70}")
            print(f"COMPONENT: {component.upper()}")
            print(f"{'─' * 70}")

        for h_col in available_h_cols:
            h_col_std = f'{h_col}_std'

            if h_col_std not in df.columns:
                continue

            # Build formula
            formula = f'{component} ~ ai_std + {h_col_std} + ai_std:{h_col_std}'

            # Clean data
            clean = df.dropna(subset=[component, 'ai_std', h_col_std, cluster_col]).copy()

            if len(clean) < 100:
                continue

            n_clusters = clean[cluster_col].nunique()

            try:
                reg = ols_with_clustered_se(clean, formula, cluster_col)

                interaction_term = f'ai_std:{h_col_std}'
                ai_coef = reg['params'].get('ai_std', np.nan)
                ai_p = reg['p_values'].get('ai_std', np.nan)
                int_coef = reg['params'].get(interaction_term, np.nan)
                int_se = reg['se'].get(interaction_term, np.nan)
                int_p = reg['p_values'].get(interaction_term, np.nan)

                result = {
                    'n': len(clean),
                    'n_clusters': n_clusters,
                    'ai_coef': ai_coef,
                    'ai_p': ai_p,
                    'interaction_coef': int_coef,
                    'interaction_se': int_se,
                    'interaction_p': int_p,
                    'rsquared': reg['rsquared']
                }

                results['by_component'][component][h_col] = result

                # Add to summary
                sig = '***' if int_p < 0.001 else ('**' if int_p < 0.01 else ('*' if int_p < 0.05 else ''))
                summary_rows.append({
                    'component': component,
                    'h_index_measure': h_col.replace('_h_index', '').replace('_author', ''),
                    'ai_coef': ai_coef,
                    'ai_p': ai_p,
                    'interaction_coef': int_coef,
                    'interaction_se': int_se,
                    'interaction_p': int_p,
                    'significant': int_p < 0.05,
                    'n': len(clean)
                })

                if verbose:
                    h_label = h_col.replace('_h_index', '').replace('_', ' ')
                    print(f"\n  {h_label}:")
                    print(f"    AI effect:     β = {ai_coef:+.4f} (p = {ai_p:.4f})")
                    print(f"    Interaction:   β = {int_coef:+.4f} (p = {int_p:.4f}) {sig}")
                    if int_p < 0.05:
                        direction = "LESS" if int_coef > 0 else "MORE"
                        print(f"    → High {h_label} authors show {direction} negative AI effect")

            except Exception as e:
                if verbose:
                    print(f"  {h_col}: Error - {str(e)}")

    # Create summary DataFrame
    if summary_rows:
        results['summary_table'] = pd.DataFrame(summary_rows)

        if verbose:
            print("\n" + "=" * 70)
            print("SUMMARY: Component × H-index Interactions")
            print("=" * 70)
            print("\n" + results['summary_table'].to_string(index=False))

            # Key insight
            print("\n" + "-" * 70)
            print("KEY INSIGHT:")

            # Check if interaction is stronger for soundness vs presentation
            sound_int = [r for r in summary_rows if r['component'] == 'soundness']
            pres_int = [r for r in summary_rows if r['component'] == 'presentation']

            if sound_int and pres_int:
                avg_sound = np.mean([r['interaction_coef'] for r in sound_int])
                avg_pres = np.mean([r['interaction_coef'] for r in pres_int])

                print(f"  Average interaction (Soundness):    {avg_sound:+.4f}")
                print(f"  Average interaction (Presentation): {avg_pres:+.4f}")

                if avg_sound > avg_pres:
                    print("\n  → Experience moderates AI effect MORE for Soundness than Presentation")
                    print("    Interpretation: Experienced authors maintain judgment quality")
                    print("    even when using AI, but the benefit is specific to substance.")
                else:
                    print("\n  → Experience moderates AI effect similarly across components")

    return results


def _bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for the mean.

    Parameters
    ----------
    values : array
        Data values
    n_boot : int
        Number of bootstrap iterations
    ci : float
        Confidence level (default 0.95 for 95% CI)

    Returns
    -------
    tuple: (mean, ci_lower, ci_upper)
    """
    values = np.array(values)
    values = values[~np.isnan(values)]

    if len(values) < 3:
        return np.nan, np.nan, np.nan

    observed_mean = np.mean(values)

    # Bootstrap
    np.random.seed(42)
    boot_means = []
    for _ in range(n_boot):
        sample = np.random.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(sample))

    boot_means = np.array(boot_means)
    alpha = 1 - ci
    ci_lower = np.percentile(boot_means, 100 * alpha / 2)
    ci_upper = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return observed_mean, ci_lower, ci_upper


def create_interaction_plots(
    df: pd.DataFrame,
    outcome: str = 'rating',
    h_index_cols: List[str] = ['first_author_h_index', 'last_author_h_index', 'mean_author_h_index'],
    ai_col: str = 'ai_percentage',
    ai_threshold: float = 50.0,
    bins: List = [0, 10, 20, 30, 50, 100, 500],
    bin_labels: List[str] = ['0-10', '10-20', '20-30', '30-50', '50-100', '100+'],
    figsize: Tuple[int, int] = (15, 5),
    output_dir: Optional[str] = None,
    verbose: bool = True,
    show_ci: bool = True,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    aggregate_to_paper: bool = True,
    paper_id_col: str = 'submission_number'
) -> Dict:
    """
    Create interaction plots showing AI effect by h-index bins with confidence intervals.

    Creates a multi-panel figure with one plot per h-index measure,
    showing how the AI penalty varies by author experience.

    IMPORTANT: By default, aggregates to paper-level first (mean rating per paper)
    to avoid pseudo-replication from multiple reviews per paper.

    Parameters
    ----------
    df : DataFrame
        Data with reviews and author metrics
    outcome : str
        Outcome variable (rating, soundness, presentation)
    h_index_cols : list
        H-index columns to plot
    ai_col : str
        AI percentage column
    ai_threshold : float
        Threshold for classifying AI vs Human papers
    bins : list
        Bin edges for h-index
    bin_labels : list
        Labels for bins
    figsize : tuple
        Figure size
    output_dir : str, optional
        Directory to save figures
    verbose : bool
        Print summary
    show_ci : bool
        Show bootstrap confidence intervals as error bars
    n_bootstrap : int
        Number of bootstrap iterations for CI
    ci_level : float
        Confidence level (default 0.95 for 95% CI)
    aggregate_to_paper : bool
        If True, aggregate to paper-level first (recommended)
    paper_id_col : str
        Column identifying unique papers for aggregation

    Returns
    -------
    dict with plot data and figure paths
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not available for plotting")
        return {}

    df = df.copy()

    # Classify AI vs Human
    ai_clean = pd.to_numeric(df[ai_col], errors='coerce')
    df['paper_AI'] = (ai_clean >= ai_threshold).astype(int)

    available_h_cols = [h for h in h_index_cols if h in df.columns]
    n_plots = len(available_h_cols)

    if n_plots == 0:
        warnings.warn("No h-index columns available for plotting")
        return {}

    # Aggregate to paper level if requested (IMPORTANT for proper statistics)
    if aggregate_to_paper and paper_id_col in df.columns:
        agg_dict = {outcome: 'mean', 'paper_AI': 'first'}
        for h_col in available_h_cols:
            agg_dict[h_col] = 'first'

        analysis_df = df.groupby(paper_id_col).agg(agg_dict).reset_index()

        if verbose:
            print(f"Aggregated to paper level: {len(df):,} reviews → {len(analysis_df):,} papers")
    else:
        analysis_df = df
        if verbose and aggregate_to_paper:
            print(f"Warning: paper_id_col '{paper_id_col}' not found, using review-level data")

    results = {'plots': {}, 'data': {}}

    # Create figure
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    for idx, h_col in enumerate(available_h_cols):
        ax = axes[idx]

        # Create bins (use analysis_df which may be paper-level aggregated)
        analysis_df[f'{h_col}_bin'] = pd.cut(
            analysis_df[h_col],
            bins=bins,
            labels=bin_labels,
            include_lowest=True
        )

        # Calculate means and CIs by bin and AI status
        bin_data = {
            'bins': [],
            'human_means': [], 'human_ci_lower': [], 'human_ci_upper': [], 'human_n': [],
            'ai_means': [], 'ai_ci_lower': [], 'ai_ci_upper': [], 'ai_n': [],
            'gaps': [], 'gap_significant': []
        }

        for bin_label in bin_labels:
            bin_mask = analysis_df[f'{h_col}_bin'] == bin_label

            human_values = analysis_df.loc[bin_mask & (analysis_df['paper_AI'] == 0), outcome].dropna().values
            ai_values = analysis_df.loc[bin_mask & (analysis_df['paper_AI'] == 1), outcome].dropna().values

            # Bootstrap CIs
            if show_ci:
                h_mean, h_ci_l, h_ci_u = _bootstrap_mean_ci(human_values, n_bootstrap, ci_level)
                a_mean, a_ci_l, a_ci_u = _bootstrap_mean_ci(ai_values, n_bootstrap, ci_level)
            else:
                h_mean = np.mean(human_values) if len(human_values) > 0 else np.nan
                a_mean = np.mean(ai_values) if len(ai_values) > 0 else np.nan
                h_ci_l, h_ci_u, a_ci_l, a_ci_u = np.nan, np.nan, np.nan, np.nan

            bin_data['bins'].append(bin_label)
            bin_data['human_means'].append(h_mean)
            bin_data['human_ci_lower'].append(h_ci_l)
            bin_data['human_ci_upper'].append(h_ci_u)
            bin_data['human_n'].append(len(human_values))
            bin_data['ai_means'].append(a_mean)
            bin_data['ai_ci_lower'].append(a_ci_l)
            bin_data['ai_ci_upper'].append(a_ci_u)
            bin_data['ai_n'].append(len(ai_values))

            # Gap and significance (CIs don't overlap)
            gap = h_mean - a_mean if not (np.isnan(h_mean) or np.isnan(a_mean)) else np.nan
            bin_data['gaps'].append(gap)

            # Check if CIs overlap (significant if they don't)
            if show_ci and not any(np.isnan([h_ci_l, h_ci_u, a_ci_l, a_ci_u])):
                # CIs don't overlap if human lower > ai upper OR ai lower > human upper
                significant = (h_ci_l > a_ci_u) or (a_ci_l > h_ci_u)
            else:
                significant = False
            bin_data['gap_significant'].append(significant)

        # Convert to arrays for plotting
        x = np.arange(len(bin_data['bins']))
        human_means = np.array(bin_data['human_means'])
        ai_means = np.array(bin_data['ai_means'])

        # Calculate error bar sizes (distance from mean to CI bound)
        if show_ci:
            human_yerr_lower = human_means - np.array(bin_data['human_ci_lower'])
            human_yerr_upper = np.array(bin_data['human_ci_upper']) - human_means
            ai_yerr_lower = ai_means - np.array(bin_data['ai_ci_lower'])
            ai_yerr_upper = np.array(bin_data['ai_ci_upper']) - ai_means

            # Replace NaN with 0 for error bars
            human_yerr_lower = np.nan_to_num(human_yerr_lower, nan=0)
            human_yerr_upper = np.nan_to_num(human_yerr_upper, nan=0)
            ai_yerr_lower = np.nan_to_num(ai_yerr_lower, nan=0)
            ai_yerr_upper = np.nan_to_num(ai_yerr_upper, nan=0)

        # Plot with error bars
        if show_ci:
            ax.errorbar(x, human_means, yerr=[human_yerr_lower, human_yerr_upper],
                       fmt='o-', color='#2166AC', label='Human Papers',
                       markersize=8, linewidth=2, capsize=4, capthick=1.5)
            ax.errorbar(x, ai_means, yerr=[ai_yerr_lower, ai_yerr_upper],
                       fmt='s-', color='#B2182B', label='AI Papers',
                       markersize=8, linewidth=2, capsize=4, capthick=1.5)
        else:
            ax.plot(x, human_means, 'o-', color='#2166AC', label='Human Papers',
                   markersize=8, linewidth=2)
            ax.plot(x, ai_means, 's-', color='#B2182B', label='AI Papers',
                   markersize=8, linewidth=2)

        # Fill gap (shaded area between lines)
        valid_mask = ~(np.isnan(human_means) | np.isnan(ai_means))
        if valid_mask.any():
            ax.fill_between(x[valid_mask], human_means[valid_mask], ai_means[valid_mask],
                           alpha=0.15, color='gray')

        # Add significance markers
        if show_ci:
            for i, (sig, h_m, a_m) in enumerate(zip(bin_data['gap_significant'], human_means, ai_means)):
                if sig and not (np.isnan(h_m) or np.isnan(a_m)):
                    # Place asterisk above the higher point
                    y_pos = max(h_m, a_m) + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0] if ax.get_ylim()[1] != ax.get_ylim()[0] else 0.1)
                    ax.annotate('*', xy=(i, y_pos), ha='center', va='bottom',
                               fontsize=14, fontweight='bold', color='#333333')

        # Create x-axis labels with sample sizes
        total_n = [h + a for h, a in zip(bin_data['human_n'], bin_data['ai_n'])]
        x_labels_with_n = [f'{b}\n(n={n:,})' for b, n in zip(bin_data['bins'], total_n)]

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels_with_n, rotation=0, fontsize=9)
        ax.set_xlabel(h_col.replace('_', ' ').title(), fontsize=10)
        ax.set_ylabel(f'Mean {outcome.title()}', fontsize=10)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Style improvements
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Store data
        bin_data['total_n'] = total_n
        results['data'][h_col] = bin_data

        # Add title with gap range
        valid_gaps = [g for g in bin_data['gaps'] if not np.isnan(g)]
        if valid_gaps:
            min_gap = min(valid_gaps)
            max_gap = max(valid_gaps)
            ax.set_title(f'{h_col.replace("_", " ").title()}\nGap: {min_gap:.2f} to {max_gap:.2f}',
                        fontsize=11, fontweight='bold')

    ci_label = f'{int(ci_level*100)}% CI' if show_ci else ''
    title = f'{outcome.title()} by AI Status and Author Reputation'
    if show_ci:
        title += f'\n(Error bars: {ci_label}; * = significant difference)'
    plt.suptitle(title, fontsize=12, fontweight='bold', y=1.05)
    plt.tight_layout()

    # Save figure
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        fig_path = f'{output_dir}/interaction_plot_{outcome}.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
        results['figure_path'] = fig_path
        if verbose:
            print(f"Saved: {fig_path}")

    results['figure'] = fig

    if verbose:
        print("\n" + "=" * 70)
        print(f"INTERACTION PLOT: {outcome.upper()}")
        if show_ci:
            print(f"(Bootstrap {int(ci_level*100)}% CI, {n_bootstrap} iterations)")
        print("=" * 70)
        for h_col, data in results['data'].items():
            print(f"\n{h_col}:")
            if show_ci:
                print(f"  {'Bin':<10} {'Human Mean':>12} {'[95% CI]':>18} {'AI Mean':>12} {'[95% CI]':>18} {'Gap':>8} {'Sig':>5}")
                print(f"  {'-'*10} {'-'*12} {'-'*18} {'-'*12} {'-'*18} {'-'*8} {'-'*5}")
                for i, b in enumerate(data['bins']):
                    h_m = data['human_means'][i]
                    h_ci = f"[{data['human_ci_lower'][i]:.2f}, {data['human_ci_upper'][i]:.2f}]"
                    a_m = data['ai_means'][i]
                    a_ci = f"[{data['ai_ci_lower'][i]:.2f}, {data['ai_ci_upper'][i]:.2f}]"
                    gap = data['gaps'][i]
                    sig = '*' if data['gap_significant'][i] else ''
                    print(f"  {b:<10} {h_m:>12.3f} {h_ci:>18} {a_m:>12.3f} {a_ci:>18} {gap:>+8.3f} {sig:>5}")
            else:
                print(f"  {'Bin':<10} {'Gap':>8} {'Human n':>10} {'AI n':>10} {'Total n':>10}")
                print(f"  {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
                for i, b in enumerate(data['bins']):
                    gap = data['gaps'][i]
                    h_n = data['human_n'][i]
                    ai_n = data['ai_n'][i]
                    total_n = data['total_n'][i]
                    print(f"  {b:<10} {gap:>+8.3f} {h_n:>10,} {ai_n:>10,} {total_n:>10,}")

    return results


def create_interaction_plots_regression(
    df: pd.DataFrame,
    outcome: str = 'rating',
    h_index_cols: List[str] = ['first_author_h_index', 'last_author_h_index', 'mean_author_h_index'],
    ai_col: str = 'ai_percentage',
    ai_threshold: float = 50.0,
    figsize: Tuple[int, int] = (15, 5),
    output_dir: Optional[str] = None,
    verbose: bool = True,
    show_scatter: bool = True,
    scatter_alpha: float = 0.15,
    ci_level: float = 0.95,
    xlim_percentile: float = 99.0,
    aggregate_to_paper: bool = True,
    paper_id_col: str = 'submission_number'
) -> Dict:
    """
    Create interaction plots using OLS regression with interaction terms.

    Instead of arbitrary binning, fits a linear model:
        outcome ~ AI + h_index + AI × h_index

    IMPORTANT: By default, aggregates to paper-level first (mean rating per paper)
    to avoid pseudo-replication from multiple reviews per paper.

    This provides:
    - Proper statistical inference on the interaction effect
    - Continuous visualization without information loss
    - Confidence bands based on the fitted model
    - Paper-level analysis (not review-level)

    Parameters
    ----------
    df : DataFrame
        Data with reviews and author metrics
    outcome : str
        Outcome variable (rating, soundness, presentation)
    h_index_cols : list
        H-index columns to analyze
    ai_col : str
        AI percentage column
    ai_threshold : float
        Threshold for classifying AI vs Human papers
    figsize : tuple
        Figure size
    output_dir : str, optional
        Directory to save figures
    verbose : bool
        Print regression summary
    show_scatter : bool
        Show raw data points as scatter
    scatter_alpha : float
        Transparency of scatter points
    ci_level : float
        Confidence level for bands (default 0.95)
    xlim_percentile : float
        Percentile for x-axis limit (handles outliers)
    aggregate_to_paper : bool
        If True, aggregate to paper-level first (recommended)
    paper_id_col : str
        Column identifying unique papers for aggregation

    Returns
    -------
    dict with:
        - 'figure': matplotlib figure
        - 'models': dict of fitted OLS models per h-index column
        - 'interaction_effects': dict with interaction coefficients and p-values
        - 'figure_path': path if saved
        - 'paper_level_data': DataFrame of aggregated paper-level data
    """
    if not HAS_STATSMODELS:
        warnings.warn("statsmodels required for regression plots")
        return {}

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not available for plotting")
        return {}

    df = df.copy()

    # Classify AI vs Human
    ai_clean = pd.to_numeric(df[ai_col], errors='coerce')
    df['paper_AI'] = (ai_clean >= ai_threshold).astype(int)

    available_h_cols = [h for h in h_index_cols if h in df.columns]
    n_plots = len(available_h_cols)

    if n_plots == 0:
        warnings.warn("No h-index columns available for plotting")
        return {}

    # Aggregate to paper level if requested (IMPORTANT for proper statistics)
    if aggregate_to_paper and paper_id_col in df.columns:
        # Columns to aggregate (mean for numeric outcomes)
        agg_cols = [outcome] + available_h_cols
        agg_cols = [c for c in agg_cols if c in df.columns]

        # Group by paper, take mean of outcome and first value of paper-level vars
        agg_dict = {outcome: 'mean'}  # Mean rating across reviews
        for h_col in available_h_cols:
            agg_dict[h_col] = 'first'  # H-index is paper-level, take first
        agg_dict['paper_AI'] = 'first'  # AI status is paper-level

        paper_df = df.groupby(paper_id_col).agg(agg_dict).reset_index()

        if verbose:
            print(f"Aggregated to paper level: {len(df):,} reviews → {len(paper_df):,} papers")
            print(f"  Mean reviews per paper: {len(df)/len(paper_df):.1f}")

        analysis_df = paper_df
    else:
        analysis_df = df
        if verbose and aggregate_to_paper:
            print(f"Warning: paper_id_col '{paper_id_col}' not found, using review-level data")

    results = {
        'models': {},
        'interaction_effects': {},
        'regression_data': {},
        'paper_level_data': analysis_df if aggregate_to_paper else None
    }

    # Create figure
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    for idx, h_col in enumerate(available_h_cols):
        ax = axes[idx]

        # Prepare data for regression
        reg_df = analysis_df[[outcome, 'paper_AI', h_col]].dropna().copy()
        reg_df = reg_df.rename(columns={h_col: 'h_index'})

        if len(reg_df) < 10:
            warnings.warn(f"Insufficient data for {h_col}")
            continue

        # Fit OLS: outcome ~ AI + h_index + AI:h_index
        try:
            formula = f'{outcome} ~ paper_AI * h_index'
            model = smf.ols(formula, data=reg_df).fit()
            results['models'][h_col] = model

            # Extract interaction effect
            interaction_coef = model.params.get('paper_AI:h_index', np.nan)
            interaction_pval = model.pvalues.get('paper_AI:h_index', np.nan)
            interaction_se = model.bse.get('paper_AI:h_index', np.nan)

            results['interaction_effects'][h_col] = {
                'coefficient': interaction_coef,
                'std_error': interaction_se,
                'p_value': interaction_pval,
                'significant': interaction_pval < (1 - ci_level) if not np.isnan(interaction_pval) else False,
                'ai_main_effect': model.params.get('paper_AI', np.nan),
                'ai_main_pval': model.pvalues.get('paper_AI', np.nan),
                'h_index_effect': model.params.get('h_index', np.nan),
                'r_squared': model.rsquared,
                'n_obs': int(model.nobs)
            }

        except Exception as e:
            warnings.warn(f"Regression failed for {h_col}: {e}")
            continue

        # Create prediction grid
        h_min = reg_df['h_index'].min()
        h_max = np.percentile(reg_df['h_index'], xlim_percentile)
        h_grid = np.linspace(h_min, h_max, 100)

        # Predict for Human (paper_AI=0) and AI (paper_AI=1)
        pred_human_df = pd.DataFrame({'paper_AI': 0, 'h_index': h_grid})
        pred_ai_df = pd.DataFrame({'paper_AI': 1, 'h_index': h_grid})

        # Get predictions with confidence intervals
        pred_human = model.get_prediction(pred_human_df)
        pred_ai = model.get_prediction(pred_ai_df)

        human_mean = pred_human.predicted_mean
        human_ci = pred_human.conf_int(alpha=1 - ci_level)
        ai_mean = pred_ai.predicted_mean
        ai_ci = pred_ai.conf_int(alpha=1 - ci_level)

        # Store regression data
        results['regression_data'][h_col] = {
            'h_grid': h_grid,
            'human_pred': human_mean,
            'human_ci_lower': human_ci[:, 0],
            'human_ci_upper': human_ci[:, 1],
            'ai_pred': ai_mean,
            'ai_ci_lower': ai_ci[:, 0],
            'ai_ci_upper': ai_ci[:, 1]
        }

        # Plot scatter of raw data (optional)
        if show_scatter:
            human_data = reg_df[reg_df['paper_AI'] == 0]
            ai_data = reg_df[reg_df['paper_AI'] == 1]

            # Filter to xlim for scatter
            human_data = human_data[human_data['h_index'] <= h_max]
            ai_data = ai_data[ai_data['h_index'] <= h_max]

            ax.scatter(human_data['h_index'], human_data[outcome],
                      alpha=scatter_alpha, color='#2166AC', s=15, label='_nolegend_')
            ax.scatter(ai_data['h_index'], ai_data[outcome],
                      alpha=scatter_alpha, color='#B2182B', s=15, label='_nolegend_')

        # Plot regression lines
        ax.plot(h_grid, human_mean, '-', color='#2166AC', linewidth=2.5, label='Human Papers')
        ax.plot(h_grid, ai_mean, '-', color='#B2182B', linewidth=2.5, label='AI Papers')

        # Plot confidence bands
        ax.fill_between(h_grid, human_ci[:, 0], human_ci[:, 1],
                       alpha=0.25, color='#2166AC', label='_nolegend_')
        ax.fill_between(h_grid, ai_ci[:, 0], ai_ci[:, 1],
                       alpha=0.25, color='#B2182B', label='_nolegend_')

        # Formatting
        ax.set_xlabel(h_col.replace('_', ' ').title(), fontsize=10)
        ax.set_ylabel(f'Mean {outcome.title()}', fontsize=10)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(h_min, h_max)

        # Title with interaction effect
        int_eff = results['interaction_effects'][h_col]
        sig_marker = '*' if int_eff['significant'] else ''
        ax.set_title(
            f"{h_col.replace('_', ' ').title()}\n"
            f"Interaction: β={int_eff['coefficient']:.4f}{sig_marker} (p={int_eff['p_value']:.3f})",
            fontsize=10, fontweight='bold'
        )

    # Overall title
    ci_pct = int(ci_level * 100)
    level_str = "Paper-Level" if aggregate_to_paper else "Review-Level"
    plt.suptitle(
        f'{outcome.title()} vs Author H-index by AI Status ({level_str} OLS Regression)\n'
        f'Lines: fitted values; Bands: {ci_pct}% CI; * = significant interaction (p<{1-ci_level:.2f})',
        fontsize=11, fontweight='bold', y=1.02
    )
    plt.tight_layout()

    # Save figure
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        fig_path = f'{output_dir}/interaction_regression_{outcome}.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
        results['figure_path'] = fig_path
        if verbose:
            print(f"Saved: {fig_path}")

    results['figure'] = fig

    # Print summary
    if verbose:
        print("\n" + "=" * 70)
        print(f"REGRESSION INTERACTION ANALYSIS: {outcome.upper()} ({level_str})")
        print(f"Model: {outcome} ~ AI + h_index + AI × h_index")
        print("=" * 70)

        print(f"\n{'H-index Measure':<25} {'Interaction β':>15} {'SE':>10} {'p-value':>10} {'Sig':>6} {'R²':>8} {'N':>8}")
        print("-" * 82)

        for h_col, eff in results['interaction_effects'].items():
            sig = '*' if eff['significant'] else ''
            print(f"{h_col:<25} {eff['coefficient']:>+15.5f} {eff['std_error']:>10.5f} "
                  f"{eff['p_value']:>10.4f} {sig:>6} {eff['r_squared']:>8.4f} {eff['n_obs']:>8,}")

        print("\nInterpretation:")
        print("  - Interaction β > 0: AI penalty DECREASES with higher h-index")
        print("  - Interaction β < 0: AI penalty INCREASES with higher h-index")
        print("  - Interaction β ≈ 0: AI penalty is CONSTANT across h-index levels")

        # Check if any significant
        any_sig = any(e['significant'] for e in results['interaction_effects'].values())
        if not any_sig:
            print("\n  → No significant interactions: AI effect appears uniform across author reputation")
        else:
            sig_cols = [h for h, e in results['interaction_effects'].items() if e['significant']]
            print(f"\n  → Significant interactions found for: {', '.join(sig_cols)}")

    return results


def create_interaction_plots_terciles(
    df: pd.DataFrame,
    outcome: str = 'rating',
    h_index_cols: List[str] = ['first_author_h_index', 'last_author_h_index', 'mean_author_h_index'],
    ai_col: str = 'ai_percentage',
    ai_threshold: float = 50.0,
    n_quantiles: int = 3,
    figsize: Tuple[int, int] = (15, 5),
    output_dir: Optional[str] = None,
    verbose: bool = True,
    show_ci: bool = True,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    aggregate_to_paper: bool = True,
    paper_id_col: str = 'submission_number'
) -> Dict:
    """
    Create interaction plots using data-driven quantile bins (terciles by default).

    Instead of arbitrary bins, uses quantiles of the h-index distribution to create
    balanced groups. This avoids arbitrary cutoff choices while maintaining
    categorical interpretability.

    Parameters
    ----------
    df : DataFrame
        Data with reviews and author metrics
    outcome : str
        Outcome variable (rating, soundness, presentation)
    h_index_cols : list
        H-index columns to analyze
    ai_col : str
        AI percentage column
    ai_threshold : float
        Threshold for classifying AI vs Human papers
    n_quantiles : int
        Number of quantile groups (3=terciles, 4=quartiles, 5=quintiles)
    figsize : tuple
        Figure size
    output_dir : str, optional
        Directory to save figures
    verbose : bool
        Print summary
    show_ci : bool
        Show bootstrap confidence intervals
    n_bootstrap : int
        Number of bootstrap iterations
    ci_level : float
        Confidence level (default 0.95)
    aggregate_to_paper : bool
        If True, aggregate to paper-level first
    paper_id_col : str
        Column identifying unique papers

    Returns
    -------
    dict with plot data and statistics
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not available for plotting")
        return {}

    df = df.copy()

    # Classify AI vs Human
    ai_clean = pd.to_numeric(df[ai_col], errors='coerce')
    df['paper_AI'] = (ai_clean >= ai_threshold).astype(int)

    available_h_cols = [h for h in h_index_cols if h in df.columns]
    n_plots = len(available_h_cols)

    if n_plots == 0:
        warnings.warn("No h-index columns available for plotting")
        return {}

    # Aggregate to paper level if requested
    if aggregate_to_paper and paper_id_col in df.columns:
        agg_dict = {outcome: 'mean', 'paper_AI': 'first'}
        for h_col in available_h_cols:
            agg_dict[h_col] = 'first'

        paper_df = df.groupby(paper_id_col).agg(agg_dict).reset_index()

        if verbose:
            print(f"Aggregated to paper level: {len(df):,} reviews → {len(paper_df):,} papers")

        analysis_df = paper_df
    else:
        analysis_df = df

    # Quantile labels
    if n_quantiles == 3:
        quantile_labels = ['Low', 'Medium', 'High']
    elif n_quantiles == 4:
        quantile_labels = ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']
    elif n_quantiles == 5:
        quantile_labels = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    else:
        quantile_labels = [f'Q{i+1}' for i in range(n_quantiles)]

    results = {'plots': {}, 'data': {}, 'quantile_ranges': {}}

    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    for idx, h_col in enumerate(available_h_cols):
        ax = axes[idx]

        # Create quantile bins
        try:
            analysis_df[f'{h_col}_quantile'], bin_edges = pd.qcut(
                analysis_df[h_col],
                q=n_quantiles,
                labels=quantile_labels,
                retbins=True,
                duplicates='drop'
            )
        except ValueError:
            # If qcut fails due to too many duplicates, use cut with computed quantiles
            quantiles = analysis_df[h_col].quantile([i/n_quantiles for i in range(n_quantiles+1)]).values
            analysis_df[f'{h_col}_quantile'] = pd.cut(
                analysis_df[h_col],
                bins=quantiles,
                labels=quantile_labels[:len(quantiles)-1],
                include_lowest=True
            )
            bin_edges = quantiles

        # Store quantile ranges
        results['quantile_ranges'][h_col] = {
            'edges': bin_edges.tolist() if hasattr(bin_edges, 'tolist') else list(bin_edges),
            'labels': quantile_labels
        }

        # Calculate statistics by quantile and AI status
        bin_data = {
            'quantiles': [],
            'human_means': [], 'human_ci_lower': [], 'human_ci_upper': [], 'human_n': [],
            'ai_means': [], 'ai_ci_lower': [], 'ai_ci_upper': [], 'ai_n': [],
            'gaps': [], 'h_index_range': []
        }

        actual_labels = analysis_df[f'{h_col}_quantile'].dropna().unique()
        for q_label in quantile_labels:
            if q_label not in actual_labels.tolist():
                continue

            q_mask = analysis_df[f'{h_col}_quantile'] == q_label
            human_values = analysis_df.loc[q_mask & (analysis_df['paper_AI'] == 0), outcome].dropna().values
            ai_values = analysis_df.loc[q_mask & (analysis_df['paper_AI'] == 1), outcome].dropna().values

            # Get h-index range for this quantile
            h_vals = analysis_df.loc[q_mask, h_col].dropna()
            h_range = f"{h_vals.min():.0f}-{h_vals.max():.0f}" if len(h_vals) > 0 else "N/A"

            # Bootstrap CIs
            if show_ci:
                h_mean, h_ci_l, h_ci_u = _bootstrap_mean_ci(human_values, n_bootstrap, ci_level)
                a_mean, a_ci_l, a_ci_u = _bootstrap_mean_ci(ai_values, n_bootstrap, ci_level)
            else:
                h_mean = np.mean(human_values) if len(human_values) > 0 else np.nan
                a_mean = np.mean(ai_values) if len(ai_values) > 0 else np.nan
                h_ci_l, h_ci_u, a_ci_l, a_ci_u = np.nan, np.nan, np.nan, np.nan

            bin_data['quantiles'].append(q_label)
            bin_data['human_means'].append(h_mean)
            bin_data['human_ci_lower'].append(h_ci_l)
            bin_data['human_ci_upper'].append(h_ci_u)
            bin_data['human_n'].append(len(human_values))
            bin_data['ai_means'].append(a_mean)
            bin_data['ai_ci_lower'].append(a_ci_l)
            bin_data['ai_ci_upper'].append(a_ci_u)
            bin_data['ai_n'].append(len(ai_values))
            bin_data['h_index_range'].append(h_range)

            gap = h_mean - a_mean if not (np.isnan(h_mean) or np.isnan(a_mean)) else np.nan
            bin_data['gaps'].append(gap)

        # Plot
        x = np.arange(len(bin_data['quantiles']))
        human_means = np.array(bin_data['human_means'])
        ai_means = np.array(bin_data['ai_means'])

        if show_ci:
            human_yerr = [
                np.nan_to_num(human_means - np.array(bin_data['human_ci_lower']), nan=0),
                np.nan_to_num(np.array(bin_data['human_ci_upper']) - human_means, nan=0)
            ]
            ai_yerr = [
                np.nan_to_num(ai_means - np.array(bin_data['ai_ci_lower']), nan=0),
                np.nan_to_num(np.array(bin_data['ai_ci_upper']) - ai_means, nan=0)
            ]

            ax.errorbar(x, human_means, yerr=human_yerr,
                       fmt='o-', color='#2166AC', label='Human Papers',
                       markersize=8, linewidth=2, capsize=4, capthick=1.5)
            ax.errorbar(x, ai_means, yerr=ai_yerr,
                       fmt='s-', color='#B2182B', label='AI Papers',
                       markersize=8, linewidth=2, capsize=4, capthick=1.5)
        else:
            ax.plot(x, human_means, 'o-', color='#2166AC', label='Human Papers', markersize=8, linewidth=2)
            ax.plot(x, ai_means, 's-', color='#B2182B', label='AI Papers', markersize=8, linewidth=2)

        # Fill gap
        valid_mask = ~(np.isnan(human_means) | np.isnan(ai_means))
        if valid_mask.any():
            ax.fill_between(x[valid_mask], human_means[valid_mask], ai_means[valid_mask],
                           alpha=0.15, color='gray')

        # X-axis labels with h-index range and sample size
        total_n = [h + a for h, a in zip(bin_data['human_n'], bin_data['ai_n'])]
        x_labels = [f'{q}\n[{r}]\n(n={n:,})' for q, r, n in
                   zip(bin_data['quantiles'], bin_data['h_index_range'], total_n)]

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_xlabel(f'{h_col.replace("_", " ").title()} (Quantile [h-index range])', fontsize=9)
        ax.set_ylabel(f'Mean {outcome.title()}', fontsize=10)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Title with gap range
        valid_gaps = [g for g in bin_data['gaps'] if not np.isnan(g)]
        if valid_gaps:
            ax.set_title(f'{h_col.replace("_", " ").title()}\nAI Gap: {min(valid_gaps):.3f} to {max(valid_gaps):.3f}',
                        fontsize=10, fontweight='bold')

        results['data'][h_col] = bin_data

    # Overall title
    level_str = "Paper-Level" if aggregate_to_paper else "Review-Level"
    quantile_name = {3: 'Terciles', 4: 'Quartiles', 5: 'Quintiles'}.get(n_quantiles, f'{n_quantiles}-tiles')
    plt.suptitle(
        f'{outcome.title()} by AI Status and Author Reputation ({quantile_name}, {level_str})\n'
        f'Data-driven bins based on h-index distribution',
        fontsize=11, fontweight='bold', y=1.02
    )
    plt.tight_layout()

    # Save figure
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        fig_path = f'{output_dir}/interaction_terciles_{outcome}.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
        results['figure_path'] = fig_path
        if verbose:
            print(f"Saved: {fig_path}")

    results['figure'] = fig

    # Print summary
    if verbose:
        print("\n" + "=" * 70)
        print(f"QUANTILE-BASED INTERACTION ANALYSIS: {outcome.upper()} ({level_str})")
        print(f"Using {quantile_name} (data-driven bins)")
        print("=" * 70)

        for h_col, data in results['data'].items():
            print(f"\n{h_col}:")
            print(f"  {'Quantile':<12} {'H-index':<12} {'Human':>10} {'AI':>10} {'Gap':>10} {'N':>8}")
            print(f"  {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
            for i, q in enumerate(data['quantiles']):
                h_range = data['h_index_range'][i]
                h_m = data['human_means'][i]
                a_m = data['ai_means'][i]
                gap = data['gaps'][i]
                n = data['human_n'][i] + data['ai_n'][i]
                print(f"  {q:<12} {h_range:<12} {h_m:>10.3f} {a_m:>10.3f} {gap:>+10.3f} {n:>8,}")

    return results


def run_full_interaction_analysis(
    reviews_df: pd.DataFrame,
    submissions_df: pd.DataFrame,
    enriched_df: pd.DataFrame,
    components: List[str] = ['rating', 'soundness', 'presentation', 'contribution'],
    h_index_cols: List[str] = ['first_author_h_index', 'last_author_h_index', 'mean_author_h_index'],
    output_dir: str = '.',
    verbose: bool = True,
    save_plots: bool = True,
    show_ci: bool = True,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95
) -> Dict:
    """
    Run comprehensive interaction analysis across components and h-index measures.

    This is the main entry point for the extended interaction analysis.

    Parameters
    ----------
    reviews_df : DataFrame
        Review-level data
    submissions_df : DataFrame
        Submission-level data
    enriched_df : DataFrame
        Author-enriched data
    components : list
        Components to analyze
    h_index_cols : list
        H-index measures to test
    output_dir : str
        Output directory for plots and tables
    verbose : bool
        Print results
    save_plots : bool
        Save interaction plots
    show_ci : bool
        Show bootstrap confidence intervals on plots
    n_bootstrap : int
        Number of bootstrap iterations for CI
    ci_level : float
        Confidence level (default 0.95 for 95% CI)

    Returns
    -------
    dict with all results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print("\n" + "=" * 70)
        print("FULL INTERACTION ANALYSIS")
        print("Components × H-index Measures")
        print("=" * 70)

    # Merge data
    merged = merge_author_data(reviews_df, submissions_df, enriched_df)
    merged = create_analysis_variables(merged)

    results = {
        'component_interactions': None,
        'plots': {},
        'summary': {}
    }

    # 1. Run component × h-index interaction models
    if verbose:
        print("\n1. Running component × h-index interaction models...")

    results['component_interactions'] = run_component_interaction_analysis(
        df=merged,
        components=[c for c in components if c != 'rating'],  # rating is avg_rating at submission level
        h_index_cols=h_index_cols,
        verbose=verbose
    )

    # 2. Create interaction plots for each outcome
    if save_plots:
        if verbose:
            print("\n2. Creating interaction plots...")
            if show_ci:
                print(f"   (Bootstrap {int(ci_level*100)}% CI, {n_bootstrap} iterations)")

        for outcome in ['rating', 'soundness', 'presentation']:
            if outcome in merged.columns:
                plot_results = create_interaction_plots(
                    df=merged,
                    outcome=outcome,
                    h_index_cols=h_index_cols,
                    output_dir=output_dir,
                    verbose=verbose,
                    show_ci=show_ci,
                    n_bootstrap=n_bootstrap,
                    ci_level=ci_level
                )
                results['plots'][outcome] = plot_results

    # 3. Save summary table
    if results['component_interactions'] and results['component_interactions'].get('summary_table') is not None:
        summary_path = f'{output_dir}/component_interaction_summary.csv'
        results['component_interactions']['summary_table'].to_csv(summary_path, index=False)
        if verbose:
            print(f"\nSaved summary to: {summary_path}")

    # 4. Generate key insights
    if verbose:
        print("\n" + "=" * 70)
        print("KEY FINDINGS")
        print("=" * 70)

        summary_df = results['component_interactions'].get('summary_table')
        if summary_df is not None:
            # Count significant interactions
            n_sig = summary_df['significant'].sum()
            n_total = len(summary_df)
            print(f"\nSignificant interactions: {n_sig}/{n_total}")

            # By component
            print("\nBy component:")
            for comp in summary_df['component'].unique():
                comp_df = summary_df[summary_df['component'] == comp]
                avg_int = comp_df['interaction_coef'].mean()
                n_sig_comp = comp_df['significant'].sum()
                print(f"  {comp}: avg interaction = {avg_int:+.4f}, {n_sig_comp}/{len(comp_df)} significant")

            # By h-index measure
            print("\nBy h-index measure:")
            for h_measure in summary_df['h_index_measure'].unique():
                h_df = summary_df[summary_df['h_index_measure'] == h_measure]
                avg_int = h_df['interaction_coef'].mean()
                n_sig_h = h_df['significant'].sum()
                print(f"  {h_measure}: avg interaction = {avg_int:+.4f}, {n_sig_h}/{len(h_df)} significant")

    return results


# =============================================================================
# H-INDEX INTERACTION ROBUSTNESS TABLE (QJE/REStud-Ready)
# =============================================================================

def test_hindex_interactions_robust(
    df: pd.DataFrame,
    outcomes: List[str] = ['rating', 'soundness', 'presentation'],
    h_index_cols: List[str] = ['first_author_h_index', 'last_author_h_index', 'mean_author_h_index'],
    ai_col: str = 'ai_percentage',  # Match figure function default
    paper_id_col: str = 'submission_number',
    aggregate_to_paper: bool = True,
    fdr_method: str = 'fdr_bh',
    alpha: float = 0.05,
    verbose: bool = True,
    create_latex: bool = True
) -> Dict:
    """
    QJE/REStud-ready test for AI × H-index interactions with multiple testing correction.

    This function:
    1. Tests all outcome × h-index combinations
    2. Applies FDR correction (Benjamini-Hochberg)
    3. Generates publication-ready robustness table
    4. Pre-specifies first author h-index as primary hypothesis

    Model for each test:
        outcome ~ AI + h_index + AI × h_index

    Parameters
    ----------
    df : DataFrame
        Merged review-level data with author info
    outcomes : list
        Outcome variables to test
    h_index_cols : list
        H-index measures to test
    ai_col : str
        Binary AI indicator column
    paper_id_col : str
        Paper identifier for aggregation
    aggregate_to_paper : bool
        If True, aggregate to paper-level first
    fdr_method : str
        Multiple testing correction method ('fdr_bh', 'bonferroni', 'holm')
    alpha : float
        Significance level
    verbose : bool
        Print detailed output
    create_latex : bool
        Generate LaTeX table

    Returns
    -------
    dict with:
        - results_df: DataFrame with all test results
        - primary_test: Results for pre-specified primary test (first author × rating)
        - latex_table: LaTeX formatted table
        - summary: Text summary for paper
    """
    if verbose:
        print("\n" + "=" * 70)
        print("H-INDEX INTERACTION ANALYSIS (QJE/REStud-Ready)")
        print("=" * 70)
        print(f"\nPrimary hypothesis (pre-specified):")
        print(f"  First author h-index moderates the AI effect on rating")
        print(f"\nSecondary tests (exploratory):")
        print(f"  All {len(outcomes)} outcomes × {len(h_index_cols)} h-index measures")
        print(f"\nMultiple testing correction: {fdr_method.upper()}")

    # =========================================================================
    # STEP 1: Prepare data - EXACTLY matching figure function logic
    # =========================================================================
    df = df.copy()  # Don't filter anything yet - match figure behavior

    # Convert AI column to binary indicator
    # Check if it's already binary or needs conversion from percentage
    ai_clean = pd.to_numeric(df[ai_col], errors='coerce')

    if ai_clean.max() > 1:
        # It's a percentage (0-100), convert to binary at threshold 50
        # This matches figure function (lines 1691-1692)
        df['paper_AI'] = (ai_clean >= 50).astype(int)
    else:
        # Already binary (0/1), use as-is
        df['paper_AI'] = ai_clean.fillna(0).astype(int)

    ai_var = 'paper_AI'

    # Aggregate to paper level if requested
    # Include all potential columns, NaN handling happens per-test
    if aggregate_to_paper and paper_id_col in df.columns:
        agg_dict = {ai_var: 'first'}
        for outcome in outcomes:
            if outcome in df.columns:
                agg_dict[outcome] = 'mean'
        for h_col in h_index_cols:
            if h_col in df.columns:
                agg_dict[h_col] = 'first'

        paper_df = df.groupby(paper_id_col).agg(agg_dict).reset_index()

        if verbose:
            print(f"\nAggregated to paper level: {len(df):,} reviews → {len(paper_df):,} papers")
    else:
        paper_df = df

    # =========================================================================
    # STEP 2: Run all interaction tests
    # =========================================================================
    all_results = []

    for outcome in outcomes:
        if outcome not in paper_df.columns:
            continue

        for h_col in h_index_cols:
            if h_col not in paper_df.columns:
                continue

            # Prepare data for this test
            test_df = paper_df.dropna(subset=[outcome, h_col, ai_var]).copy()

            if len(test_df) < 50:
                continue

            # Store h-index stats for potential standardization reporting
            h_mean = test_df[h_col].mean()
            h_std = test_df[h_col].std()

            # Run interaction regression with RAW (unstandardized) h-index
            # This gives coefficients that are directly interpretable:
            # "change in outcome per 1-point increase in h-index"
            try:
                formula = f'{outcome} ~ {ai_var} * {h_col}'
                model = smf.ols(formula, data=test_df).fit(cov_type='HC3')

                # Extract interaction term
                interaction_key = f'{ai_var}:{h_col}'
                if interaction_key not in model.params:
                    # Try alternative naming
                    interaction_key = f'{h_col}:{ai_var}'

                if interaction_key in model.params:
                    interaction_coef = model.params[interaction_key]
                    interaction_se = model.bse[interaction_key]
                    interaction_p = model.pvalues[interaction_key]
                else:
                    continue

                # Also get main effects
                ai_coef = model.params.get(ai_var, np.nan)
                ai_p = model.pvalues.get(ai_var, np.nan)

                # Compute effect at different h-index levels (using unstandardized coefficients)
                # Model: outcome ~ AI + h_index + AI*h_index
                # AI effect at h-index = h: ai_coef + interaction_coef * h
                # At mean h-index: effect = ai_coef + interaction_coef * h_mean
                # At +1 SD h-index: effect = ai_coef + interaction_coef * (h_mean + h_std)
                # At -1 SD h-index: effect = ai_coef + interaction_coef * (h_mean - h_std)
                effect_at_mean = ai_coef + interaction_coef * h_mean
                effect_at_high = ai_coef + interaction_coef * (h_mean + h_std)
                effect_at_low = ai_coef + interaction_coef * (h_mean - h_std)

                all_results.append({
                    'outcome': outcome,
                    'h_index_measure': h_col.replace('_h_index', '').replace('_author', ' author').title(),
                    'h_index_col': h_col,
                    'n_papers': len(test_df),
                    'interaction_coef': interaction_coef,
                    'interaction_se': interaction_se,
                    'interaction_p': interaction_p,
                    'ai_main_effect': ai_coef,
                    'ai_main_p': ai_p,
                    'effect_at_low_h': effect_at_low,
                    'effect_at_mean_h': effect_at_mean,
                    'effect_at_high_h': effect_at_high,
                    'h_index_mean': h_mean,
                    'h_index_std': h_std,
                    'r_squared': model.rsquared
                })

            except Exception as e:
                if verbose:
                    print(f"  Warning: {outcome} × {h_col} failed: {e}")

    # =========================================================================
    # STEP 3: Apply multiple testing correction
    # =========================================================================
    results_df = pd.DataFrame(all_results)

    if len(results_df) == 0:
        if verbose:
            print("\nNo valid tests completed!")
        return {'results_df': None, 'primary_test': None}

    # Apply FDR correction
    p_values = results_df['interaction_p'].values
    rejected, p_adjusted, _, _ = multipletests(p_values, alpha=alpha, method=fdr_method)

    results_df['p_adjusted'] = p_adjusted
    results_df['significant_raw'] = results_df['interaction_p'] < alpha
    results_df['significant_fdr'] = rejected

    # =========================================================================
    # STEP 4: Identify primary test result
    # =========================================================================
    primary_mask = (
        (results_df['outcome'] == 'rating') &
        (results_df['h_index_col'] == 'first_author_h_index')
    )

    primary_test = None
    if primary_mask.any():
        primary_test = results_df[primary_mask].iloc[0].to_dict()

    # =========================================================================
    # STEP 5: Generate output
    # =========================================================================
    if verbose:
        print("\n" + "-" * 70)
        print("RESULTS SUMMARY")
        print("-" * 70)

        # Summary statistics
        n_tests = len(results_df)
        n_sig_raw = results_df['significant_raw'].sum()
        n_sig_fdr = results_df['significant_fdr'].sum()

        print(f"\nTotal tests: {n_tests}")
        print(f"Significant (raw p < {alpha}): {n_sig_raw}")
        print(f"Significant (FDR-corrected): {n_sig_fdr}")

        # Primary test result
        if primary_test:
            print("\n" + "-" * 70)
            print("PRIMARY TEST (Pre-specified)")
            print("-" * 70)
            print(f"\nOutcome: Rating")
            print(f"H-index: First Author")
            print(f"N: {primary_test['n_papers']:,} papers")
            print(f"\nInteraction (AI × H-index):")
            print(f"  β = {primary_test['interaction_coef']:+.4f}")
            print(f"  SE = {primary_test['interaction_se']:.4f}")
            print(f"  p = {primary_test['interaction_p']:.4f}")
            print(f"  p (FDR-adjusted) = {primary_test['p_adjusted']:.4f}")

            if primary_test['interaction_coef'] > 0:
                print(f"\n→ POSITIVE interaction: AI penalty is SMALLER for high h-index authors")
                print(f"   At low h-index (-1 SD):  AI effect = {primary_test['effect_at_low_h']:+.3f}")
                print(f"   At mean h-index:         AI effect = {primary_test['effect_at_mean_h']:+.3f}")
                print(f"   At high h-index (+1 SD): AI effect = {primary_test['effect_at_high_h']:+.3f}")
            else:
                print(f"\n→ NEGATIVE interaction: AI penalty is LARGER for high h-index authors")

            if primary_test['significant_fdr']:
                print(f"\n✓ SIGNIFICANT after FDR correction")
            elif primary_test['significant_raw']:
                print(f"\n⚠ Significant at raw p < {alpha}, but NOT after FDR correction")
            else:
                print(f"\n✗ Not significant")

        # Full results table
        print("\n" + "-" * 70)
        print("FULL RESULTS TABLE")
        print("-" * 70)

        display_cols = ['outcome', 'h_index_measure', 'interaction_coef', 'interaction_p', 'p_adjusted', 'significant_fdr']
        print(results_df[display_cols].to_string(index=False))

    # =========================================================================
    # STEP 6: Generate LaTeX table
    # =========================================================================
    latex_table = None
    if create_latex:
        latex_table = _create_hindex_interaction_latex(results_df, primary_test, alpha)

        if verbose:
            print("\n" + "-" * 70)
            print("LATEX TABLE")
            print("-" * 70)
            print(latex_table)

    # =========================================================================
    # STEP 7: Generate text summary for paper
    # =========================================================================
    summary_text = _generate_hindex_summary(results_df, primary_test, fdr_method, alpha)

    return {
        'results_df': results_df,
        'primary_test': primary_test,
        'latex_table': latex_table,
        'summary_text': summary_text,
        'n_tests': len(results_df),
        'n_significant_raw': results_df['significant_raw'].sum(),
        'n_significant_fdr': results_df['significant_fdr'].sum()
    }


def _create_hindex_interaction_latex(results_df: pd.DataFrame, primary_test: Dict, alpha: float) -> str:
    """Generate publication-quality LaTeX table for h-index interactions."""

    # Pivot to wide format: outcomes as rows, h-index measures as columns
    table_data = []

    for outcome in results_df['outcome'].unique():
        row = {'Outcome': outcome.capitalize()}

        for h_col in results_df['h_index_col'].unique():
            mask = (results_df['outcome'] == outcome) & (results_df['h_index_col'] == h_col)
            if mask.any():
                res = results_df[mask].iloc[0]
                coef = res['interaction_coef']
                p = res['interaction_p']
                p_adj = res['p_adjusted']

                # Significance stars using RAW p-values to match figure output
                # ** p < 0.01, * p < 0.05 (raw, not FDR-adjusted)
                if p < 0.01:
                    stars = '**'
                elif p < 0.05:
                    stars = '*'
                else:
                    stars = ''

                h_name = h_col.replace('_h_index', '').replace('_author', '').title()
                row[h_name] = f'{coef:+.4f}{stars}'
            else:
                h_name = h_col.replace('_h_index', '').replace('_author', '').title()
                row[h_name] = '--'

        table_data.append(row)

    table_df = pd.DataFrame(table_data)

    # Generate LaTeX
    latex = r"""\begin{table}[htbp]
\centering
\caption{AI × H-Index Interaction Effects on Review Scores}
\label{tab:hindex_interactions}
\begin{threeparttable}
"""

    # Determine columns
    cols = ['Outcome'] + [c for c in table_df.columns if c != 'Outcome']
    n_cols = len(cols)

    latex += r'\begin{tabular}{l' + 'c' * (n_cols - 1) + r'}' + '\n'
    latex += r'\toprule' + '\n'

    # Header
    header = ' & '.join([r'\textbf{' + c + '}' for c in cols])
    latex += header + r' \\' + '\n'
    latex += r'\midrule' + '\n'

    # Data rows
    for _, row in table_df.iterrows():
        row_str = ' & '.join([str(row[c]) for c in cols])
        latex += row_str + r' \\' + '\n'

    latex += r'\bottomrule' + '\n'
    latex += r'\end{tabular}' + '\n'

    # Notes
    latex += r"""\begin{tablenotes}
\small
\item \textit{Notes:} Each cell shows the coefficient on AI × H-index (unstandardized) from an OLS regression with heteroskedasticity-robust standard errors.
$^{**} p < 0.01$, $^{*} p < 0.05$.
Positive coefficients indicate that the AI penalty is smaller for higher h-index authors.
\end{tablenotes}
\end{threeparttable}
\end{table}"""

    return latex


def _generate_hindex_summary(results_df: pd.DataFrame, primary_test: Dict, fdr_method: str, alpha: float) -> str:
    """Generate text summary for paper."""

    n_tests = len(results_df)
    n_sig_raw = results_df['significant_raw'].sum()
    n_sig_fdr = results_df['significant_fdr'].sum()

    summary = []
    summary.append("H-INDEX INTERACTION ANALYSIS SUMMARY")
    summary.append("=" * 50)

    if primary_test:
        coef = primary_test['interaction_coef']
        p = primary_test['interaction_p']
        p_adj = primary_test['p_adjusted']
        n = primary_test['n_papers']

        summary.append(f"\nPre-specified primary test (Rating × First Author H-index):")
        summary.append(f"  N = {n:,} papers")
        summary.append(f"  Interaction coefficient: β = {coef:+.4f}")
        summary.append(f"  Raw p-value: {p:.4f}")
        summary.append(f"  FDR-adjusted p-value: {p_adj:.4f}")

        if coef > 0 and p_adj < alpha:
            summary.append(f"\n  INTERPRETATION: The AI penalty is significantly smaller for")
            summary.append(f"  papers with high h-index first authors (p = {p_adj:.3f}, FDR-corrected).")
            summary.append(f"  A one-SD increase in first author h-index reduces the AI penalty by")
            summary.append(f"  {abs(coef):.2f} rating points.")
        elif coef > 0 and p < alpha:
            summary.append(f"\n  INTERPRETATION: There is suggestive evidence that the AI penalty")
            summary.append(f"  is smaller for high h-index first authors (p = {p:.3f}), though")
            summary.append(f"  this does not survive FDR correction for multiple testing.")

    summary.append(f"\nRobustness ({n_tests} tests across outcomes and h-index measures):")
    summary.append(f"  Significant before correction: {n_sig_raw}/{n_tests}")
    summary.append(f"  Significant after FDR correction: {n_sig_fdr}/{n_tests}")

    return '\n'.join(summary)


def create_hindex_interaction_figure(
    df: pd.DataFrame,
    outcome: str = 'rating',
    h_index_col: str = 'first_author_h_index',
    ai_col: str = 'paper_AI',
    paper_id_col: str = 'submission_number',
    aggregate_to_paper: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    verbose: bool = True
) -> plt.Figure:
    """
    Create publication-quality figure for h-index interaction.

    Shows AI effect at different h-index levels with proper confidence intervals.
    """
    # Prepare data
    clean = df.dropna(subset=[outcome, h_index_col, ai_col]).copy()

    if aggregate_to_paper and paper_id_col in clean.columns:
        agg_dict = {outcome: 'mean', ai_col: 'first', h_index_col: 'first'}
        paper_df = clean.groupby(paper_id_col).agg(agg_dict).reset_index()
    else:
        paper_df = clean

    # Create terciles
    paper_df['h_tercile'] = pd.qcut(paper_df[h_index_col], q=3, labels=['Low', 'Medium', 'High'])

    # Calculate AI effect in each tercile
    tercile_results = []
    for tercile in ['Low', 'Medium', 'High']:
        tercile_df = paper_df[paper_df['h_tercile'] == tercile]

        # Get mean and SE for AI vs non-AI
        ai_papers = tercile_df[tercile_df[ai_col] >= 0.5][outcome]
        human_papers = tercile_df[tercile_df[ai_col] < 0.5][outcome]

        if len(ai_papers) > 5 and len(human_papers) > 5:
            diff = ai_papers.mean() - human_papers.mean()
            se = np.sqrt(ai_papers.var()/len(ai_papers) + human_papers.var()/len(human_papers))
            t_stat, p_val = scipy_stats.ttest_ind(ai_papers, human_papers)

            # H-index range for this tercile
            h_min = tercile_df[h_index_col].min()
            h_max = tercile_df[h_index_col].max()

            tercile_results.append({
                'tercile': tercile,
                'ai_effect': diff,
                'se': se,
                'ci_low': diff - 1.96 * se,
                'ci_high': diff + 1.96 * se,
                'p_value': p_val,
                'n_ai': len(ai_papers),
                'n_human': len(human_papers),
                'h_range': f'{h_min:.0f}-{h_max:.0f}'
            })

    tercile_df = pd.DataFrame(tercile_results)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(tercile_df))
    colors = ['#e74c3c' if p < 0.05 else '#95a5a6' for p in tercile_df['p_value']]

    bars = ax.bar(x, tercile_df['ai_effect'],
                  yerr=[tercile_df['ai_effect'] - tercile_df['ci_low'],
                        tercile_df['ci_high'] - tercile_df['ai_effect']],
                  color=colors, edgecolor='black', capsize=8, linewidth=1.5)

    ax.axhline(0, color='black', linewidth=1, linestyle='-')

    # Labels with h-index ranges
    labels = [f"{row['tercile']}\n(h: {row['h_range']})" for _, row in tercile_df.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_xlabel(f'{h_index_col.replace("_", " ").title()} Tercile', fontsize=11)
    ax.set_ylabel(f'AI Effect on {outcome.capitalize()}\n(AI - Human papers)', fontsize=11)
    ax.set_title(f'AI Effect by Author Experience ({outcome.capitalize()})', fontsize=12, fontweight='bold')

    # Add significance stars
    for i, (bar, row) in enumerate(zip(bars, tercile_df.itertuples())):
        p = row.p_value
        if p < 0.001:
            sig = '***'
        elif p < 0.01:
            sig = '**'
        elif p < 0.05:
            sig = '*'
        else:
            sig = ''

        if sig:
            height = bar.get_height()
            offset = row.ci_high - row.ai_effect + 0.02 if height < 0 else row.ci_high - row.ai_effect + 0.02
            ax.annotate(sig, xy=(bar.get_x() + bar.get_width()/2, row.ci_high + 0.02),
                        ha='center', fontsize=14, fontweight='bold')

    # Add sample sizes
    for i, row in tercile_df.iterrows():
        ax.annotate(f'n={row["n_ai"]+row["n_human"]}',
                    xy=(i, tercile_df['ci_low'].min() - 0.05),
                    ha='center', fontsize=9, color='gray')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"Figure saved to: {save_path}")

    return fig


if __name__ == "__main__":
    print("Selection Robustness Analysis Module")
    print("Run with: run_selection_robustness_analysis(reviews_df, submissions_df, enriched_df)")
    print("For extended analysis: run_full_interaction_analysis(reviews_df, submissions_df, enriched_df)")
    print("For h-index interaction table: test_hindex_interactions_robust(merged_df)")
