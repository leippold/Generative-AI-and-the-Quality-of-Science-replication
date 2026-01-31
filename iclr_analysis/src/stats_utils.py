"""
Statistical Utilities for ICLR Analysis - STATE OF THE ART
===========================================================

Includes:
- Robust inference with clustered standard errors
- Bootstrap confidence intervals
- Permutation tests for interaction effects
- Effect sizes (Cohen's d, rank-biserial, eta-squared, omega-squared)
- Multiple comparison corrections (Bonferroni, FDR)
- Mixed effects models for nested data
- Ordinal regression for rating outcomes
- Weighted statistics for confidence-weighted analysis

METHODOLOGICAL NOTES:
- Reviews are clustered within submissions → use clustered SEs or mixed models
- Ratings are ordinal → ordinal regression preferred over OLS
- Multiple comparisons → FDR correction recommended
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    spearmanr, kendalltau, pearsonr,
    mannwhitneyu, kruskal, wilcoxon,
    chi2_contingency, levene, ttest_ind, ttest_1samp
)
import warnings

from .constants import N_BOOTSTRAP, N_PERMUTATIONS, RANDOM_SEED, ALPHA


# =============================================================================
# EFFECT SIZES
# =============================================================================

def cohens_d(group1, group2):
    """
    Calculate Cohen's d effect size (standardized mean difference).
    
    Uses pooled standard deviation (Hedges' correction available separately).
    
    Parameters
    ----------
    group1, group2 : array-like
    
    Returns
    -------
    float : Cohen's d
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def hedges_g(group1, group2):
    """
    Calculate Hedges' g (bias-corrected Cohen's d).
    
    Better for small samples.
    """
    d = cohens_d(group1, group2)
    n = len(group1) + len(group2)
    
    # Hedges' correction factor
    correction = 1 - (3 / (4 * (n - 2) - 1))
    
    return d * correction


def rank_biserial_correlation(group1, group2):
    """
    Calculate rank-biserial correlation from Mann-Whitney U.
    
    Effect size for non-parametric comparison.
    r = 1 - (2U)/(n1*n2)
    
    Returns
    -------
    float : r in [-1, 1], positive means group1 > group2
    """
    n1, n2 = len(group1), len(group2)
    u_stat, _ = mannwhitneyu(group1, group2, alternative='two-sided')
    
    return 1 - (2 * u_stat) / (n1 * n2)


def epsilon_squared(h_stat, n):
    """
    Calculate epsilon-squared effect size for Kruskal-Wallis.
    
    ε² = H / (n - 1)
    """
    return h_stat / (n - 1)


def eta_squared(ss_between, ss_total):
    """
    Calculate eta-squared effect size for ANOVA.
    
    η² = SS_between / SS_total
    """
    return ss_between / ss_total


def omega_squared_from_f(f_stat, df_between, df_within):
    """
    Calculate omega-squared (less biased than eta-squared).
    
    ω² = (SS_between - df_between * MS_within) / (SS_total + MS_within)
    """
    # Approximation from F statistic
    return (f_stat - 1) / (f_stat + (df_within + 1) / df_between)


def cliffs_delta(group1, group2):
    """
    Calculate Cliff's Delta (non-parametric effect size).
    
    Proportion of pairs where x1 > x2 minus proportion where x1 < x2.
    """
    n1, n2 = len(group1), len(group2)
    
    greater = sum(1 for x1 in group1 for x2 in group2 if x1 > x2)
    less = sum(1 for x1 in group1 for x2 in group2 if x1 < x2)
    
    return (greater - less) / (n1 * n2)


# =============================================================================
# BASIC STATISTICAL TESTS
# =============================================================================

def mann_whitney_test(group1, group2, alternative='two-sided'):
    """
    Mann-Whitney U test with effect size.
    
    Returns
    -------
    dict with u_stat, p_value, effect_size (rank-biserial), mean_diff
    """
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)
    
    u_stat, p_value = mannwhitneyu(group1, group2, alternative=alternative)
    effect_size = rank_biserial_correlation(group1, group2)
    
    return {
        'u_stat': u_stat,
        'p_value': p_value,
        'effect_size': effect_size,
        'effect_size_name': 'rank_biserial_r',
        'mean_diff': np.mean(group1) - np.mean(group2),
        'n1': len(group1),
        'n2': len(group2)
    }


def kruskal_wallis_test(groups, group_labels=None):
    """
    Kruskal-Wallis H-test with effect size.
    
    Parameters
    ----------
    groups : list of arrays
    group_labels : list of str, optional
    
    Returns
    -------
    dict with h_stat, p_value, effect_size (epsilon-squared)
    """
    groups = [np.asarray(g) for g in groups if len(g) > 0]
    
    if len(groups) < 2:
        return {'h_stat': np.nan, 'p_value': np.nan, 'effect_size': np.nan}
    
    h_stat, p_value = kruskal(*groups)
    n_total = sum(len(g) for g in groups)
    effect_size = epsilon_squared(h_stat, n_total)
    
    return {
        'h_stat': h_stat,
        'p_value': p_value,
        'effect_size': effect_size,
        'effect_size_name': 'epsilon_squared',
        'n_groups': len(groups),
        'n_total': n_total
    }


def chi_square_test(contingency_table):
    """
    Chi-square test of independence with effect size (Cramér's V).
    
    Parameters
    ----------
    contingency_table : DataFrame or 2D array
    
    Returns
    -------
    dict with chi2, p_value, cramers_v
    """
    if isinstance(contingency_table, pd.DataFrame):
        contingency_table = contingency_table.values
    
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    n = contingency_table.sum()
    min_dim = min(contingency_table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
    
    return {
        'chi2': chi2,
        'p_value': p_value,
        'dof': dof,
        'cramers_v': cramers_v,
        'n': n
    }


def levene_test(group1, group2):
    """
    Levene's test for equality of variances.
    
    Returns
    -------
    dict with statistic, p_value, var_ratio
    """
    stat, p_value = levene(group1, group2)
    
    return {
        'statistic': stat,
        'p_value': p_value,
        'var1': np.var(group1, ddof=1),
        'var2': np.var(group2, ddof=1),
        'var_ratio': np.var(group1, ddof=1) / np.var(group2, ddof=1)
    }


# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

def bootstrap_ci(data, statistic=np.mean, n_bootstrap=N_BOOTSTRAP, 
                 confidence=0.95, random_state=RANDOM_SEED):
    """
    Bootstrap confidence interval for a statistic.
    
    Parameters
    ----------
    data : array-like
    statistic : callable
        Function to compute statistic (default: np.mean)
    n_bootstrap : int
    confidence : float
    random_state : int
    
    Returns
    -------
    dict with estimate, ci_lower, ci_upper, se
    """
    rng = np.random.RandomState(random_state)
    data = np.asarray(data)
    n = len(data)
    
    # Bootstrap resampling
    boot_stats = np.array([
        statistic(data[rng.randint(0, n, n)])
        for _ in range(n_bootstrap)
    ])
    
    alpha = 1 - confidence
    ci_lower = np.percentile(boot_stats, 100 * alpha / 2)
    ci_upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    
    return {
        'estimate': statistic(data),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'se': np.std(boot_stats, ddof=1),
        'n_bootstrap': n_bootstrap
    }


def bootstrap_diff_ci(group1, group2, statistic=np.mean, n_bootstrap=N_BOOTSTRAP,
                      confidence=0.95, random_state=RANDOM_SEED):
    """
    Bootstrap confidence interval for difference between two groups.
    
    Returns
    -------
    dict with diff, ci_lower, ci_upper, se, p_value (from bootstrap)
    """
    rng = np.random.RandomState(random_state)
    group1, group2 = np.asarray(group1), np.asarray(group2)
    n1, n2 = len(group1), len(group2)
    
    observed_diff = statistic(group1) - statistic(group2)
    
    boot_diffs = []
    for _ in range(n_bootstrap):
        boot1 = group1[rng.randint(0, n1, n1)]
        boot2 = group2[rng.randint(0, n2, n2)]
        boot_diffs.append(statistic(boot1) - statistic(boot2))
    
    boot_diffs = np.array(boot_diffs)
    
    alpha = 1 - confidence
    ci_lower = np.percentile(boot_diffs, 100 * alpha / 2)
    ci_upper = np.percentile(boot_diffs, 100 * (1 - alpha / 2))
    
    # Bootstrap p-value (two-sided)
    p_value = 2 * min(
        np.mean(boot_diffs >= 0),
        np.mean(boot_diffs <= 0)
    )
    
    return {
        'diff': observed_diff,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'se': np.std(boot_diffs, ddof=1),
        'p_value': p_value,
        'n_bootstrap': n_bootstrap
    }


# =============================================================================
# BIAS-CORRECTED ACCELERATED (BCa) BOOTSTRAP
# =============================================================================

def bootstrap_bca_ci(data, statistic=np.mean, n_bootstrap=N_BOOTSTRAP,
                     confidence=0.95, random_state=RANDOM_SEED):
    """
    Bias-Corrected and Accelerated (BCa) Bootstrap Confidence Interval.

    BCa adjusts for both bias and skewness in the bootstrap distribution,
    providing better coverage than percentile bootstrap, especially for
    small samples or skewed statistics.

    Parameters
    ----------
    data : array-like
    statistic : callable
        Function to compute statistic (default: np.mean)
    n_bootstrap : int
    confidence : float
    random_state : int

    Returns
    -------
    dict with estimate, ci_lower, ci_upper, ci_lower_percentile, ci_upper_percentile,
         bias_correction, acceleration, se

    References
    ----------
    Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap.
    """
    rng = np.random.RandomState(random_state)
    data = np.asarray(data)
    n = len(data)

    # Original estimate
    theta_hat = statistic(data)

    # Bootstrap distribution
    boot_stats = np.array([
        statistic(data[rng.randint(0, n, n)])
        for _ in range(n_bootstrap)
    ])

    # Percentile CI (for comparison)
    alpha = 1 - confidence
    ci_lower_pct = np.percentile(boot_stats, 100 * alpha / 2)
    ci_upper_pct = np.percentile(boot_stats, 100 * (1 - alpha / 2))

    # =========================================================================
    # BCa Correction
    # =========================================================================

    # 1. Bias correction (z0)
    # Proportion of bootstrap estimates less than original estimate
    prop_less = np.mean(boot_stats < theta_hat)
    # Avoid edge cases
    prop_less = np.clip(prop_less, 1e-10, 1 - 1e-10)
    z0 = stats.norm.ppf(prop_less)

    # 2. Acceleration (a) via jackknife
    # Jackknife estimates (leave-one-out)
    jackknife_stats = np.array([
        statistic(np.delete(data, i))
        for i in range(n)
    ])
    jack_mean = np.mean(jackknife_stats)

    # Acceleration factor
    numerator = np.sum((jack_mean - jackknife_stats) ** 3)
    denominator = 6 * (np.sum((jack_mean - jackknife_stats) ** 2) ** 1.5)

    if denominator == 0:
        a = 0
    else:
        a = numerator / denominator

    # 3. BCa adjusted percentiles
    z_alpha_lower = stats.norm.ppf(alpha / 2)
    z_alpha_upper = stats.norm.ppf(1 - alpha / 2)

    # Adjusted percentiles
    def bca_percentile(z_alpha):
        num = z0 + z_alpha
        denom = 1 - a * num
        if denom == 0:
            return stats.norm.cdf(z0 + z_alpha)
        return stats.norm.cdf(z0 + num / denom)

    p_lower = bca_percentile(z_alpha_lower)
    p_upper = bca_percentile(z_alpha_upper)

    # Clip to valid range
    p_lower = np.clip(p_lower, 0.001, 0.999)
    p_upper = np.clip(p_upper, 0.001, 0.999)

    ci_lower_bca = np.percentile(boot_stats, 100 * p_lower)
    ci_upper_bca = np.percentile(boot_stats, 100 * p_upper)

    return {
        'estimate': theta_hat,
        'ci_lower': ci_lower_bca,
        'ci_upper': ci_upper_bca,
        'ci_lower_percentile': ci_lower_pct,
        'ci_upper_percentile': ci_upper_pct,
        'bias_correction': z0,
        'acceleration': a,
        'se': np.std(boot_stats, ddof=1),
        'n_bootstrap': n_bootstrap,
        'method': 'BCa'
    }


def bootstrap_bca_diff_ci(group1, group2, statistic=np.mean, n_bootstrap=N_BOOTSTRAP,
                          confidence=0.95, random_state=RANDOM_SEED):
    """
    BCa Bootstrap CI for difference between two groups.

    Parameters
    ----------
    group1, group2 : array-like
    statistic : callable
    n_bootstrap : int
    confidence : float
    random_state : int

    Returns
    -------
    dict with BCa confidence interval for the difference
    """
    rng = np.random.RandomState(random_state)
    group1, group2 = np.asarray(group1), np.asarray(group2)
    n1, n2 = len(group1), len(group2)

    # Original estimate
    theta_hat = statistic(group1) - statistic(group2)

    # Bootstrap distribution
    boot_diffs = []
    for _ in range(n_bootstrap):
        boot1 = group1[rng.randint(0, n1, n1)]
        boot2 = group2[rng.randint(0, n2, n2)]
        boot_diffs.append(statistic(boot1) - statistic(boot2))
    boot_diffs = np.array(boot_diffs)

    alpha = 1 - confidence

    # Bias correction
    prop_less = np.mean(boot_diffs < theta_hat)
    prop_less = np.clip(prop_less, 1e-10, 1 - 1e-10)
    z0 = stats.norm.ppf(prop_less)

    # Acceleration via jackknife on combined influence
    # Jackknife for group1
    jack1 = np.array([
        statistic(np.delete(group1, i)) - statistic(group2)
        for i in range(n1)
    ])
    # Jackknife for group2
    jack2 = np.array([
        statistic(group1) - statistic(np.delete(group2, i))
        for i in range(n2)
    ])

    all_jack = np.concatenate([jack1, jack2])
    jack_mean = np.mean(all_jack)

    numerator = np.sum((jack_mean - all_jack) ** 3)
    denominator = 6 * (np.sum((jack_mean - all_jack) ** 2) ** 1.5)
    a = numerator / denominator if denominator != 0 else 0

    # BCa percentiles
    z_alpha_lower = stats.norm.ppf(alpha / 2)
    z_alpha_upper = stats.norm.ppf(1 - alpha / 2)

    def bca_percentile(z_alpha):
        num = z0 + z_alpha
        denom = 1 - a * num
        if denom == 0:
            return stats.norm.cdf(z0 + z_alpha)
        return stats.norm.cdf(z0 + num / denom)

    p_lower = np.clip(bca_percentile(z_alpha_lower), 0.001, 0.999)
    p_upper = np.clip(bca_percentile(z_alpha_upper), 0.001, 0.999)

    ci_lower = np.percentile(boot_diffs, 100 * p_lower)
    ci_upper = np.percentile(boot_diffs, 100 * p_upper)

    # Percentile CI for comparison
    ci_lower_pct = np.percentile(boot_diffs, 100 * alpha / 2)
    ci_upper_pct = np.percentile(boot_diffs, 100 * (1 - alpha / 2))

    return {
        'diff': theta_hat,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'ci_lower_percentile': ci_lower_pct,
        'ci_upper_percentile': ci_upper_pct,
        'bias_correction': z0,
        'acceleration': a,
        'se': np.std(boot_diffs, ddof=1),
        'n_bootstrap': n_bootstrap,
        'method': 'BCa'
    }


# =============================================================================
# PERMUTATION TESTS
# =============================================================================

def permutation_test_two_groups(group1, group2, statistic=lambda x, y: np.mean(x) - np.mean(y),
                                n_permutations=N_PERMUTATIONS, random_state=RANDOM_SEED):
    """
    Permutation test for difference between two groups.
    
    Parameters
    ----------
    group1, group2 : array-like
    statistic : callable(group1, group2) -> float
    n_permutations : int
    random_state : int
    
    Returns
    -------
    dict with observed, p_value, null_distribution
    """
    rng = np.random.RandomState(random_state)
    group1, group2 = np.asarray(group1), np.asarray(group2)
    
    observed = statistic(group1, group2)
    
    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    
    perm_stats = []
    for _ in range(n_permutations):
        perm = rng.permutation(combined)
        perm_stat = statistic(perm[:n1], perm[n1:])
        perm_stats.append(perm_stat)
    
    perm_stats = np.array(perm_stats)
    
    # Two-sided p-value
    p_value = np.mean(np.abs(perm_stats) >= np.abs(observed))
    
    return {
        'observed': observed,
        'p_value': p_value,
        'null_mean': np.mean(perm_stats),
        'null_std': np.std(perm_stats),
        'n_permutations': n_permutations
    }


def permutation_test_interaction(hh, ha, ah, aa, n_permutations=N_PERMUTATIONS,
                                  random_state=RANDOM_SEED):
    """
    Permutation test for 2×2 interaction effect.
    
    Tests: (AA - AH) - (HA - HH) ≠ 0
    
    Parameters
    ----------
    hh, ha, ah, aa : array-like
        Data for each cell (Human-Human, Human-AI, AI-Human, AI-AI)
    
    Returns
    -------
    dict with observed_interaction, p_value, null_distribution, ci
    """
    rng = np.random.RandomState(random_state)
    
    hh, ha = np.asarray(hh), np.asarray(ha)
    ah, aa = np.asarray(ah), np.asarray(aa)
    
    # Observed interaction
    observed = (np.mean(aa) - np.mean(ah)) - (np.mean(ha) - np.mean(hh))
    
    # Combine all data
    all_data = np.concatenate([hh, ha, ah, aa])
    n_hh, n_ha, n_ah, n_aa = len(hh), len(ha), len(ah), len(aa)
    
    # Permutation distribution
    perm_interactions = []
    for _ in range(n_permutations):
        p = rng.permutation(all_data)
        
        p_hh = p[:n_hh]
        p_ha = p[n_hh:n_hh+n_ha]
        p_ah = p[n_hh+n_ha:n_hh+n_ha+n_ah]
        p_aa = p[n_hh+n_ha+n_ah:]
        
        perm_int = (np.mean(p_aa) - np.mean(p_ah)) - (np.mean(p_ha) - np.mean(p_hh))
        perm_interactions.append(perm_int)
    
    perm_interactions = np.array(perm_interactions)
    
    # Two-sided p-value
    p_value = np.mean(np.abs(perm_interactions) >= np.abs(observed))

    # Null distribution bounds (values expected under H0)
    # Note: These are NOT confidence intervals for the observed effect
    null_lower = np.percentile(perm_interactions, 2.5)
    null_upper = np.percentile(perm_interactions, 97.5)

    # Bootstrap CI for observed interaction (proper confidence interval)
    boot_interactions = []
    for _ in range(min(n_permutations, 5000)):  # Use fewer iterations for speed
        boot_hh = hh[rng.randint(0, n_hh, n_hh)]
        boot_ha = ha[rng.randint(0, n_ha, n_ha)]
        boot_ah = ah[rng.randint(0, n_ah, n_ah)]
        boot_aa = aa[rng.randint(0, n_aa, n_aa)]
        boot_int = (np.mean(boot_aa) - np.mean(boot_ah)) - (np.mean(boot_ha) - np.mean(boot_hh))
        boot_interactions.append(boot_int)

    ci_lower = np.percentile(boot_interactions, 2.5)
    ci_upper = np.percentile(boot_interactions, 97.5)

    return {
        'observed_interaction': observed,
        'p_value': p_value,
        'null_mean': np.mean(perm_interactions),
        'null_std': np.std(perm_interactions),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'null_lower': null_lower,  # 2.5th percentile of null distribution
        'null_upper': null_upper,  # 97.5th percentile of null distribution
        'n_permutations': n_permutations,
        'null_distribution': perm_interactions
    }


# =============================================================================
# MULTIPLE COMPARISON CORRECTIONS
# =============================================================================

def bonferroni_correction(p_values, alpha=ALPHA):
    """
    Bonferroni correction for multiple comparisons.
    
    Returns
    -------
    dict with adjusted_alpha, significant (boolean array)
    """
    p_values = np.asarray(p_values)
    n_tests = len(p_values)
    adjusted_alpha = alpha / n_tests
    
    return {
        'adjusted_alpha': adjusted_alpha,
        'significant': p_values < adjusted_alpha,
        'n_tests': n_tests
    }


def holm_bonferroni_correction(p_values, alpha=ALPHA):
    """
    Holm-Bonferroni (step-down) correction for multiple comparisons.

    Less conservative than Bonferroni while still controlling FWER.
    Tests p-values sequentially from smallest to largest, adjusting
    the threshold at each step.

    Parameters
    ----------
    p_values : array-like
    alpha : float
        Family-wise error rate to control

    Returns
    -------
    dict with adjusted_p, significant, n_tests

    References
    ----------
    Holm, S. (1979). A simple sequentially rejective multiple test procedure.
    Scandinavian Journal of Statistics, 6(2), 65-70.
    """
    p_values = np.asarray(p_values)
    n = len(p_values)

    # Sort p-values and track original indices
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    # Holm thresholds: α/(n-i+1) for i = 1, 2, ..., n
    thresholds = alpha / (n - np.arange(n))

    # Find first non-significant p-value
    # All p-values before it are significant
    significant_sorted = np.zeros(n, dtype=bool)

    for i in range(n):
        if sorted_p[i] <= thresholds[i]:
            significant_sorted[i] = True
        else:
            # Stop: all remaining are non-significant
            break

    # Map back to original order
    significant = np.zeros(n, dtype=bool)
    significant[sorted_idx] = significant_sorted

    # Adjusted p-values (Holm-adjusted)
    # p_adj[i] = max(p[j] * (n-j+1) for j <= i)
    adjusted_p_sorted = np.zeros(n)
    running_max = 0
    for i in range(n):
        adjusted = sorted_p[i] * (n - i)
        running_max = max(running_max, adjusted)
        adjusted_p_sorted[i] = min(running_max, 1.0)

    adjusted_p = np.zeros(n)
    adjusted_p[sorted_idx] = adjusted_p_sorted

    return {
        'adjusted_p': adjusted_p,
        'significant': significant,
        'n_significant': significant.sum(),
        'n_tests': n,
        'method': 'Holm-Bonferroni'
    }


def hochberg_correction(p_values, alpha=ALPHA):
    """
    Hochberg (step-up) correction for multiple comparisons.

    More powerful than Holm-Bonferroni when tests are independent
    or positively dependent.

    Parameters
    ----------
    p_values : array-like
    alpha : float

    Returns
    -------
    dict with adjusted_p, significant, n_tests

    References
    ----------
    Hochberg, Y. (1988). A sharper Bonferroni procedure for multiple tests
    of significance. Biometrika, 75(4), 800-802.
    """
    p_values = np.asarray(p_values)
    n = len(p_values)

    # Sort p-values (descending for step-up)
    sorted_idx = np.argsort(p_values)[::-1]  # Largest first
    sorted_p = p_values[sorted_idx]

    # Hochberg thresholds: α/(n-i+1) for i = n, n-1, ..., 1
    thresholds = alpha / np.arange(1, n + 1)

    # Find first significant p-value (from largest)
    significant_sorted = np.zeros(n, dtype=bool)

    for i in range(n):
        if sorted_p[i] <= thresholds[i]:
            # This and all smaller p-values are significant
            significant_sorted[i:] = True
            break

    # Map back to original order
    significant = np.zeros(n, dtype=bool)
    significant[sorted_idx] = significant_sorted

    # Adjusted p-values
    adjusted_p_sorted = np.zeros(n)
    running_min = 1.0
    for i in range(n):
        adjusted = sorted_p[i] * (i + 1)
        running_min = min(running_min, adjusted)
        adjusted_p_sorted[i] = min(running_min, 1.0)

    # Reverse to match ascending order for output
    adjusted_p = np.zeros(n)
    adjusted_p[sorted_idx] = adjusted_p_sorted

    return {
        'adjusted_p': adjusted_p,
        'significant': significant,
        'n_significant': significant.sum(),
        'n_tests': n,
        'method': 'Hochberg'
    }


def fdr_correction(p_values, alpha=ALPHA, method='bh'):
    """
    False Discovery Rate correction (Benjamini-Hochberg).
    
    Parameters
    ----------
    p_values : array-like
    alpha : float
        Target FDR level
    method : str
        'bh' for Benjamini-Hochberg, 'by' for Benjamini-Yekutieli
    
    Returns
    -------
    dict with adjusted_p, significant, n_discoveries
    """
    p_values = np.asarray(p_values)
    n = len(p_values)
    
    # Sort p-values
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    
    # BH procedure
    if method == 'bh':
        thresholds = alpha * np.arange(1, n + 1) / n
    elif method == 'by':
        c_n = np.sum(1 / np.arange(1, n + 1))
        thresholds = alpha * np.arange(1, n + 1) / (n * c_n)
    
    # Find largest k where p[k] <= threshold[k]
    significant_sorted = sorted_p <= thresholds
    
    if significant_sorted.any():
        max_k = np.max(np.where(significant_sorted)[0])
        significant_sorted[:max_k + 1] = True
    
    # Map back to original order
    significant = np.zeros(n, dtype=bool)
    significant[sorted_idx] = significant_sorted
    
    # Adjusted p-values
    adjusted_p = np.minimum.accumulate((sorted_p * n / np.arange(1, n + 1))[::-1])[::-1]
    adjusted_p = np.minimum(adjusted_p, 1.0)
    adjusted_p_original = np.zeros(n)
    adjusted_p_original[sorted_idx] = adjusted_p
    
    return {
        'adjusted_p': adjusted_p_original,
        'significant': significant,
        'n_discoveries': significant.sum(),
        'method': method
    }


def pairwise_comparisons(groups, group_labels, baseline_idx=0, 
                         correction='fdr', alpha=ALPHA):
    """
    Pairwise comparisons against a baseline with multiple comparison correction.
    
    Parameters
    ----------
    groups : list of arrays
    group_labels : list of str
    baseline_idx : int
        Index of baseline group
    correction : str
        'bonferroni', 'fdr', or 'none'
    
    Returns
    -------
    DataFrame with comparison results
    """
    baseline = groups[baseline_idx]
    baseline_label = group_labels[baseline_idx]
    
    results = []
    p_values = []
    
    for i, (group, label) in enumerate(zip(groups, group_labels)):
        if i == baseline_idx:
            continue
        
        mw = mann_whitney_test(baseline, group)
        
        results.append({
            'comparison': f'{label} vs {baseline_label}',
            'mean_diff': mw['mean_diff'],
            'effect_size': mw['effect_size'],
            'p_value': mw['p_value'],
            'n_baseline': len(baseline),
            'n_comparison': len(group)
        })
        p_values.append(mw['p_value'])
    
    # Apply correction
    if correction == 'bonferroni':
        corrected = bonferroni_correction(p_values, alpha)
        for i, res in enumerate(results):
            res['significant'] = corrected['significant'][i]
            res['adjusted_alpha'] = corrected['adjusted_alpha']
    elif correction == 'fdr':
        corrected = fdr_correction(p_values, alpha)
        for i, res in enumerate(results):
            res['adjusted_p'] = corrected['adjusted_p'][i]
            res['significant'] = corrected['significant'][i]
    else:
        for res in results:
            res['significant'] = res['p_value'] < alpha
    
    return pd.DataFrame(results)


# =============================================================================
# REGRESSION WITH CLUSTERED STANDARD ERRORS
# =============================================================================

def ols_with_clustered_se(df, formula, cluster_col):
    """
    OLS regression with clustered standard errors.
    
    Parameters
    ----------
    df : DataFrame
    formula : str
        Patsy formula (e.g., 'rating ~ paper_AI * reviewer_AI')
    cluster_col : str
        Column to cluster on (e.g., 'submission_number')
    
    Returns
    -------
    dict with coefficients, se, p_values, ci, model
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        raise ImportError("statsmodels required for regression analysis")
    
    # Fit model with clustered SEs
    model = smf.ols(formula, data=df).fit(
        cov_type='cluster',
        cov_kwds={'groups': df[cluster_col]}
    )
    
    return {
        'params': model.params.to_dict(),
        'se': model.bse.to_dict(),
        'p_values': model.pvalues.to_dict(),
        'ci_lower': model.conf_int()[0].to_dict(),
        'ci_upper': model.conf_int()[1].to_dict(),
        'rsquared': model.rsquared,
        'rsquared_adj': model.rsquared_adj,
        'nobs': model.nobs,
        'model': model
    }


def ols_robust(df, formula, robust_type='HC3'):
    """
    OLS with heteroskedasticity-robust standard errors.
    
    Parameters
    ----------
    robust_type : str
        'HC0', 'HC1', 'HC2', 'HC3' (default: HC3, best for small samples)
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        raise ImportError("statsmodels required")
    
    model = smf.ols(formula, data=df).fit(cov_type=robust_type)
    
    return {
        'params': model.params.to_dict(),
        'se': model.bse.to_dict(),
        'p_values': model.pvalues.to_dict(),
        'rsquared': model.rsquared,
        'model': model
    }


# =============================================================================
# ORDINAL REGRESSION
# =============================================================================

def ordinal_regression(df, outcome, predictors, method='logit'):
    """
    Ordered logit/probit regression for ordinal outcomes like ratings.
    
    Parameters
    ----------
    df : DataFrame
    outcome : str
        Ordinal outcome column
    predictors : list of str
        Predictor column names
    method : str
        'logit' or 'probit'
    
    Returns
    -------
    dict with coefficients, se, p_values, odds_ratios
    """
    try:
        from statsmodels.miscmodels.ordinal_model import OrderedModel
    except ImportError:
        raise ImportError("statsmodels required for ordinal regression")
    
    clean_df = df.dropna(subset=[outcome] + predictors)
    
    if len(clean_df) < 100:
        warnings.warn(f"Small sample size ({len(clean_df)})")
    
    model = OrderedModel(
        clean_df[outcome].astype(int),
        clean_df[predictors],
        distr=method
    )
    
    result = model.fit(method='bfgs', disp=False)
    
    # Extract results
    params = {k: v for k, v in result.params.items() if k in predictors}
    se = {k: v for k, v in result.bse.items() if k in predictors}
    p_values = {k: v for k, v in result.pvalues.items() if k in predictors}
    
    # Odds ratios (for logit)
    odds_ratios = {k: np.exp(v) for k, v in params.items()} if method == 'logit' else None
    
    return {
        'params': params,
        'se': se,
        'p_values': p_values,
        'odds_ratios': odds_ratios,
        'pseudo_rsquared': result.prsquared,
        'nobs': result.nobs,
        'model': result
    }


# =============================================================================
# MIXED EFFECTS MODELS
# =============================================================================

def mixed_effects_model(df, formula, groups_col, random_effects='intercept'):
    """
    Linear mixed effects model for nested/clustered data.
    
    Parameters
    ----------
    df : DataFrame
    formula : str
        Fixed effects formula
    groups_col : str
        Grouping variable (e.g., 'submission_number')
    random_effects : str
        'intercept' for random intercepts, or formula for more complex RE
    
    Returns
    -------
    dict with fixed effects, random effects variance, model
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        raise ImportError("statsmodels required")
    
    if random_effects == 'intercept':
        model = smf.mixedlm(formula, df, groups=df[groups_col])
    else:
        model = smf.mixedlm(formula, df, groups=df[groups_col], 
                           re_formula=random_effects)
    
    result = model.fit(reml=True)

    # Handle n_groups - may not be available in all statsmodels versions
    try:
        n_groups = result.n_groups
    except AttributeError:
        # Calculate from data
        n_groups = df[groups_col].nunique()

    # Handle random effects variance safely
    try:
        if hasattr(result.cov_re, 'iloc'):
            re_variance = result.cov_re.iloc[0, 0]
        elif hasattr(result.cov_re, 'values'):
            re_variance = float(result.cov_re.values.flatten()[0])
        else:
            re_variance = float(result.cov_re)
    except:
        re_variance = np.nan

    return {
        'fe_params': result.fe_params.to_dict(),
        'fe_se': result.bse_fe.to_dict(),
        'fe_pvalues': result.pvalues.to_dict(),
        're_variance': re_variance,
        'residual_variance': result.scale,
        'icc': None,  # Compute separately if needed
        'nobs': result.nobs,
        'n_groups': n_groups,
        'model': result
    }


def compute_icc(df, outcome_col, group_col):
    """
    Compute Intraclass Correlation Coefficient.
    
    ICC = variance_between / (variance_between + variance_within)
    
    High ICC (>0.1) suggests clustering should be addressed.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        # Fallback: simple variance decomposition
        overall_var = df[outcome_col].var()
        group_means = df.groupby(group_col)[outcome_col].mean()
        between_var = group_means.var()
        return between_var / overall_var if overall_var > 0 else 0
    
    # Fit null mixed model
    model = smf.mixedlm(f'{outcome_col} ~ 1', df, groups=df[group_col])
    result = model.fit(reml=True)
    
    re_var = result.cov_re.iloc[0, 0] if hasattr(result.cov_re, 'iloc') else float(result.cov_re)
    resid_var = result.scale
    
    icc = re_var / (re_var + resid_var)
    
    return icc


# =============================================================================
# CONFIDENCE-WEIGHTED STATISTICS
# =============================================================================

def weighted_mean(values, weights):
    """
    Compute confidence-weighted mean.
    """
    values, weights = np.asarray(values), np.asarray(weights)
    mask = ~(np.isnan(values) | np.isnan(weights))
    
    if mask.sum() == 0:
        return np.nan
    
    return np.average(values[mask], weights=weights[mask])


def weighted_std(values, weights):
    """
    Compute confidence-weighted standard deviation.
    """
    values, weights = np.asarray(values), np.asarray(weights)
    mask = ~(np.isnan(values) | np.isnan(weights))
    
    if mask.sum() < 2:
        return np.nan
    
    v, w = values[mask], weights[mask]
    avg = np.average(v, weights=w)
    variance = np.average((v - avg) ** 2, weights=w)
    
    return np.sqrt(variance)


def paper_weighted_rating(reviews_df, paper_col='submission_number',
                          rating_col='rating', confidence_col='confidence'):
    """
    Compute confidence-weighted average rating per paper.
    
    Returns
    -------
    DataFrame with submission_number, raw_rating, weighted_rating
    """
    def compute_weighted(group):
        raw = group[rating_col].mean()
        
        if confidence_col in group.columns and group[confidence_col].notna().any():
            weighted = weighted_mean(group[rating_col], group[confidence_col])
        else:
            weighted = raw
        
        return pd.Series({
            'raw_rating': raw,
            'weighted_rating': weighted,
            'n_reviews': len(group),
            'mean_confidence': group[confidence_col].mean() if confidence_col in group.columns else np.nan
        })
    
    return reviews_df.groupby(paper_col).apply(compute_weighted).reset_index()


def confident_leniency_index(df, rating_col='rating', confidence_col='confidence',
                             confidence_threshold=None):
    """
    Compute Confident Leniency Index (CLI).
    
    CLI = mean_rating(high_confidence) - mean_rating(low_confidence)
    
    Positive CLI means high-confidence reviewers give higher ratings → problematic
    if combined with systematic bias.
    
    Parameters
    ----------
    df : DataFrame
    confidence_threshold : float, optional
        Split point for high/low confidence (default: median)
    
    Returns
    -------
    dict with cli, p_value, high_conf_mean, low_conf_mean
    """
    clean = df.dropna(subset=[rating_col, confidence_col])
    
    if len(clean) < 20:
        return {'cli': np.nan, 'p_value': np.nan, 'n': len(clean)}
    
    if confidence_threshold is None:
        confidence_threshold = clean[confidence_col].median()
    
    high_conf = clean[clean[confidence_col] > confidence_threshold][rating_col]
    low_conf = clean[clean[confidence_col] <= confidence_threshold][rating_col]
    
    if len(high_conf) < 10 or len(low_conf) < 10:
        return {'cli': np.nan, 'p_value': np.nan, 'n': len(clean)}
    
    cli = high_conf.mean() - low_conf.mean()
    
    # Test significance
    u_stat, p_value = mannwhitneyu(high_conf, low_conf, alternative='two-sided')
    
    return {
        'cli': cli,
        'p_value': p_value,
        'high_conf_mean': high_conf.mean(),
        'low_conf_mean': low_conf.mean(),
        'n_high': len(high_conf),
        'n_low': len(low_conf),
        'threshold': confidence_threshold
    }


# =============================================================================
# COMPREHENSIVE COMPARISON FUNCTION
# =============================================================================

def comprehensive_comparison(group1, group2, label1='Group1', label2='Group2',
                             cluster_ids1=None, cluster_ids2=None):
    """
    Comprehensive statistical comparison with multiple methods.
    
    Returns all relevant statistics for robustness.
    
    Parameters
    ----------
    group1, group2 : array-like
    label1, label2 : str
    cluster_ids1, cluster_ids2 : array-like, optional
        Cluster IDs for clustered inference
    
    Returns
    -------
    dict with all test results
    """
    g1, g2 = np.asarray(group1), np.asarray(group2)
    
    results = {
        'labels': (label1, label2),
        'n1': len(g1),
        'n2': len(g2),
        'mean1': np.mean(g1),
        'mean2': np.mean(g2),
        'std1': np.std(g1, ddof=1),
        'std2': np.std(g2, ddof=1),
        'mean_diff': np.mean(g1) - np.mean(g2),
    }
    
    # Effect sizes
    results['cohens_d'] = cohens_d(g1, g2)
    results['hedges_g'] = hedges_g(g1, g2)
    results['cliffs_delta'] = cliffs_delta(g1, g2)
    
    # Parametric tests
    t_stat, t_pval = ttest_ind(g1, g2)
    results['t_statistic'] = t_stat
    results['t_pvalue'] = t_pval
    
    # Non-parametric tests
    mw = mann_whitney_test(g1, g2)
    results['mann_whitney_u'] = mw['u_stat']
    results['mann_whitney_p'] = mw['p_value']
    results['rank_biserial'] = mw['effect_size']
    
    # Bootstrap
    boot = bootstrap_diff_ci(g1, g2)
    results['bootstrap_ci_lower'] = boot['ci_lower']
    results['bootstrap_ci_upper'] = boot['ci_upper']
    results['bootstrap_p'] = boot['p_value']
    
    # Permutation test
    perm = permutation_test_two_groups(g1, g2)
    results['permutation_p'] = perm['p_value']
    
    # Variance comparison
    lev = levene_test(g1, g2)
    results['levene_p'] = lev['p_value']
    results['var_ratio'] = lev['var_ratio']
    
    return results


def format_results_table(results_dict, precision=4):
    """
    Format comprehensive comparison results as readable table.
    """
    rows = []
    
    rows.append(('Sample Sizes', f"n1={results_dict['n1']}, n2={results_dict['n2']}"))
    rows.append(('Means', f"{results_dict['mean1']:.{precision}f} vs {results_dict['mean2']:.{precision}f}"))
    rows.append(('Mean Difference', f"{results_dict['mean_diff']:+.{precision}f}"))
    rows.append(('', ''))
    rows.append(('EFFECT SIZES', ''))
    rows.append(("Cohen's d", f"{results_dict['cohens_d']:.{precision}f}"))
    rows.append(("Hedges' g", f"{results_dict['hedges_g']:.{precision}f}"))
    rows.append(("Cliff's Delta", f"{results_dict['cliffs_delta']:.{precision}f}"))
    rows.append(('', ''))
    rows.append(('P-VALUES', ''))
    rows.append(('t-test', f"{results_dict['t_pvalue']:.{precision}e}"))
    rows.append(('Mann-Whitney', f"{results_dict['mann_whitney_p']:.{precision}e}"))
    rows.append(('Permutation', f"{results_dict['permutation_p']:.{precision}f}"))
    rows.append(('Bootstrap', f"{results_dict['bootstrap_p']:.{precision}f}"))
    rows.append(('', ''))
    rows.append(('95% BOOTSTRAP CI', f"[{results_dict['bootstrap_ci_lower']:.{precision}f}, {results_dict['bootstrap_ci_upper']:.{precision}f}]"))
    
    return pd.DataFrame(rows, columns=['Metric', 'Value'])
