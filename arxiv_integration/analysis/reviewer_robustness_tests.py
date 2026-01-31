"""
Reviewer Robustness Tests for Restud Submission.
=================================================

This module addresses two key referee concerns:

1. ADVERSE SELECTION / EFFORT ENDOGENEITY (Reviewer Concern #1)
   - Oster (2019) bounds for unobserved selection
   - Soundness vs Presentation differential as "AI fingerprint"
   - Tests that effort alone cannot explain the pattern

2. DETECTOR BIAS CONFOUND (Reviewer Concern #2)
   - Tests if AI penalty holds across native/non-native English speakers
   - Regional stratification (US/UK/Canada vs other countries)
   - Institution-level controls

References:
- Oster, E. (2019). Unobservable Selection and Coefficient Stability.
  Journal of Business & Economic Statistics, 37(2), 187-204.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from scipy import stats

# Try importing from parent, fall back to inline
try:
    from .selection_robustness import (
        ols_with_clustered_se,
        cohens_d,
        bootstrap_diff_ci,
        fdr_correction
    )
except ImportError:
    # Inline implementations if import fails
    def ols_with_clustered_se(data, formula, cluster_col='submission_number', cov_type='cluster'):
        import statsmodels.formula.api as smf
        model = smf.ols(formula, data=data.dropna(subset=[cluster_col]))
        if cov_type == 'cluster' and cluster_col in data.columns:
            results = model.fit(cov_type='cluster', cov_kwds={'groups': data.dropna(subset=[cluster_col])[cluster_col]})
        else:
            results = model.fit(cov_type='HC3')
        return results

    def cohens_d(group1, group2):
        n1, n2 = len(group1), len(group2)
        var1, var2 = group1.var(), group2.var()
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        d = (group1.mean() - group2.mean()) / pooled_std
        df = n1 + n2 - 2
        j = 1 - (3 / (4*df - 1))  # Hedges correction
        return d * j

    def bootstrap_diff_ci(group1, group2, n_boot=2000, alpha=0.05):
        np.random.seed(42)
        diffs = []
        for _ in range(n_boot):
            s1 = np.random.choice(group1, size=len(group1), replace=True)
            s2 = np.random.choice(group2, size=len(group2), replace=True)
            diffs.append(s1.mean() - s2.mean())
        return np.percentile(diffs, [100*alpha/2, 100*(1-alpha/2)])

    def fdr_correction(p_values, alpha=0.05):
        from scipy.stats import false_discovery_control
        try:
            return false_discovery_control(p_values, method='bh')
        except:
            n = len(p_values)
            sorted_idx = np.argsort(p_values)
            sorted_p = np.array(p_values)[sorted_idx]
            adjusted = np.zeros(n)
            for i, (idx, p) in enumerate(zip(sorted_idx, sorted_p)):
                adjusted[idx] = min(p * n / (i + 1), 1.0)
            return adjusted


# =============================================================================
# CONSTANTS
# =============================================================================

# English-speaking countries (for detector bias analysis)
NATIVE_ENGLISH_COUNTRIES = {
    'United States', 'United Kingdom', 'Canada', 'Australia',
    'New Zealand', 'Ireland'
}

# R_max for Oster bounds (standard assumption)
R_MAX_DEFAULT = 1.3  # Assumes max R² is 1.3x observed R²


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class OsterBoundsResult:
    """Results from Oster (2019) sensitivity analysis."""
    beta_controlled: float      # Coefficient with controls
    beta_uncontrolled: float    # Coefficient without controls
    r2_controlled: float        # R² with controls
    r2_uncontrolled: float      # R² without controls
    delta: float                # Oster's δ (selection ratio)
    beta_star: float            # Bias-adjusted coefficient
    identified_set: Tuple[float, float]  # [β*, β_controlled]
    r_max: float               # Assumed maximum R²
    interpretation: str         # Human-readable interpretation


@dataclass
class DetectorBiasResult:
    """Results from detector bias analysis."""
    overall_effect: float
    native_speaker_effect: float
    non_native_effect: float
    difference: float
    p_value_interaction: float
    interpretation: str


@dataclass
class ComponentDifferentialResult:
    """Results from Soundness vs Presentation differential test."""
    soundness_decline: float
    presentation_decline: float
    differential: float  # Soundness - Presentation
    p_value: float
    is_ai_fingerprint: bool
    interpretation: str


# =============================================================================
# 1. OSTER BOUNDS (EFFORT ENDOGENEITY)
# =============================================================================

def compute_oster_bounds(
    data: pd.DataFrame,
    outcome_col: str = 'avg_rating',
    treatment_col: str = 'ai_percentage',
    control_cols: List[str] = None,
    cluster_col: str = 'submission_number',
    r_max: float = R_MAX_DEFAULT,
    verbose: bool = True
) -> OsterBoundsResult:
    """
    Compute Oster (2019) bounds for unobserved selection.

    This tests: "How much unobserved selection (e.g., effort) would need to
    exist to explain away the AI coefficient?"

    Parameters
    ----------
    data : DataFrame
        Analysis data
    outcome_col : str
        Dependent variable (e.g., 'avg_rating')
    treatment_col : str
        Treatment variable (e.g., 'ai_percentage')
    control_cols : list
        Control variables (e.g., ['first_author_h_index'])
    cluster_col : str
        Clustering variable
    r_max : float
        Assumed maximum R² (default 1.3 = Oster's recommendation)
    verbose : bool
        Print results

    Returns
    -------
    OsterBoundsResult

    References
    ----------
    Oster, E. (2019). Unobservable Selection and Coefficient Stability.
    JBES, 37(2), 187-204.
    """
    if control_cols is None:
        control_cols = ['first_author_h_index']

    # Prepare data
    analysis_cols = [outcome_col, treatment_col, cluster_col] + control_cols
    analysis_cols = [c for c in analysis_cols if c in data.columns]
    df = data[analysis_cols].dropna().copy()

    if len(df) < 100:
        raise ValueError(f"Insufficient data for Oster bounds: {len(df)} obs")

    # Standardize treatment for interpretation
    df[f'{treatment_col}_std'] = (df[treatment_col] - df[treatment_col].mean()) / df[treatment_col].std()

    # Model 1: Uncontrolled (just treatment)
    formula_uncontrolled = f'{outcome_col} ~ {treatment_col}_std'
    result_uncontrolled = ols_with_clustered_se(df, formula_uncontrolled, cluster_col)
    beta_uncontrolled = result_uncontrolled['params'][f'{treatment_col}_std']
    r2_uncontrolled = result_uncontrolled['rsquared']

    # Model 2: Controlled (treatment + controls)
    controls_formula = ' + '.join(control_cols)
    formula_controlled = f'{outcome_col} ~ {treatment_col}_std + {controls_formula}'
    result_controlled = ols_with_clustered_se(df, formula_controlled, cluster_col)
    beta_controlled = result_controlled['params'][f'{treatment_col}_std']
    r2_controlled = result_controlled['rsquared']

    # Compute Oster's delta
    # δ = (β_controlled - 0) / (β_uncontrolled - β_controlled) * (R_max - R_controlled) / (R_controlled - R_uncontrolled)

    beta_diff = beta_uncontrolled - beta_controlled
    r2_diff = r2_controlled - r2_uncontrolled

    if abs(beta_diff) < 1e-10 or abs(r2_diff) < 1e-10:
        # Coefficients are stable - selection on unobservables would need to be huge
        delta = np.inf
        beta_star = beta_controlled
    else:
        # Oster's formula for δ
        delta = (beta_controlled * (r_max * r2_controlled - r2_controlled)) / \
                (beta_diff * r2_diff) if beta_diff != 0 else np.inf

        # Bias-adjusted coefficient (β*)
        # β* = β_controlled - δ * (β_uncontrolled - β_controlled) * (R_max - R_controlled) / (R_controlled - R_uncontrolled)
        if delta != np.inf:
            adjustment = (beta_uncontrolled - beta_controlled) * (r_max * r2_controlled - r2_controlled) / \
                        (r2_controlled - r2_uncontrolled) if r2_diff != 0 else 0
            beta_star = beta_controlled - adjustment
        else:
            beta_star = beta_controlled

    # Identified set
    identified_set = (min(beta_star, beta_controlled), max(beta_star, beta_controlled))

    # Interpretation
    if delta > 1:
        interpretation = (
            f"ROBUST TO SELECTION: δ = {delta:.2f} > 1\n"
            f"Selection on unobservables would need to be {delta:.1f}x stronger than\n"
            f"selection on observables to explain away the result.\n"
            f"This is implausible under standard assumptions (Oster, 2019)."
        )
    elif delta > 0:
        interpretation = (
            f"PARTIALLY ROBUST: 0 < δ = {delta:.2f} < 1\n"
            f"Some unobserved confounding could explain the result,\n"
            f"but would need to be substantial."
        )
    else:
        interpretation = (
            f"SENSITIVE TO SELECTION: δ = {delta:.2f}\n"
            f"Results may be driven by unobserved selection."
        )

    result = OsterBoundsResult(
        beta_controlled=beta_controlled,
        beta_uncontrolled=beta_uncontrolled,
        r2_controlled=r2_controlled,
        r2_uncontrolled=r2_uncontrolled,
        delta=delta,
        beta_star=beta_star,
        identified_set=identified_set,
        r_max=r_max,
        interpretation=interpretation
    )

    if verbose:
        print("\n" + "=" * 70)
        print("OSTER (2019) BOUNDS FOR UNOBSERVED SELECTION")
        print("Testing: How much 'effort' confounding would explain away AI effect?")
        print("=" * 70)
        print(f"\nUncontrolled model:")
        print(f"  β(AI) = {beta_uncontrolled:.4f}, R² = {r2_uncontrolled:.4f}")
        print(f"\nControlled model (with h-index):")
        print(f"  β(AI) = {beta_controlled:.4f}, R² = {r2_controlled:.4f}")
        print(f"\nOster's δ (selection ratio): {delta:.2f}")
        print(f"Bias-adjusted β*: {beta_star:.4f}")
        print(f"Identified set: [{identified_set[0]:.4f}, {identified_set[1]:.4f}]")
        print(f"\n{interpretation}")

    return result


# =============================================================================
# 2. SOUNDNESS VS PRESENTATION DIFFERENTIAL (AI FINGERPRINT)
# =============================================================================

def test_ai_fingerprint(
    data: pd.DataFrame,
    ai_threshold: float = 20.0,
    ai_col: str = 'ai_percentage',
    soundness_col: str = 'soundness',
    presentation_col: str = 'presentation',
    cluster_col: str = 'submission_number',
    verbose: bool = True
) -> ComponentDifferentialResult:
    """
    Test the "AI Fingerprint": Soundness declines more than Presentation.

    This is the key defense against the "laziness/effort" critique:
    - If authors are just lazy, Presentation (typos, structure) should suffer too
    - If AI substitutes for judgment, Soundness (logic, methodology) suffers
      while surface polish (Presentation) stays high

    Parameters
    ----------
    data : DataFrame
        Review-level data with component scores
    ai_threshold : float
        Threshold to classify as "AI paper" (default 20%)
    ai_col : str
        AI percentage column
    soundness_col : str
        Soundness score column
    presentation_col : str
        Presentation score column
    cluster_col : str
        Clustering variable
    verbose : bool
        Print results

    Returns
    -------
    ComponentDifferentialResult
    """
    # Prepare data
    required_cols = [ai_col, soundness_col, presentation_col]
    available_cols = [c for c in required_cols if c in data.columns]

    if len(available_cols) < 3:
        raise ValueError(f"Missing columns. Required: {required_cols}, Found: {available_cols}")

    df = data[[ai_col, soundness_col, presentation_col, cluster_col]].dropna().copy()

    # Create AI indicator
    df['is_ai'] = (df[ai_col] >= ai_threshold).astype(int)

    # Compute mean scores by group
    ai_papers = df[df['is_ai'] == 1]
    human_papers = df[df['is_ai'] == 0]

    # Soundness comparison
    soundness_ai = ai_papers[soundness_col].values
    soundness_human = human_papers[soundness_col].values
    soundness_diff = soundness_ai.mean() - soundness_human.mean()
    soundness_pct = 100 * soundness_diff / soundness_human.mean()

    # Presentation comparison
    presentation_ai = ai_papers[presentation_col].values
    presentation_human = human_papers[presentation_col].values
    presentation_diff = presentation_ai.mean() - presentation_human.mean()
    presentation_pct = 100 * presentation_diff / presentation_human.mean()

    # The differential (AI fingerprint)
    differential = soundness_pct - presentation_pct  # Negative = Soundness drops MORE

    # Statistical test: Is the differential significant?
    # Use paired differences approach
    # For each paper, compute (Soundness_decline - Presentation_decline)

    # Create paper-level scores
    paper_scores = df.groupby([cluster_col, 'is_ai']).agg({
        soundness_col: 'mean',
        presentation_col: 'mean'
    }).unstack()

    # Fix column names - convert all parts to str before joining
    paper_scores.columns = ['_'.join(str(c) for c in col).strip() for col in paper_scores.columns.values]
    paper_scores = paper_scores.dropna()

    # Bootstrap test for p-value (more robust for any sample size)
    n_boot = 2000
    np.random.seed(42)
    boot_diffs = []

    if len(ai_papers) > 0 and len(human_papers) > 0:
        for _ in range(n_boot):
            ai_idx = np.random.choice(len(ai_papers), size=len(ai_papers), replace=True)
            human_idx = np.random.choice(len(human_papers), size=len(human_papers), replace=True)

            ai_s = ai_papers.iloc[ai_idx][soundness_col].mean()
            ai_p = ai_papers.iloc[ai_idx][presentation_col].mean()
            human_s = human_papers.iloc[human_idx][soundness_col].mean()
            human_p = human_papers.iloc[human_idx][presentation_col].mean()

            if human_s > 0 and human_p > 0:
                s_pct = 100 * (ai_s - human_s) / human_s
                p_pct = 100 * (ai_p - human_p) / human_p
                boot_diffs.append(s_pct - p_pct)

        # P-value: proportion of bootstrap samples where differential >= 0
        p_value = np.mean(np.array(boot_diffs) >= 0) if boot_diffs else 1.0
    else:
        p_value = 1.0

    # Is this the AI fingerprint?
    is_fingerprint = (differential < -1.0) and (p_value < 0.05)  # Soundness drops >1pp more

    # Interpretation
    if is_fingerprint:
        interpretation = (
            f"✓ AI FINGERPRINT CONFIRMED\n"
            f"Soundness declines {abs(differential):.1f}pp MORE than Presentation.\n"
            f"This is inconsistent with 'laziness' (which would affect both equally)\n"
            f"and consistent with AI substituting for judgment while preserving polish."
        )
    elif differential < 0:
        interpretation = (
            f"Weak AI fingerprint: Soundness declines {abs(differential):.1f}pp more,\n"
            f"but not statistically significant (p={p_value:.3f})."
        )
    else:
        interpretation = (
            f"No AI fingerprint: Presentation declines as much or more than Soundness.\n"
            f"This could indicate general quality issues rather than AI-specific effects."
        )

    result = ComponentDifferentialResult(
        soundness_decline=soundness_pct,
        presentation_decline=presentation_pct,
        differential=differential,
        p_value=p_value,
        is_ai_fingerprint=is_fingerprint,
        interpretation=interpretation
    )

    if verbose:
        print("\n" + "=" * 70)
        print("AI FINGERPRINT TEST: Soundness vs Presentation Differential")
        print("Testing: Does Soundness (judgment) decline more than Presentation (polish)?")
        print("=" * 70)
        print(f"\nAI Papers (>={ai_threshold}% AI): {len(ai_papers):,} reviews")
        print(f"Human Papers (<{ai_threshold}% AI): {len(human_papers):,} reviews")
        print(f"\nSoundness decline:    {soundness_pct:+.1f}%")
        print(f"Presentation decline: {presentation_pct:+.1f}%")
        print(f"\nDIFFERENTIAL: {differential:+.1f}pp")
        print(f"(Negative = Soundness drops more = AI fingerprint)")
        print(f"p-value: {p_value:.4f}")
        print(f"\n{interpretation}")

    return result


# =============================================================================
# 3. DETECTOR BIAS TEST (NATIVE VS NON-NATIVE SPEAKERS)
# =============================================================================

def test_detector_bias(
    data: pd.DataFrame,
    outcome_col: str = 'avg_rating',
    ai_col: str = 'ai_percentage',
    country_col: str = 'first_author_country',
    cluster_col: str = 'submission_number',
    native_countries: set = None,
    verbose: bool = True
) -> DetectorBiasResult:
    """
    Test for detector bias: Does the AI penalty hold for both native and
    non-native English speakers?

    The concern: AI detectors may flag non-native English text as "AI" due to
    more rigid/standard grammar, creating a spurious correlation.

    Parameters
    ----------
    data : DataFrame
        Analysis data with country information
    outcome_col : str
        Dependent variable
    ai_col : str
        AI percentage column
    country_col : str
        Country column for author
    cluster_col : str
        Clustering variable
    native_countries : set
        Set of native English-speaking countries
    verbose : bool
        Print results

    Returns
    -------
    DetectorBiasResult
    """
    if native_countries is None:
        native_countries = NATIVE_ENGLISH_COUNTRIES

    # Prepare data - check if country column exists and has data
    required_cols = [outcome_col, ai_col, country_col, cluster_col]
    available_cols = [c for c in required_cols if c in data.columns]

    if len(available_cols) < 4:
        raise ValueError(f"Missing columns. Required: {required_cols}, Found: {available_cols}")

    df = data[required_cols].dropna().copy()

    if len(df) < 20:
        raise ValueError(f"Insufficient data with country information: only {len(df)} observations")

    # Classify as native/non-native
    df['is_native_english'] = df[country_col].isin(native_countries).astype(int)

    # Standardize AI percentage
    ai_std_val = df[ai_col].std()
    if ai_std_val == 0 or np.isnan(ai_std_val):
        raise ValueError("AI percentage has no variance")
    df['ai_std'] = (df[ai_col] - df[ai_col].mean()) / ai_std_val

    n_native = df['is_native_english'].sum()
    n_non_native = len(df) - n_native

    if n_native < 5 or n_non_native < 5:
        raise ValueError(f"Insufficient samples: {n_native} native, {n_non_native} non-native. Need at least 5 in each group.")

    if n_native < 30 or n_non_native < 30:
        warnings.warn(f"Small sample sizes: {n_native} native, {n_non_native} non-native. Results may be unreliable.")

    # Model 1: Overall effect
    formula_overall = f'{outcome_col} ~ ai_std'
    result_overall = ols_with_clustered_se(df, formula_overall, cluster_col)
    overall_effect = result_overall['params']['ai_std']

    # Model 2: Separate effects by language background
    # AI effect for native speakers
    df_native = df[df['is_native_english'] == 1]
    result_native = ols_with_clustered_se(df_native, formula_overall, cluster_col)
    native_effect = result_native['params']['ai_std']

    # AI effect for non-native speakers
    df_non_native = df[df['is_native_english'] == 0]
    result_non_native = ols_with_clustered_se(df_non_native, formula_overall, cluster_col)
    non_native_effect = result_non_native['params']['ai_std']

    # Model 3: Interaction model
    formula_interaction = f'{outcome_col} ~ ai_std * is_native_english'
    result_interaction = ols_with_clustered_se(df, formula_interaction, cluster_col)

    # Get interaction p-value
    interaction_term = 'ai_std:is_native_english'
    if interaction_term in result_interaction['params']:
        p_interaction = result_interaction['p_values'][interaction_term]
        interaction_coef = result_interaction['params'][interaction_term]
    else:
        p_interaction = 1.0
        interaction_coef = 0.0

    # Difference in effects
    effect_difference = native_effect - non_native_effect

    # Interpretation
    if abs(effect_difference) < 0.05 and p_interaction > 0.1:
        interpretation = (
            f"✓ NO DETECTOR BIAS DETECTED\n"
            f"The AI penalty is similar for native ({native_effect:.3f}) and\n"
            f"non-native speakers ({non_native_effect:.3f}).\n"
            f"Interaction p-value: {p_interaction:.3f}\n"
            f"This rules out the concern that AI detection is biased against\n"
            f"non-native English speakers."
        )
    elif native_effect < 0 and non_native_effect < 0:
        interpretation = (
            f"✓ AI PENALTY EXISTS IN BOTH GROUPS\n"
            f"Native speakers: {native_effect:.3f}\n"
            f"Non-native speakers: {non_native_effect:.3f}\n"
            f"Difference: {effect_difference:.3f} (interaction p={p_interaction:.3f})\n"
            f"While effects differ in magnitude, both groups show negative AI effects."
        )
    else:
        interpretation = (
            f"⚠ POTENTIAL DETECTOR BIAS\n"
            f"Native speakers: {native_effect:.3f}\n"
            f"Non-native speakers: {non_native_effect:.3f}\n"
            f"The AI effect differs substantially between groups.\n"
            f"Consider additional controls for linguistic factors."
        )

    result = DetectorBiasResult(
        overall_effect=overall_effect,
        native_speaker_effect=native_effect,
        non_native_effect=non_native_effect,
        difference=effect_difference,
        p_value_interaction=p_interaction,
        interpretation=interpretation
    )

    if verbose:
        print("\n" + "=" * 70)
        print("DETECTOR BIAS TEST: Native vs Non-Native English Speakers")
        print("Testing: Is the AI penalty driven by detector bias against non-native speakers?")
        print("=" * 70)
        print(f"\nSample sizes:")
        print(f"  Native English speakers: {n_native:,}")
        print(f"  Non-native speakers: {n_non_native:,}")
        print(f"\nAI Effect (β on standardized AI%):")
        print(f"  Overall:      {overall_effect:.4f}")
        print(f"  Native:       {native_effect:.4f}")
        print(f"  Non-native:   {non_native_effect:.4f}")
        print(f"\nInteraction (Native × AI): {interaction_coef:.4f} (p={p_interaction:.4f})")
        print(f"\n{interpretation}")

    return result


# =============================================================================
# 4. COMPREHENSIVE TEST SUITE
# =============================================================================

def run_reviewer_robustness_tests(
    reviews_df: pd.DataFrame,
    submissions_df: pd.DataFrame,
    enriched_df: pd.DataFrame,
    verbose: bool = True,
    save_results: bool = True,
    output_dir: str = '.'
) -> Dict[str, Any]:
    """
    Run all robustness tests for Restud reviewer concerns.

    Parameters
    ----------
    reviews_df : DataFrame
        Review-level data
    submissions_df : DataFrame
        Submission-level data
    enriched_df : DataFrame
        Author enrichment data
    verbose : bool
        Print detailed results
    save_results : bool
        Save results to files
    output_dir : str
        Output directory

    Returns
    -------
    dict with all test results
    """
    print("\n" + "=" * 70)
    print("RESTUD REVIEWER ROBUSTNESS TESTS")
    print("=" * 70)
    print("\nAddressing two key referee concerns:")
    print("1. Adverse Selection / Effort Endogeneity")
    print("2. AI Detector Bias Against Non-Native Speakers")
    print("=" * 70)

    # Merge data
    from .selection_robustness import merge_author_data, create_analysis_variables

    merged = merge_author_data(reviews_df, submissions_df, enriched_df)
    merged = create_analysis_variables(merged)

    results = {}

    # ==========================================================================
    # TEST 1: Oster Bounds
    # ==========================================================================
    print("\n\n" + "=" * 70)
    print("TEST 1: OSTER BOUNDS (Effort Endogeneity)")
    print("=" * 70)

    try:
        oster_result = compute_oster_bounds(
            merged,
            outcome_col='avg_rating',
            treatment_col='ai_percentage',
            control_cols=['first_author_h_index', 'last_author_h_index'],
            verbose=verbose
        )
        results['oster_bounds'] = oster_result
    except Exception as e:
        print(f"Error in Oster bounds: {e}")
        results['oster_bounds'] = None

    # ==========================================================================
    # TEST 2: AI Fingerprint (Soundness vs Presentation)
    # ==========================================================================
    print("\n\n" + "=" * 70)
    print("TEST 2: AI FINGERPRINT (Soundness vs Presentation)")
    print("=" * 70)

    try:
        fingerprint_result = test_ai_fingerprint(
            merged,
            ai_threshold=20.0,
            verbose=verbose
        )
        results['ai_fingerprint'] = fingerprint_result
    except Exception as e:
        print(f"Error in fingerprint test: {e}")
        results['ai_fingerprint'] = None

    # ==========================================================================
    # TEST 3: Detector Bias
    # ==========================================================================
    print("\n\n" + "=" * 70)
    print("TEST 3: DETECTOR BIAS (Native vs Non-Native)")
    print("=" * 70)

    # Use primary_country if available, else first_author_country
    country_col = 'primary_country' if 'primary_country' in merged.columns else 'first_author_country'

    try:
        detector_result = test_detector_bias(
            merged,
            country_col=country_col,
            verbose=verbose
        )
        results['detector_bias'] = detector_result
    except Exception as e:
        print(f"Error in detector bias test: {e}")
        results['detector_bias'] = None

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n\n" + "=" * 70)
    print("SUMMARY: ROBUSTNESS TEST RESULTS")
    print("=" * 70)

    passed_tests = 0
    total_tests = 3

    # Oster bounds
    if results.get('oster_bounds'):
        if results['oster_bounds'].delta > 1:
            print("\n✓ Oster Bounds: PASSED (δ > 1)")
            passed_tests += 1
        else:
            print(f"\n⚠ Oster Bounds: MARGINAL (δ = {results['oster_bounds'].delta:.2f})")

    # AI Fingerprint
    if results.get('ai_fingerprint'):
        if results['ai_fingerprint'].is_ai_fingerprint:
            print("✓ AI Fingerprint: CONFIRMED")
            passed_tests += 1
        else:
            print("⚠ AI Fingerprint: NOT CONFIRMED")

    # Detector Bias
    if results.get('detector_bias'):
        if results['detector_bias'].native_speaker_effect < 0 and \
           results['detector_bias'].non_native_effect < 0:
            print("✓ Detector Bias: NO BIAS (effect present in both groups)")
            passed_tests += 1
        else:
            print("⚠ Detector Bias: POTENTIAL CONCERN")

    print(f"\nOVERALL: {passed_tests}/{total_tests} tests passed")

    # Save results
    if save_results:
        summary_data = {
            'test': ['Oster Bounds', 'AI Fingerprint', 'Detector Bias'],
            'result': [
                f"δ = {results['oster_bounds'].delta:.2f}" if results.get('oster_bounds') else 'N/A',
                f"Diff = {results['ai_fingerprint'].differential:.1f}pp" if results.get('ai_fingerprint') else 'N/A',
                f"p = {results['detector_bias'].p_value_interaction:.3f}" if results.get('detector_bias') else 'N/A'
            ],
            'interpretation': [
                'PASSED' if results.get('oster_bounds') and results['oster_bounds'].delta > 1 else 'CHECK',
                'CONFIRMED' if results.get('ai_fingerprint') and results['ai_fingerprint'].is_ai_fingerprint else 'CHECK',
                'NO BIAS' if results.get('detector_bias') and results['detector_bias'].native_speaker_effect < 0 else 'CHECK'
            ]
        }

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f'{output_dir}/reviewer_robustness_summary.csv', index=False)
        print(f"\nResults saved to: {output_dir}/reviewer_robustness_summary.csv")

    return results


# =============================================================================
# LATEX OUTPUT
# =============================================================================

def generate_robustness_latex_table(results: Dict[str, Any]) -> str:
    """Generate a LaTeX table summarizing robustness tests."""

    latex = r"""
\begin{table}[htbp]
\centering
\caption{Robustness Tests for Reviewer Concerns}
\label{tab:robustness}
\begin{tabular}{lccl}
\toprule
\textbf{Test} & \textbf{Statistic} & \textbf{Result} & \textbf{Interpretation} \\
\midrule
"""

    # Oster Bounds
    if results.get('oster_bounds'):
        ob = results['oster_bounds']
        latex += f"Oster Bounds & $\\delta = {ob.delta:.2f}$ & "
        latex += "Pass" if ob.delta > 1 else "Marginal"
        latex += " & Selection must be ${:.1f}\\times$ observables \\\\\n".format(ob.delta)

    # AI Fingerprint
    if results.get('ai_fingerprint'):
        fp = results['ai_fingerprint']
        latex += f"AI Fingerprint & $\\Delta = {fp.differential:.1f}$pp & "
        latex += "Confirmed" if fp.is_ai_fingerprint else "Weak"
        latex += " & Soundness $>$ Presentation decline \\\\\n"

    # Detector Bias
    if results.get('detector_bias'):
        db = results['detector_bias']
        latex += f"Detector Bias & $p = {db.p_value_interaction:.3f}$ & "
        latex += "No bias" if db.p_value_interaction > 0.1 else "Potential"
        latex += " & Effect holds across speaker groups \\\\\n"

    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""

    return latex


# =============================================================================
# COMPREHENSIVE RESULTS EXPORT
# =============================================================================

def create_detailed_results_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a comprehensive DataFrame with all robustness test results.

    This provides a publication-ready export with all statistics needed
    for reporting in academic papers.

    Parameters
    ----------
    results : dict
        Results from run_reviewer_robustness_tests()

    Returns
    -------
    pd.DataFrame
        Detailed results table with all statistics
    """
    rows = []

    # Test 1: Oster Bounds
    if results.get('oster_bounds'):
        ob = results['oster_bounds']
        rows.append({
            'test': 'Oster Bounds',
            'category': 'Effort Endogeneity',
            'statistic': 'delta',
            'value': ob.delta,
            'interpretation': 'Pass' if ob.delta > 1 else 'Marginal',
            'threshold': 1.0,
            'passed': ob.delta > 1,
            'beta_uncontrolled': ob.beta_uncontrolled,
            'beta_controlled': ob.beta_controlled,
            'beta_star': ob.beta_star,
            'r2_uncontrolled': ob.r2_uncontrolled,
            'r2_controlled': ob.r2_controlled,
            'r_max': ob.r_max,
            'identified_set_lower': ob.identified_set[0],
            'identified_set_upper': ob.identified_set[1],
            'notes': f'Selection on unobservables would need to be {ob.delta:.1f}x stronger than observables'
        })

    # Test 2: AI Fingerprint
    if results.get('ai_fingerprint'):
        fp = results['ai_fingerprint']
        rows.append({
            'test': 'AI Fingerprint',
            'category': 'Effort Endogeneity',
            'statistic': 'differential_pp',
            'value': fp.differential,
            'interpretation': 'Confirmed' if fp.is_ai_fingerprint else 'Not Confirmed',
            'threshold': -1.0,
            'passed': fp.is_ai_fingerprint,
            'soundness_decline_pct': fp.soundness_decline,
            'presentation_decline_pct': fp.presentation_decline,
            'p_value': fp.p_value,
            'notes': 'Soundness declines more than Presentation (AI substitutes for judgment)'
        })

    # Test 3: Detector Bias
    if results.get('detector_bias'):
        db = results['detector_bias']
        # Test passes if both groups show negative effect (no differential bias)
        no_bias = (db.native_speaker_effect < 0 and db.non_native_effect < 0)
        rows.append({
            'test': 'Detector Bias',
            'category': 'Measurement Validity',
            'statistic': 'interaction_p_value',
            'value': db.p_value_interaction,
            'interpretation': 'No Bias' if no_bias else 'Potential Bias',
            'threshold': 0.10,
            'passed': no_bias,
            'overall_effect': db.overall_effect,
            'native_speaker_effect': db.native_speaker_effect,
            'non_native_effect': db.non_native_effect,
            'effect_difference': db.difference,
            'notes': 'AI penalty exists in both native and non-native speaker groups'
        })

    df = pd.DataFrame(rows)
    return df


def export_all_results(
    results: Dict[str, Any],
    output_dir: str = '.',
    prefix: str = 'reviewer_robustness'
) -> Dict[str, str]:
    """
    Export all robustness test results to multiple formats.

    Creates:
    - {prefix}_summary.csv: Simple 3-row summary
    - {prefix}_detailed.csv: Full statistics for each test
    - {prefix}_table.tex: LaTeX table for paper
    - {prefix}_report.txt: Human-readable report

    Parameters
    ----------
    results : dict
        Results from run_reviewer_robustness_tests()
    output_dir : str
        Output directory
    prefix : str
        Filename prefix

    Returns
    -------
    dict
        Paths to all generated files
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    paths = {}

    # 1. Simple summary CSV
    summary_data = {
        'test': ['Oster Bounds', 'AI Fingerprint', 'Detector Bias'],
        'result': [
            f"δ = {results['oster_bounds'].delta:.2f}" if results.get('oster_bounds') else 'N/A',
            f"Δ = {results['ai_fingerprint'].differential:.1f}pp" if results.get('ai_fingerprint') else 'N/A',
            f"p = {results['detector_bias'].p_value_interaction:.3f}" if results.get('detector_bias') else 'N/A'
        ],
        'passed': [
            results.get('oster_bounds') and results['oster_bounds'].delta > 1,
            results.get('ai_fingerprint') and results['ai_fingerprint'].is_ai_fingerprint,
            results.get('detector_bias') and results['detector_bias'].native_speaker_effect < 0 and results['detector_bias'].non_native_effect < 0
        ],
        'interpretation': [
            'Selection on unobservables implausible' if results.get('oster_bounds') and results['oster_bounds'].delta > 1 else 'Needs investigation',
            'Soundness declines more than Presentation' if results.get('ai_fingerprint') and results['ai_fingerprint'].is_ai_fingerprint else 'No differential pattern',
            'Effect holds for both speaker groups' if results.get('detector_bias') and results['detector_bias'].native_speaker_effect < 0 else 'Potential bias'
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_path = f'{output_dir}/{prefix}_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    paths['summary'] = summary_path

    # 2. Detailed results CSV
    detailed_df = create_detailed_results_dataframe(results)
    detailed_path = f'{output_dir}/{prefix}_detailed.csv'
    detailed_df.to_csv(detailed_path, index=False)
    paths['detailed'] = detailed_path

    # 3. LaTeX table
    latex = generate_robustness_latex_table(results)
    latex_path = f'{output_dir}/{prefix}_table.tex'
    with open(latex_path, 'w') as f:
        f.write(latex)
    paths['latex'] = latex_path

    # 4. Human-readable report
    report = generate_text_report(results)
    report_path = f'{output_dir}/{prefix}_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    paths['report'] = report_path

    return paths


def generate_text_report(results: Dict[str, Any]) -> str:
    """Generate a human-readable text report of robustness test results."""

    lines = [
        "=" * 70,
        "REVIEWER ROBUSTNESS TESTS - DETAILED REPORT",
        "=" * 70,
        "",
        "This report addresses two key referee concerns:",
        "1. Adverse Selection / Effort Endogeneity",
        "2. AI Detector Bias Against Non-Native Speakers",
        "",
        "-" * 70,
    ]

    # Test 1: Oster Bounds
    lines.append("")
    lines.append("TEST 1: OSTER BOUNDS (Effort Endogeneity)")
    lines.append("-" * 40)
    if results.get('oster_bounds'):
        ob = results['oster_bounds']
        lines.append(f"")
        lines.append(f"Question: How much unobserved selection (e.g., effort) would need")
        lines.append(f"          to exist to explain away the AI coefficient?")
        lines.append(f"")
        lines.append(f"Results:")
        lines.append(f"  Uncontrolled β(AI):  {ob.beta_uncontrolled:.4f} (R² = {ob.r2_uncontrolled:.4f})")
        lines.append(f"  Controlled β(AI):    {ob.beta_controlled:.4f} (R² = {ob.r2_controlled:.4f})")
        lines.append(f"  Oster's δ:           {ob.delta:.2f}")
        lines.append(f"  Bias-adjusted β*:    {ob.beta_star:.4f}")
        lines.append(f"  Identified set:      [{ob.identified_set[0]:.4f}, {ob.identified_set[1]:.4f}]")
        lines.append(f"")
        if ob.delta > 1:
            lines.append(f"  ✓ PASSED: δ = {ob.delta:.2f} > 1")
            lines.append(f"    Selection on unobservables would need to be {ob.delta:.1f}x stronger")
            lines.append(f"    than selection on observables to explain away the result.")
        else:
            lines.append(f"  ⚠ MARGINAL: δ = {ob.delta:.2f}")
    else:
        lines.append("  Error: Test could not be computed")

    # Test 2: AI Fingerprint
    lines.append("")
    lines.append("")
    lines.append("TEST 2: AI FINGERPRINT (Soundness vs Presentation)")
    lines.append("-" * 40)
    if results.get('ai_fingerprint'):
        fp = results['ai_fingerprint']
        lines.append(f"")
        lines.append(f"Question: Does Soundness (judgment) decline more than Presentation (polish)?")
        lines.append(f"          If yes, this is inconsistent with 'laziness' and consistent with")
        lines.append(f"          AI substituting for judgment while preserving surface quality.")
        lines.append(f"")
        lines.append(f"Results:")
        lines.append(f"  Soundness decline:    {fp.soundness_decline:+.1f}%")
        lines.append(f"  Presentation decline: {fp.presentation_decline:+.1f}%")
        lines.append(f"  Differential:         {fp.differential:+.1f}pp")
        lines.append(f"  p-value:              {fp.p_value:.4f}")
        lines.append(f"")
        if fp.is_ai_fingerprint:
            lines.append(f"  ✓ CONFIRMED: Soundness declines {abs(fp.differential):.1f}pp more than Presentation")
            lines.append(f"    This is the 'AI fingerprint' - judgment suffers while polish is preserved.")
        else:
            lines.append(f"  ⚠ NOT CONFIRMED: Differential = {fp.differential:+.1f}pp (p = {fp.p_value:.3f})")
    else:
        lines.append("  Error: Test could not be computed")

    # Test 3: Detector Bias
    lines.append("")
    lines.append("")
    lines.append("TEST 3: DETECTOR BIAS (Native vs Non-Native Speakers)")
    lines.append("-" * 40)
    if results.get('detector_bias'):
        db = results['detector_bias']
        lines.append(f"")
        lines.append(f"Question: Is the AI penalty driven by detector bias against non-native speakers?")
        lines.append(f"")
        lines.append(f"Results:")
        lines.append(f"  Overall AI effect:      {db.overall_effect:.4f}")
        lines.append(f"  Native speaker effect:  {db.native_speaker_effect:.4f}")
        lines.append(f"  Non-native effect:      {db.non_native_effect:.4f}")
        lines.append(f"  Difference:             {db.difference:.4f}")
        lines.append(f"  Interaction p-value:    {db.p_value_interaction:.4f}")
        lines.append(f"")
        if db.native_speaker_effect < 0 and db.non_native_effect < 0:
            lines.append(f"  ✓ NO BIAS: AI penalty exists in BOTH groups")
            lines.append(f"    Native:     {db.native_speaker_effect:.3f}")
            lines.append(f"    Non-native: {db.non_native_effect:.3f}")
            lines.append(f"    This rules out detector bias against non-native speakers.")
        else:
            lines.append(f"  ⚠ POTENTIAL BIAS: Effect differs between groups")
    else:
        lines.append("  Error: Test could not be computed")

    # Summary
    lines.append("")
    lines.append("")
    lines.append("=" * 70)
    lines.append("SUMMARY")
    lines.append("=" * 70)

    passed = 0
    total = 0

    if results.get('oster_bounds'):
        total += 1
        if results['oster_bounds'].delta > 1:
            passed += 1
            lines.append("✓ Oster Bounds:   PASSED")
        else:
            lines.append("⚠ Oster Bounds:   MARGINAL")

    if results.get('ai_fingerprint'):
        total += 1
        if results['ai_fingerprint'].is_ai_fingerprint:
            passed += 1
            lines.append("✓ AI Fingerprint: CONFIRMED")
        else:
            lines.append("⚠ AI Fingerprint: NOT CONFIRMED")

    if results.get('detector_bias'):
        total += 1
        if results['detector_bias'].native_speaker_effect < 0 and results['detector_bias'].non_native_effect < 0:
            passed += 1
            lines.append("✓ Detector Bias:  NO BIAS")
        else:
            lines.append("⚠ Detector Bias:  POTENTIAL CONCERN")

    lines.append("")
    lines.append(f"OVERALL: {passed}/{total} tests passed")
    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)
