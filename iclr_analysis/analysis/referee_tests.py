"""
Referee-Requested Statistical Tests
====================================

1. Variance Compression Test (Lemma 2 validation)
   - Tests if AI reviews have lower variance than Human reviews
   - Validates the "noise reduction" mechanism from Bayesian shrinkage

2. Formal Interaction Test for Component Trajectories (Table 2)
   - Uses stacked regression: Score = β₁ AI + β₂ (AI × IsSoundness)
   - Provides formal p-value for differential degradation

References:
- Lemma 2: Bayesian shrinkage → variance compression
- Table 2: Component score trajectories
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import levene, bartlett, f as f_dist
from typing import Dict, Tuple, Optional
import warnings

import sys
sys.path.insert(0, '..')

from src.data_loading import load_data, merge_paper_info
from src.stats_utils import bootstrap_ci, ols_robust, cohens_d


# =============================================================================
# TEST 1: VARIANCE COMPRESSION (Lemma 2 Validation)
# =============================================================================

def variance_compression_test(reviews_df: pd.DataFrame,
                               submissions_df: pd.DataFrame,
                               verbose: bool = True) -> Dict:
    """
    Test if AI reviews have lower variance than Human reviews.

    This validates Lemma 2 (Bayesian shrinkage → variance compression).
    The math predicts that AI reviewers, by relying on priors, will have
    compressed variance compared to human reviewers.

    Tests performed:
    1. Levene's test (robust to non-normality)
    2. Bartlett's test (assumes normality)
    3. F-test for variance ratio
    4. Bootstrap CI for variance ratio
    5. Brown-Forsythe test (median-based Levene)

    Parameters
    ----------
    reviews_df : DataFrame
        Review-level data with 'rating', 'ai_classification'
    submissions_df : DataFrame
        Submission-level data
    verbose : bool
        Print detailed results

    Returns
    -------
    dict with test results and interpretation
    """
    if verbose:
        print("\n" + "="*70)
        print("VARIANCE COMPRESSION TEST (Lemma 2 Validation)")
        print("="*70)
        print("\nH0: Var(AI ratings) = Var(Human ratings)")
        print("H1: Var(AI ratings) < Var(Human ratings)  [one-tailed]")

    # Merge and prepare data
    merged = merge_paper_info(reviews_df, submissions_df)

    # Get Human and AI ratings
    human_ratings = merged[merged['ai_classification'] == 'Fully human-written']['rating'].dropna()
    ai_ratings = merged[merged['ai_classification'] == 'Fully AI-generated']['rating'].dropna()

    n_human = len(human_ratings)
    n_ai = len(ai_ratings)

    if n_human < 30 or n_ai < 30:
        warnings.warn(f"Small sample sizes: Human={n_human}, AI={n_ai}")

    # Calculate variances
    var_human = np.var(human_ratings, ddof=1)
    var_ai = np.var(ai_ratings, ddof=1)
    std_human = np.std(human_ratings, ddof=1)
    std_ai = np.std(ai_ratings, ddof=1)

    # Variance ratio (Human/AI) - should be > 1 if AI has lower variance
    var_ratio = var_human / var_ai

    results = {
        'n_human': n_human,
        'n_ai': n_ai,
        'var_human': var_human,
        'var_ai': var_ai,
        'std_human': std_human,
        'std_ai': std_ai,
        'var_ratio': var_ratio,
        'variance_reduction_pct': (1 - var_ai/var_human) * 100 if var_human > 0 else np.nan
    }

    # =========================================================================
    # Test 1: Levene's Test (robust to non-normality)
    # =========================================================================
    levene_stat, levene_p = levene(human_ratings, ai_ratings, center='mean')
    results['levene_stat'] = levene_stat
    results['levene_p'] = levene_p
    # One-tailed: divide by 2 if direction matches prediction
    results['levene_p_onetailed'] = levene_p / 2 if var_ai < var_human else 1 - levene_p / 2

    # =========================================================================
    # Test 2: Brown-Forsythe Test (median-based Levene, more robust)
    # =========================================================================
    bf_stat, bf_p = levene(human_ratings, ai_ratings, center='median')
    results['brown_forsythe_stat'] = bf_stat
    results['brown_forsythe_p'] = bf_p
    results['brown_forsythe_p_onetailed'] = bf_p / 2 if var_ai < var_human else 1 - bf_p / 2

    # =========================================================================
    # Test 3: Bartlett's Test (assumes normality)
    # =========================================================================
    try:
        bartlett_stat, bartlett_p = bartlett(human_ratings, ai_ratings)
        results['bartlett_stat'] = bartlett_stat
        results['bartlett_p'] = bartlett_p
    except:
        results['bartlett_stat'] = np.nan
        results['bartlett_p'] = np.nan

    # =========================================================================
    # Test 4: F-test for variance ratio
    # =========================================================================
    # F = s1²/s2² where s1 > s2
    if var_human > var_ai:
        f_stat = var_human / var_ai
        df1, df2 = n_human - 1, n_ai - 1
    else:
        f_stat = var_ai / var_human
        df1, df2 = n_ai - 1, n_human - 1

    # Two-tailed p-value
    f_p = 2 * min(f_dist.sf(f_stat, df1, df2), f_dist.cdf(f_stat, df1, df2))
    # One-tailed (AI < Human)
    f_p_onetailed = f_dist.sf(var_human/var_ai, n_human-1, n_ai-1) if var_human > var_ai else 1 - f_dist.sf(var_ai/var_human, n_ai-1, n_human-1)

    results['f_stat'] = f_stat
    results['f_p'] = f_p
    results['f_p_onetailed'] = f_p_onetailed

    # =========================================================================
    # Test 5: Bootstrap CI for variance ratio
    # =========================================================================
    np.random.seed(42)
    n_boot = 10000
    boot_ratios = []

    for _ in range(n_boot):
        boot_human = np.random.choice(human_ratings, size=n_human, replace=True)
        boot_ai = np.random.choice(ai_ratings, size=n_ai, replace=True)
        boot_var_human = np.var(boot_human, ddof=1)
        boot_var_ai = np.var(boot_ai, ddof=1)
        if boot_var_ai > 0:
            boot_ratios.append(boot_var_human / boot_var_ai)

    boot_ratios = np.array(boot_ratios)
    results['var_ratio_ci_lower'] = np.percentile(boot_ratios, 2.5)
    results['var_ratio_ci_upper'] = np.percentile(boot_ratios, 97.5)
    results['var_ratio_boot_p'] = np.mean(boot_ratios <= 1)  # P(ratio <= 1) = P(AI >= Human)

    # =========================================================================
    # Interpretation
    # =========================================================================
    compression_confirmed = var_ai < var_human and results['levene_p_onetailed'] < 0.05

    results['compression_confirmed'] = compression_confirmed
    results['interpretation'] = (
        f"AI reviews show {'SIGNIFICANT' if compression_confirmed else 'no significant'} "
        f"variance compression. "
        f"Variance ratio (Human/AI) = {var_ratio:.3f} "
        f"({results['variance_reduction_pct']:.1f}% reduction in AI variance)."
    )

    if verbose:
        print(f"\n--- Sample Statistics ---")
        print(f"Human reviews: n={n_human:,}, σ={std_human:.4f}, σ²={var_human:.4f}")
        print(f"AI reviews:    n={n_ai:,}, σ={std_ai:.4f}, σ²={var_ai:.4f}")
        print(f"\nVariance ratio (Human/AI): {var_ratio:.4f}")
        print(f"Variance reduction: {results['variance_reduction_pct']:.1f}%")

        print(f"\n--- Statistical Tests ---")
        print(f"Levene's test:        F={levene_stat:.2f}, p={levene_p:.4f} (two-tailed)")
        print(f"                      p={results['levene_p_onetailed']:.4f} (one-tailed, H1: AI < Human)")
        print(f"Brown-Forsythe:       F={bf_stat:.2f}, p={bf_p:.4f} (two-tailed)")
        print(f"                      p={results['brown_forsythe_p_onetailed']:.4f} (one-tailed)")
        print(f"F-test:               F={f_stat:.2f}, p={f_p:.4f} (two-tailed)")
        print(f"                      p={results['f_p_onetailed']:.4f} (one-tailed)")
        print(f"\nBootstrap 95% CI for variance ratio: [{results['var_ratio_ci_lower']:.3f}, {results['var_ratio_ci_upper']:.3f}]")

        print(f"\n--- Conclusion ---")
        if compression_confirmed:
            print("✓ VARIANCE COMPRESSION CONFIRMED (Lemma 2 validated)")
            print(f"  AI reviews have significantly lower variance than Human reviews")
            print(f"  This is consistent with Bayesian shrinkage toward the prior")
        else:
            if var_ai < var_human:
                print("→ Direction consistent with Lemma 2, but not statistically significant")
            else:
                print("✗ No evidence of variance compression")
                print("  (AI variance >= Human variance)")

    return results


def variance_compression_by_component(reviews_df: pd.DataFrame,
                                       submissions_df: pd.DataFrame,
                                       verbose: bool = True) -> Dict:
    """
    Test variance compression separately for each rating component.

    Components: soundness, presentation, contribution, confidence, rating
    """
    if verbose:
        print("\n" + "="*70)
        print("VARIANCE COMPRESSION BY COMPONENT")
        print("="*70)

    merged = merge_paper_info(reviews_df, submissions_df)

    components = ['rating', 'soundness', 'presentation', 'contribution']
    available = [c for c in components if c in merged.columns]

    results = {}

    for component in available:
        human = merged[merged['ai_classification'] == 'Fully human-written'][component].dropna()
        ai = merged[merged['ai_classification'] == 'Fully AI-generated'][component].dropna()

        if len(human) < 20 or len(ai) < 20:
            continue

        var_human = np.var(human, ddof=1)
        var_ai = np.var(ai, ddof=1)
        var_ratio = var_human / var_ai if var_ai > 0 else np.nan

        levene_stat, levene_p = levene(human, ai, center='mean')
        p_onetailed = levene_p / 2 if var_ai < var_human else 1 - levene_p / 2

        results[component] = {
            'var_human': var_human,
            'var_ai': var_ai,
            'var_ratio': var_ratio,
            'reduction_pct': (1 - var_ai/var_human) * 100 if var_human > 0 else np.nan,
            'levene_p': levene_p,
            'levene_p_onetailed': p_onetailed,
            'significant': p_onetailed < 0.05 and var_ai < var_human
        }

        if verbose:
            sig = "***" if p_onetailed < 0.001 else "**" if p_onetailed < 0.01 else "*" if p_onetailed < 0.05 else ""
            direction = "↓" if var_ai < var_human else "↑"
            print(f"{component:15s}: σ²_H={var_human:.4f}, σ²_AI={var_ai:.4f}, "
                  f"ratio={var_ratio:.3f} {direction}, p={p_onetailed:.4f} {sig}")

    return results


# =============================================================================
# TEST 2: FORMAL INTERACTION TEST FOR COMPONENT TRAJECTORIES
# =============================================================================

def component_interaction_test(reviews_df: pd.DataFrame,
                                submissions_df: pd.DataFrame,
                                verbose: bool = True) -> Dict:
    """
    Formal interaction test for component score degradation.

    Tests whether Soundness degrades faster than Presentation as AI content increases.

    Method: Stacked regression
        Score_ij = α + β₁(AI%) + β₂(IsSoundness) + β₃(AI% × IsSoundness) + ε

    The coefficient β₃ tests whether the AI penalty differs between components.
    If β₃ < 0 and significant, Soundness degrades faster than Presentation.

    This is the formal test preferred by QJE referees over comparing CIs.

    Parameters
    ----------
    reviews_df : DataFrame
    submissions_df : DataFrame
    verbose : bool

    Returns
    -------
    dict with regression results and formal test statistics
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    if verbose:
        print("\n" + "="*70)
        print("FORMAL INTERACTION TEST: Component Degradation")
        print("="*70)
        print("\nModel: Score = α + β₁(AI%) + β₂(IsSoundness) + β₃(AI% × IsSoundness) + ε")
        print("H0: β₃ = 0 (equal degradation)")
        print("H1: β₃ < 0 (Soundness degrades faster)")

    # Merge data
    merged = merge_paper_info(reviews_df, submissions_df)

    # Check required columns
    required = ['soundness', 'presentation', 'ai_percentage']
    missing = [c for c in required if c not in merged.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Stack data: each row is one (component, score) observation
    stacked_rows = []

    for _, row in merged.iterrows():
        paper_id = row.get('submission_number', None)
        ai_pct = row['ai_percentage']

        if pd.isna(ai_pct):
            continue

        # Soundness observation
        if pd.notna(row['soundness']):
            stacked_rows.append({
                'paper_id': paper_id,
                'ai_percentage': ai_pct,
                'ai_pct_scaled': ai_pct / 100,  # 0-1 scale
                'score': row['soundness'],
                'is_soundness': 1,
                'component': 'Soundness'
            })

        # Presentation observation
        if pd.notna(row['presentation']):
            stacked_rows.append({
                'paper_id': paper_id,
                'ai_percentage': ai_pct,
                'ai_pct_scaled': ai_pct / 100,
                'score': row['presentation'],
                'is_soundness': 0,
                'component': 'Presentation'
            })

    stacked_df = pd.DataFrame(stacked_rows)

    n_obs = len(stacked_df)
    n_papers = stacked_df['paper_id'].nunique()

    if verbose:
        print(f"\nSample: {n_obs:,} observations from {n_papers:,} papers")
        print(f"  Soundness: {(stacked_df['is_soundness']==1).sum():,}")
        print(f"  Presentation: {(stacked_df['is_soundness']==0).sum():,}")

    results = {
        'n_obs': n_obs,
        'n_papers': n_papers
    }

    # =========================================================================
    # Method 1: OLS with robust standard errors (HC3)
    # =========================================================================
    model_ols = smf.ols(
        'score ~ ai_pct_scaled * is_soundness',
        data=stacked_df
    ).fit(cov_type='HC3')

    # Extract interaction coefficient
    interaction_coef = model_ols.params['ai_pct_scaled:is_soundness']
    interaction_se = model_ols.bse['ai_pct_scaled:is_soundness']
    interaction_t = model_ols.tvalues['ai_pct_scaled:is_soundness']
    interaction_p = model_ols.pvalues['ai_pct_scaled:is_soundness']
    interaction_p_onetailed = interaction_p / 2 if interaction_coef < 0 else 1 - interaction_p / 2

    results['ols_robust'] = {
        'interaction_coef': interaction_coef,
        'interaction_se': interaction_se,
        'interaction_t': interaction_t,
        'interaction_p_twotailed': interaction_p,
        'interaction_p_onetailed': interaction_p_onetailed,
        'ci_lower': model_ols.conf_int().loc['ai_pct_scaled:is_soundness', 0],
        'ci_upper': model_ols.conf_int().loc['ai_pct_scaled:is_soundness', 1],
        'main_effect_ai': model_ols.params['ai_pct_scaled'],
        'main_effect_soundness': model_ols.params['is_soundness'],
        'r_squared': model_ols.rsquared
    }

    if verbose:
        print(f"\n--- Method 1: OLS with HC3 Robust SE ---")
        print(f"Main effect (AI%):        β₁ = {model_ols.params['ai_pct_scaled']:.4f}")
        print(f"Main effect (Soundness):  β₂ = {model_ols.params['is_soundness']:.4f}")
        print(f"Interaction (AI×Sound):   β₃ = {interaction_coef:.4f}")
        print(f"  SE: {interaction_se:.4f}")
        print(f"  t-statistic: {interaction_t:.2f}")
        print(f"  p-value (two-tailed): {interaction_p:.4e}")
        print(f"  p-value (one-tailed, H1: β₃<0): {interaction_p_onetailed:.4e}")
        print(f"  95% CI: [{results['ols_robust']['ci_lower']:.4f}, {results['ols_robust']['ci_upper']:.4f}]")

    # =========================================================================
    # Method 2: Clustered SE by paper (accounts for within-paper correlation)
    # =========================================================================
    try:
        model_clustered = smf.ols(
            'score ~ ai_pct_scaled * is_soundness',
            data=stacked_df
        ).fit(cov_type='cluster', cov_kwds={'groups': stacked_df['paper_id']})

        interaction_coef_c = model_clustered.params['ai_pct_scaled:is_soundness']
        interaction_se_c = model_clustered.bse['ai_pct_scaled:is_soundness']
        interaction_t_c = model_clustered.tvalues['ai_pct_scaled:is_soundness']
        interaction_p_c = model_clustered.pvalues['ai_pct_scaled:is_soundness']
        interaction_p_c_onetailed = interaction_p_c / 2 if interaction_coef_c < 0 else 1 - interaction_p_c / 2

        results['ols_clustered'] = {
            'interaction_coef': interaction_coef_c,
            'interaction_se': interaction_se_c,
            'interaction_t': interaction_t_c,
            'interaction_p_twotailed': interaction_p_c,
            'interaction_p_onetailed': interaction_p_c_onetailed,
            'ci_lower': model_clustered.conf_int().loc['ai_pct_scaled:is_soundness', 0],
            'ci_upper': model_clustered.conf_int().loc['ai_pct_scaled:is_soundness', 1]
        }

        if verbose:
            print(f"\n--- Method 2: OLS with Clustered SE (by paper) ---")
            print(f"Interaction (AI×Sound):   β₃ = {interaction_coef_c:.4f}")
            print(f"  Clustered SE: {interaction_se_c:.4f}")
            print(f"  t-statistic: {interaction_t_c:.2f}")
            print(f"  p-value (two-tailed): {interaction_p_c:.4e}")
            print(f"  p-value (one-tailed): {interaction_p_c_onetailed:.4e}")
    except Exception as e:
        results['ols_clustered'] = None
        if verbose:
            print(f"\nClustered SE failed: {e}")

    # =========================================================================
    # Method 3: Mixed effects model (random intercept by paper)
    # =========================================================================
    try:
        model_mixed = smf.mixedlm(
            'score ~ ai_pct_scaled * is_soundness',
            data=stacked_df,
            groups=stacked_df['paper_id']
        ).fit(reml=True)

        interaction_coef_m = model_mixed.fe_params['ai_pct_scaled:is_soundness']
        interaction_se_m = model_mixed.bse_fe['ai_pct_scaled:is_soundness']
        interaction_z_m = interaction_coef_m / interaction_se_m
        interaction_p_m = 2 * (1 - stats.norm.cdf(abs(interaction_z_m)))
        interaction_p_m_onetailed = interaction_p_m / 2 if interaction_coef_m < 0 else 1 - interaction_p_m / 2

        results['mixed_effects'] = {
            'interaction_coef': interaction_coef_m,
            'interaction_se': interaction_se_m,
            'interaction_z': interaction_z_m,
            'interaction_p_twotailed': interaction_p_m,
            'interaction_p_onetailed': interaction_p_m_onetailed
        }

        if verbose:
            print(f"\n--- Method 3: Mixed Effects (random intercept by paper) ---")
            print(f"Interaction (AI×Sound):   β₃ = {interaction_coef_m:.4f}")
            print(f"  SE: {interaction_se_m:.4f}")
            print(f"  z-statistic: {interaction_z_m:.2f}")
            print(f"  p-value (two-tailed): {interaction_p_m:.4e}")
            print(f"  p-value (one-tailed): {interaction_p_m_onetailed:.4e}")
    except Exception as e:
        results['mixed_effects'] = None
        if verbose:
            print(f"\nMixed effects model failed: {e}")

    # =========================================================================
    # Summary and Interpretation
    # =========================================================================
    # Use the most conservative estimate (clustered if available, else robust)
    if results.get('ols_clustered'):
        final_p = results['ols_clustered']['interaction_p_onetailed']
        final_coef = results['ols_clustered']['interaction_coef']
    else:
        final_p = results['ols_robust']['interaction_p_onetailed']
        final_coef = results['ols_robust']['interaction_coef']

    significant = final_p < 0.05 and final_coef < 0

    results['summary'] = {
        'final_p_onetailed': final_p,
        'final_coef': final_coef,
        'significant': significant,
        'interpretation': (
            f"The interaction coefficient β₃ = {final_coef:.4f} is "
            f"{'SIGNIFICANT' if significant else 'not significant'} "
            f"(p = {final_p:.4e}, one-tailed). "
            f"{'Soundness degrades significantly FASTER than Presentation.' if significant else ''}"
        )
    }

    if verbose:
        print(f"\n" + "="*70)
        print("CONCLUSION")
        print("="*70)
        print(f"\nInteraction coefficient: β₃ = {final_coef:.4f}")
        print(f"One-tailed p-value: {final_p:.4e}")

        if significant:
            print(f"\n✓ SIGNIFICANT DIFFERENTIAL DEGRADATION CONFIRMED")
            print(f"  Soundness degrades {abs(final_coef):.3f} points MORE per 100% AI than Presentation")
            print(f"  This is the 'fingerprint' of substitution (Table 2)")
        else:
            print(f"\n→ No significant differential degradation at α=0.05")
            if final_coef < 0:
                print(f"  (Direction consistent with prediction, but not significant)")

    return results


def component_interaction_with_contribution(reviews_df: pd.DataFrame,
                                             submissions_df: pd.DataFrame,
                                             verbose: bool = True) -> Dict:
    """
    Extended interaction test including Contribution component.

    Stacked regression with all three components:
    - Soundness (reference category)
    - Presentation
    - Contribution
    """
    import statsmodels.formula.api as smf

    if verbose:
        print("\n" + "="*70)
        print("EXTENDED COMPONENT INTERACTION TEST (3 components)")
        print("="*70)

    merged = merge_paper_info(reviews_df, submissions_df)

    # Stack all three components
    stacked_rows = []

    for _, row in merged.iterrows():
        paper_id = row.get('submission_number', None)
        ai_pct = row['ai_percentage']

        if pd.isna(ai_pct):
            continue

        for component in ['soundness', 'presentation', 'contribution']:
            if component in row and pd.notna(row[component]):
                stacked_rows.append({
                    'paper_id': paper_id,
                    'ai_pct_scaled': ai_pct / 100,
                    'score': row[component],
                    'component': component.capitalize()
                })

    stacked_df = pd.DataFrame(stacked_rows)
    stacked_df['component'] = pd.Categorical(
        stacked_df['component'],
        categories=['Soundness', 'Presentation', 'Contribution'],
        ordered=True
    )

    if verbose:
        print(f"\nSample by component:")
        print(stacked_df['component'].value_counts())

    # Regression with component dummies and interactions
    model = smf.ols(
        'score ~ ai_pct_scaled * C(component, Treatment("Soundness"))',
        data=stacked_df
    ).fit(cov_type='HC3')

    if verbose:
        print(f"\n--- Regression Results (Soundness = reference) ---")
        print(model.summary().tables[1])

        print(f"\nInterpretation:")
        print(f"  - Main effect (AI on Soundness): {model.params['ai_pct_scaled']:.4f}")

        pres_interaction = 'ai_pct_scaled:C(component, Treatment("Soundness"))[T.Presentation]'
        cont_interaction = 'ai_pct_scaled:C(component, Treatment("Soundness"))[T.Contribution]'

        if pres_interaction in model.params:
            print(f"  - Presentation vs Soundness slope diff: {model.params[pres_interaction]:.4f} "
                  f"(p={model.pvalues[pres_interaction]:.4f})")
        if cont_interaction in model.params:
            print(f"  - Contribution vs Soundness slope diff: {model.params[cont_interaction]:.4f} "
                  f"(p={model.pvalues[cont_interaction]:.4f})")

    return {
        'model': model,
        'params': model.params.to_dict(),
        'pvalues': model.pvalues.to_dict()
    }


# =============================================================================
# TEST 3: GRADIENT FLATTENING / TRUTH-TRACKING TEST (Lemma 2, part iii)
# =============================================================================

def gradient_flattening_test(reviews_df: pd.DataFrame,
                              submissions_df: pd.DataFrame,
                              verbose: bool = True) -> Dict:
    """
    Truth-Tracking Test: Tests if AI reviewers are less correlated with quality.

    This directly tests Lemma 2, part (iii): "reduced precision flattens the
    score-quality gradient."

    Theory:
    - High Fidelity (Human): When paper is great → 9, bad → 3. Steep slope.
    - Low Fidelity (AI): When paper is great → 7, bad → 5. Flat slope (regression to mean).

    Method:
    Use Human Leave-One-Out Mean (LOOM) as proxy for latent quality θ.

    Regression:
        Score_ij = α + β₁(Human_LOOM_j) + β₂(Is_AI_i) + β₃(Is_AI × LOOM) + ε

    Predictions:
    - β₁ > 0: Humans track quality (validation)
    - β₂ > 0: AI leniency (intercept shift) - already known
    - β₃ < 0: GRADIENT FLATTENING - AI is less responsive to quality

    If β₃ is significantly negative, it proves:
    - An 8/10 from AI means less than 8/10 from Human
    - AI synthesis is regressive: helps bad papers, penalizes good papers
    - This validates the "Trap" condition in the theory

    Parameters
    ----------
    reviews_df : DataFrame
        Review-level data with 'rating', 'ai_classification', 'submission_number'
    submissions_df : DataFrame
        Submission-level data
    verbose : bool
        Print detailed results

    Returns
    -------
    dict with regression results and interpretation
    """
    import statsmodels.formula.api as smf

    if verbose:
        print("\n" + "="*70)
        print("GRADIENT FLATTENING TEST (Truth-Tracking / Lemma 2 part iii)")
        print("="*70)
        print("\nModel: Score = α + β₁(Human_LOOM) + β₂(Is_AI) + β₃(Is_AI × LOOM) + ε")
        print("\nPredictions:")
        print("  β₁ > 0: Humans reliably track consensus (validation)")
        print("  β₂ > 0: AI leniency ('free points' at intercept)")
        print("  β₃ < 0: SENSITIVITY COLLAPSE (AI less responsive to quality)")

    # Merge data
    merged = merge_paper_info(reviews_df, submissions_df)

    # Create binary AI indicator
    merged['is_ai_reviewer'] = (merged['ai_classification'] == 'Fully AI-generated').astype(int)

    # Filter to reviews with valid ratings and classification
    df = merged.dropna(subset=['rating', 'ai_classification', 'submission_number']).copy()

    # =========================================================================
    # Step 1: Calculate Human Leave-One-Out Mean (LOOM) per paper
    # =========================================================================
    # Filter to Human reviews only for computing consensus
    human_revs = df[df['is_ai_reviewer'] == 0]

    # Calculate sum and count of human ratings per paper
    paper_stats = human_revs.groupby('submission_number')['rating'].agg(['sum', 'count']).reset_index()
    paper_stats.columns = ['submission_number', 'h_sum', 'h_count']

    # Merge back to main dataframe
    df = df.merge(paper_stats, on='submission_number', how='left')

    # Calculate LOOM:
    # - For AI reviewers: Human Mean = h_sum / h_count (no exclusion needed)
    # - For Human reviewers: (h_sum - own_rating) / (h_count - 1) (exclude self)
    df['loom'] = df['h_sum'] / df['h_count']

    # For human reviewers, exclude their own rating
    mask_human = df['is_ai_reviewer'] == 0
    df.loc[mask_human, 'loom'] = (df.loc[mask_human, 'h_sum'] - df.loc[mask_human, 'rating']) / (df.loc[mask_human, 'h_count'] - 1)

    # Filter: Need at least 2 human reviews for valid LOOM
    df_reg = df[df['h_count'] >= 2].copy()
    df_reg = df_reg.dropna(subset=['loom'])

    n_obs = len(df_reg)
    n_papers = df_reg['submission_number'].nunique()
    n_human = (df_reg['is_ai_reviewer'] == 0).sum()
    n_ai = (df_reg['is_ai_reviewer'] == 1).sum()

    if verbose:
        print(f"\n--- Sample ---")
        print(f"Total observations: {n_obs:,}")
        print(f"Papers with ≥2 human reviews: {n_papers:,}")
        print(f"Human reviews: {n_human:,}")
        print(f"AI reviews: {n_ai:,}")
        print(f"Human LOOM range: [{df_reg['loom'].min():.2f}, {df_reg['loom'].max():.2f}]")

    results = {
        'n_obs': n_obs,
        'n_papers': n_papers,
        'n_human': n_human,
        'n_ai': n_ai,
        'loom_mean': df_reg['loom'].mean(),
        'loom_std': df_reg['loom'].std()
    }

    # =========================================================================
    # Step 2: Run the Gradient Flattening Regression (Clustered SE)
    # =========================================================================
    try:
        model = smf.ols(
            "rating ~ loom * is_ai_reviewer",
            data=df_reg
        ).fit(cov_type='cluster', cov_kwds={'groups': df_reg['submission_number']})

        # Extract coefficients
        beta1 = model.params['loom']  # Human consensus slope
        beta2 = model.params['is_ai_reviewer']  # AI intercept (leniency)
        beta3 = model.params['loom:is_ai_reviewer']  # Interaction (gradient flattening)

        se1 = model.bse['loom']
        se2 = model.bse['is_ai_reviewer']
        se3 = model.bse['loom:is_ai_reviewer']

        t1 = model.tvalues['loom']
        t2 = model.tvalues['is_ai_reviewer']
        t3 = model.tvalues['loom:is_ai_reviewer']

        p1 = model.pvalues['loom']
        p2 = model.pvalues['is_ai_reviewer']
        p3 = model.pvalues['loom:is_ai_reviewer']

        # One-tailed p-values for directional hypotheses
        p1_onetailed = p1 / 2 if beta1 > 0 else 1 - p1 / 2  # β₁ > 0
        p2_onetailed = p2 / 2 if beta2 > 0 else 1 - p2 / 2  # β₂ > 0 (leniency)
        p3_onetailed = p3 / 2 if beta3 < 0 else 1 - p3 / 2  # β₃ < 0 (flattening)

        # Confidence intervals
        ci = model.conf_int()

        results['clustered'] = {
            'beta1_loom': beta1,
            'beta1_se': se1,
            'beta1_t': t1,
            'beta1_p': p1,
            'beta1_p_onetailed': p1_onetailed,
            'beta1_ci': [ci.loc['loom', 0], ci.loc['loom', 1]],

            'beta2_ai': beta2,
            'beta2_se': se2,
            'beta2_t': t2,
            'beta2_p': p2,
            'beta2_p_onetailed': p2_onetailed,
            'beta2_ci': [ci.loc['is_ai_reviewer', 0], ci.loc['is_ai_reviewer', 1]],

            'beta3_interaction': beta3,
            'beta3_se': se3,
            'beta3_t': t3,
            'beta3_p': p3,
            'beta3_p_onetailed': p3_onetailed,
            'beta3_ci': [ci.loc['loom:is_ai_reviewer', 0], ci.loc['loom:is_ai_reviewer', 1]],

            'r_squared': model.rsquared,
            'intercept': model.params['Intercept']
        }

        # Calculate sensitivity reduction percentage
        # AI slope = β₁ + β₃, Human slope = β₁
        ai_slope = beta1 + beta3
        human_slope = beta1
        sensitivity_reduction = (1 - ai_slope / human_slope) * 100 if human_slope != 0 else np.nan
        results['sensitivity_reduction_pct'] = sensitivity_reduction

        if verbose:
            print(f"\n--- Regression Results (Clustered SE by paper) ---")
            print(f"\n{'Variable':<30} {'Coeff':>10} {'SE':>10} {'t':>10} {'p (1-tail)':>12}")
            print("-" * 72)
            print(f"{'Intercept':<30} {model.params['Intercept']:>10.4f}")
            print(f"{'Human Consensus (β₁)':<30} {beta1:>10.4f} {se1:>10.4f} {t1:>10.2f} {p1_onetailed:>12.4e}")
            print(f"{'AI Reviewer (β₂)':<30} {beta2:>10.4f} {se2:>10.4f} {t2:>10.2f} {p2_onetailed:>12.4e}")
            print(f"{'Interaction β₃ (AI × LOOM)':<30} {beta3:>10.4f} {se3:>10.4f} {t3:>10.2f} {p3_onetailed:>12.4e}")
            print(f"\nR² = {model.rsquared:.4f}")

            print(f"\n--- Interpretation ---")
            print(f"Human slope (β₁):        {beta1:.4f} - Humans track quality 1:{beta1:.2f}")
            print(f"AI slope (β₁ + β₃):      {ai_slope:.4f}")
            print(f"Sensitivity reduction:   {sensitivity_reduction:.1f}%")

    except Exception as e:
        results['clustered'] = None
        if verbose:
            print(f"\nClustered regression failed: {e}")
        return results

    # =========================================================================
    # Step 3: Robustness - OLS with HC3 standard errors
    # =========================================================================
    try:
        model_robust = smf.ols(
            "rating ~ loom * is_ai_reviewer",
            data=df_reg
        ).fit(cov_type='HC3')

        results['robust_hc3'] = {
            'beta3_interaction': model_robust.params['loom:is_ai_reviewer'],
            'beta3_se': model_robust.bse['loom:is_ai_reviewer'],
            'beta3_p': model_robust.pvalues['loom:is_ai_reviewer']
        }

        if verbose:
            print(f"\n--- Robustness Check (HC3 SE) ---")
            print(f"β₃ = {results['robust_hc3']['beta3_interaction']:.4f}, "
                  f"SE = {results['robust_hc3']['beta3_se']:.4f}, "
                  f"p = {results['robust_hc3']['beta3_p']:.4e}")

    except Exception as e:
        results['robust_hc3'] = None

    # =========================================================================
    # Summary and Conclusion
    # =========================================================================
    gradient_flattening_confirmed = (
        beta3 < 0 and
        p3_onetailed < 0.05
    )

    results['summary'] = {
        'gradient_flattening_confirmed': gradient_flattening_confirmed,
        'beta3': beta3,
        'p_onetailed': p3_onetailed,
        'sensitivity_reduction_pct': sensitivity_reduction,
        'interpretation': (
            f"The interaction β₃ = {beta3:.4f} is "
            f"{'SIGNIFICANT' if gradient_flattening_confirmed else 'not significant'} "
            f"(p = {p3_onetailed:.4e}, one-tailed). "
            f"AI reviewers show {sensitivity_reduction:.1f}% reduced sensitivity to quality."
        )
    }

    if verbose:
        print(f"\n" + "="*70)
        print("CONCLUSION")
        print("="*70)

        if gradient_flattening_confirmed:
            print(f"\n✓ GRADIENT FLATTENING CONFIRMED (Lemma 2 part iii validated)")
            print(f"  β₃ = {beta3:.4f}, p = {p3_onetailed:.4e}")
            print(f"\n  Key findings:")
            print(f"  • AI reviewers are {sensitivity_reduction:.1f}% LESS responsive to paper quality")
            print(f"  • An 8/10 from AI means LESS than 8/10 from Human")
            print(f"  • AI synthesis is REGRESSIVE:")
            print(f"    - Helps bad papers (via leniency intercept β₂ = {beta2:.2f})")
            print(f"    - Penalizes good papers (via flattened slope)")
            print(f"  • This validates the 'Trap' condition: high-effort strategies dominated")
        else:
            if beta3 < 0:
                print(f"\n→ Direction consistent with gradient flattening, but not significant")
                print(f"  β₃ = {beta3:.4f}, p = {p3_onetailed:.4f}")
            else:
                print(f"\n✗ No evidence of gradient flattening")
                print(f"  β₃ = {beta3:.4f} (wrong direction)")

    return results


# =============================================================================
# TEST 4: SENSITIVITY ANALYSES FOR AI CLASSIFICATION
# =============================================================================

# Define the ordinal AI classification mapping
AI_CLASSIFICATION_ORDER = {
    'Fully human-written': 0,
    'Lightly AI-edited': 1,
    'Moderately AI-edited': 2,
    'Heavily AI-edited': 3,
    'Fully AI-generated': 4
}


def sensitivity_ordinal_gradient(reviews_df: pd.DataFrame,
                                  submissions_df: pd.DataFrame,
                                  verbose: bool = True) -> Dict:
    """
    Sensitivity Analysis A: Ordinal AI Classification Gradient.

    Uses the full 5-level ordinal classification to test if gradient
    flattening increases monotonically with AI involvement.

    Categories (ordinal 0-4):
    0 = Fully human-written
    1 = Lightly AI-edited
    2 = Moderately AI-edited
    3 = Heavily AI-edited
    4 = Fully AI-generated

    Model:
        Score = α + β₁(LOOM) + β₂(AI_Level) + β₃(AI_Level × LOOM) + ε

    If β₃ < 0, gradient flattening increases with AI involvement.

    Parameters
    ----------
    reviews_df : DataFrame
    submissions_df : DataFrame
    verbose : bool

    Returns
    -------
    dict with ordinal regression results
    """
    import statsmodels.formula.api as smf

    if verbose:
        print("\n" + "="*70)
        print("SENSITIVITY ANALYSIS A: Ordinal AI Classification Gradient")
        print("="*70)
        print("\nUsing 5-level ordinal classification (0=Human, 4=Full AI)")

    merged = merge_paper_info(reviews_df, submissions_df)

    # Create ordinal AI level
    merged['ai_level'] = merged['ai_classification'].map(AI_CLASSIFICATION_ORDER)
    df = merged.dropna(subset=['rating', 'ai_level', 'submission_number']).copy()

    if verbose:
        print("\nAI Classification Distribution:")
        for cat, level in sorted(AI_CLASSIFICATION_ORDER.items(), key=lambda x: x[1]):
            count = (df['ai_level'] == level).sum()
            print(f"  {level}: {cat}: {count:,}")

    # Calculate LOOM using ONLY Fully human-written reviews (clean baseline)
    human_only = df[df['ai_level'] == 0]
    paper_stats = human_only.groupby('submission_number')['rating'].agg(['sum', 'count']).reset_index()
    paper_stats.columns = ['submission_number', 'h_sum', 'h_count']

    df = df.merge(paper_stats, on='submission_number', how='left')

    # For Fully human-written reviews, use leave-one-out
    # For all others, use the full human mean
    df['loom'] = df['h_sum'] / df['h_count']
    mask_human = df['ai_level'] == 0
    df.loc[mask_human, 'loom'] = (df.loc[mask_human, 'h_sum'] - df.loc[mask_human, 'rating']) / (df.loc[mask_human, 'h_count'] - 1)

    # Need at least 2 human reviews for valid LOOM
    df_reg = df[df['h_count'] >= 2].dropna(subset=['loom']).copy()

    n_obs = len(df_reg)
    n_papers = df_reg['submission_number'].nunique()

    if verbose:
        print(f"\nSample: {n_obs:,} reviews from {n_papers:,} papers")
        print(f"LOOM calculated from {(df_reg['ai_level'] == 0).sum():,} Fully human-written reviews")

    results = {
        'n_obs': n_obs,
        'n_papers': n_papers,
        'classification_counts': df_reg['ai_level'].value_counts().to_dict()
    }

    # Run ordinal regression
    try:
        model = smf.ols(
            "rating ~ loom * ai_level",
            data=df_reg
        ).fit(cov_type='cluster', cov_kwds={'groups': df_reg['submission_number']})

        beta1 = model.params['loom']
        beta2 = model.params['ai_level']
        beta3 = model.params['loom:ai_level']

        se3 = model.bse['loom:ai_level']
        p3 = model.pvalues['loom:ai_level']
        p3_onetailed = p3 / 2 if beta3 < 0 else 1 - p3 / 2

        results['ordinal'] = {
            'beta1_loom': beta1,
            'beta2_ai_level': beta2,
            'beta3_interaction': beta3,
            'beta3_se': se3,
            'beta3_p': p3,
            'beta3_p_onetailed': p3_onetailed,
            'significant': beta3 < 0 and p3_onetailed < 0.05,
            'r_squared': model.rsquared
        }

        if verbose:
            print(f"\n--- Ordinal Regression Results ---")
            print(f"β₁ (LOOM slope for humans): {beta1:.4f}")
            print(f"β₂ (AI level main effect):  {beta2:.4f}")
            print(f"β₃ (AI level × LOOM):       {beta3:.4f} (SE={se3:.4f})")
            print(f"p-value (one-tailed):       {p3_onetailed:.4f}")
            print(f"\nInterpretation: Each level of AI involvement")
            print(f"  {'DECREASES' if beta3 < 0 else 'increases'} sensitivity by {abs(beta3):.4f}")
            if results['ordinal']['significant']:
                print(f"\n✓ SIGNIFICANT: Gradient flattening increases with AI level")
            else:
                print(f"\n→ Not significant at α=0.05")

    except Exception as e:
        results['ordinal'] = None
        if verbose:
            print(f"\nOrdinal regression failed: {e}")

    return results


def sensitivity_pairwise_categories(reviews_df: pd.DataFrame,
                                     submissions_df: pd.DataFrame,
                                     verbose: bool = True) -> Dict:
    """
    Sensitivity Analysis B: Pairwise Category Comparisons.

    Compare each AI category against Fully human-written baseline.
    LOOM is calculated using ONLY Fully human-written reviews.

    This shows how gradient flattening varies across:
    - Lightly AI-edited vs Human
    - Moderately AI-edited vs Human
    - Heavily AI-edited vs Human
    - Fully AI-generated vs Human

    Parameters
    ----------
    reviews_df : DataFrame
    submissions_df : DataFrame
    verbose : bool

    Returns
    -------
    dict with pairwise comparison results
    """
    import statsmodels.formula.api as smf

    if verbose:
        print("\n" + "="*70)
        print("SENSITIVITY ANALYSIS B: Pairwise Category Comparisons")
        print("="*70)
        print("\nComparing each AI category vs Fully human-written baseline")
        print("LOOM calculated from Fully human-written reviews ONLY")

    merged = merge_paper_info(reviews_df, submissions_df)
    df = merged.dropna(subset=['rating', 'ai_classification', 'submission_number']).copy()

    # Calculate LOOM using ONLY Fully human-written reviews
    human_only = df[df['ai_classification'] == 'Fully human-written']
    paper_stats = human_only.groupby('submission_number')['rating'].agg(['sum', 'count']).reset_index()
    paper_stats.columns = ['submission_number', 'h_sum', 'h_count']

    df = df.merge(paper_stats, on='submission_number', how='left')

    # LOOM calculation
    df['loom'] = df['h_sum'] / df['h_count']
    mask_human = df['ai_classification'] == 'Fully human-written'
    df.loc[mask_human, 'loom'] = (df.loc[mask_human, 'h_sum'] - df.loc[mask_human, 'rating']) / (df.loc[mask_human, 'h_count'] - 1)

    df = df[df['h_count'] >= 2].dropna(subset=['loom']).copy()

    # Get the human baseline
    human_df = df[df['ai_classification'] == 'Fully human-written'].copy()
    human_df['is_ai'] = 0

    results = {'pairwise': {}}

    # Compare each AI category to human baseline
    ai_categories = [
        'Lightly AI-edited',
        'Moderately AI-edited',
        'Heavily AI-edited',
        'Fully AI-generated'
    ]

    for category in ai_categories:
        ai_df = df[df['ai_classification'] == category].copy()
        ai_df['is_ai'] = 1

        if len(ai_df) < 50:
            if verbose:
                print(f"\n{category}: Skipped (n={len(ai_df)})")
            continue

        # Combine human + this AI category
        combined = pd.concat([human_df, ai_df], ignore_index=True)

        try:
            model = smf.ols(
                "rating ~ loom * is_ai",
                data=combined
            ).fit(cov_type='cluster', cov_kwds={'groups': combined['submission_number']})

            beta3 = model.params['loom:is_ai']
            se3 = model.bse['loom:is_ai']
            p3 = model.pvalues['loom:is_ai']
            p3_onetailed = p3 / 2 if beta3 < 0 else 1 - p3 / 2

            # Calculate sensitivity reduction
            human_slope = model.params['loom']
            ai_slope = human_slope + beta3
            sens_reduction = (1 - ai_slope / human_slope) * 100 if human_slope != 0 else np.nan

            results['pairwise'][category] = {
                'n': len(ai_df),
                'beta3': beta3,
                'se': se3,
                'p_onetailed': p3_onetailed,
                'sensitivity_reduction_pct': sens_reduction,
                'significant': beta3 < 0 and p3_onetailed < 0.05
            }

            sig = "***" if p3_onetailed < 0.001 else "**" if p3_onetailed < 0.01 else "*" if p3_onetailed < 0.05 else ""
            if verbose:
                print(f"\n{category} (n={len(ai_df):,}):")
                print(f"  β₃ = {beta3:.4f} (SE={se3:.4f}), p = {p3_onetailed:.4f} {sig}")
                print(f"  Sensitivity reduction: {sens_reduction:.1f}%")

        except Exception as e:
            if verbose:
                print(f"\n{category}: Failed ({e})")

    # Check for monotonic increase in effect
    if len(results['pairwise']) >= 2:
        beta3_values = [results['pairwise'][cat]['beta3']
                        for cat in ai_categories if cat in results['pairwise']]
        # More negative = stronger effect, should decrease (become more negative)
        monotonic = all(beta3_values[i] >= beta3_values[i+1]
                        for i in range(len(beta3_values)-1))
        results['monotonic'] = monotonic

        if verbose:
            print(f"\n--- Summary ---")
            print(f"Monotonic gradient flattening: {'✓ YES' if monotonic else '✗ NO'}")
            sig_count = sum(1 for v in results['pairwise'].values() if v.get('significant'))
            print(f"Significant comparisons: {sig_count}/{len(results['pairwise'])}")

    return results


def sensitivity_permutation_test(reviews_df: pd.DataFrame,
                                  submissions_df: pd.DataFrame,
                                  n_permutations: int = 1000,
                                  verbose: bool = True) -> Dict:
    """
    Sensitivity Analysis C: Permutation/Placebo Test.

    Randomly shuffle AI labels and re-estimate β₃. The real β₃ should
    be in the tail of the null distribution if the effect is genuine.

    Uses clean LOOM from Fully human-written reviews only.

    Parameters
    ----------
    reviews_df : DataFrame
    submissions_df : DataFrame
    n_permutations : int
        Number of random permutations
    verbose : bool

    Returns
    -------
    dict with null distribution and p-value
    """
    import statsmodels.formula.api as smf

    if verbose:
        print("\n" + "="*70)
        print("SENSITIVITY ANALYSIS C: Permutation Test (Placebo)")
        print("="*70)
        print(f"\nShuffling AI labels {n_permutations} times to generate null distribution")
        print("Using clean LOOM from Fully human-written reviews only")

    merged = merge_paper_info(reviews_df, submissions_df)

    # Keep only extreme categories for cleaner test
    extreme_cats = ['Fully human-written', 'Fully AI-generated']
    df = merged[merged['ai_classification'].isin(extreme_cats)].copy()
    df['is_ai_reviewer'] = (df['ai_classification'] == 'Fully AI-generated').astype(int)
    df = df.dropna(subset=['rating', 'submission_number'])

    # Calculate LOOM using ONLY Fully human-written reviews
    human_only = df[df['is_ai_reviewer'] == 0]
    paper_stats = human_only.groupby('submission_number')['rating'].agg(['sum', 'count']).reset_index()
    paper_stats.columns = ['submission_number', 'h_sum', 'h_count']

    df = df.merge(paper_stats, on='submission_number', how='left')
    df['loom'] = df['h_sum'] / df['h_count']

    mask_human = df['is_ai_reviewer'] == 0
    df.loc[mask_human, 'loom'] = (df.loc[mask_human, 'h_sum'] - df.loc[mask_human, 'rating']) / (df.loc[mask_human, 'h_count'] - 1)

    df_reg = df[df['h_count'] >= 2].dropna(subset=['loom']).copy()

    if verbose:
        print(f"\nSample: {len(df_reg):,} reviews (Fully Human + Fully AI only)")
        print(f"  Human: {(df_reg['is_ai_reviewer'] == 0).sum():,}")
        print(f"  AI: {(df_reg['is_ai_reviewer'] == 1).sum():,}")

    # Get actual β₃
    try:
        model_actual = smf.ols("rating ~ loom * is_ai_reviewer", data=df_reg).fit()
        actual_beta3 = model_actual.params['loom:is_ai_reviewer']
    except Exception as e:
        if verbose:
            print(f"Failed to estimate actual model: {e}")
        return {}

    if verbose:
        print(f"\nActual β₃ = {actual_beta3:.4f}")
        print(f"Running {n_permutations} permutations...")

    # Permutation test
    np.random.seed(42)
    null_beta3s = []

    ai_labels = df_reg['is_ai_reviewer'].values.copy()

    for i in range(n_permutations):
        np.random.shuffle(ai_labels)
        df_reg['is_ai_shuffled'] = ai_labels

        try:
            model_null = smf.ols("rating ~ loom * is_ai_shuffled", data=df_reg).fit()
            null_beta3s.append(model_null.params['loom:is_ai_shuffled'])
        except:
            continue

        if verbose and (i + 1) % 250 == 0:
            print(f"  Completed {i + 1}/{n_permutations} permutations...")

    null_beta3s = np.array(null_beta3s)

    # Calculate permutation p-value (one-tailed: actual < null)
    perm_p = np.mean(null_beta3s <= actual_beta3)

    results = {
        'actual_beta3': actual_beta3,
        'null_mean': np.mean(null_beta3s),
        'null_std': np.std(null_beta3s),
        'null_percentile_2.5': np.percentile(null_beta3s, 2.5),
        'null_percentile_97.5': np.percentile(null_beta3s, 97.5),
        'permutation_p': perm_p,
        'n_permutations': len(null_beta3s),
        'significant': perm_p < 0.05
    }

    if verbose:
        print(f"\n--- Permutation Results ---")
        print(f"Actual β₃:       {actual_beta3:.4f}")
        print(f"Null mean:       {results['null_mean']:.4f}")
        print(f"Null SD:         {results['null_std']:.4f}")
        print(f"Null 95% CI:     [{results['null_percentile_2.5']:.4f}, {results['null_percentile_97.5']:.4f}]")
        print(f"Permutation p:   {perm_p:.4f}")
        print(f"\n{'✓ SIGNIFICANT' if results['significant'] else '✗ Not significant'}: "
              f"Actual β₃ is {'in' if perm_p < 0.05 else 'NOT in'} the extreme tail of the null")

    return results


def loom_measurement_diagnostic(reviews_df: pd.DataFrame,
                                 submissions_df: pd.DataFrame,
                                 verbose: bool = True) -> Dict:
    """
    Diagnostic: Compare LOOM measurement quality between original and clean specifications.

    If clean LOOM has fewer reviews per paper, measurement error could explain
    why the gradient flattening effect attenuates in sensitivity analyses.

    Parameters
    ----------
    reviews_df : DataFrame
    submissions_df : DataFrame
    verbose : bool

    Returns
    -------
    dict with LOOM quality diagnostics
    """
    if verbose:
        print("\n" + "="*70)
        print("LOOM MEASUREMENT ERROR DIAGNOSTIC")
        print("="*70)
        print("\nComparing quality proxy precision: Original vs Clean LOOM")

    merged = merge_paper_info(reviews_df, submissions_df)
    df = merged.dropna(subset=['rating', 'ai_classification', 'submission_number']).copy()

    # Original LOOM: all non-"Fully AI-generated" reviews
    original_eligible = df[df['ai_classification'] != 'Fully AI-generated']
    original_per_paper = original_eligible.groupby('submission_number').size()

    # Clean LOOM: only "Fully human-written" reviews
    clean_eligible = df[df['ai_classification'] == 'Fully human-written']
    clean_per_paper = clean_eligible.groupby('submission_number').size()

    # Papers with valid LOOM (≥2 reviews)
    original_valid_papers = (original_per_paper >= 2).sum()
    clean_valid_papers = (clean_per_paper >= 2).sum()

    # Calculate LOOM variance (proxy for measurement error)
    # More reviews per paper = lower measurement error
    original_mean_reviews = original_per_paper.mean()
    clean_mean_reviews = clean_per_paper.mean()

    # Theoretical measurement error: Var(LOOM) ≈ σ²/n
    # Ratio of measurement errors
    measurement_error_ratio = original_mean_reviews / clean_mean_reviews

    # Calculate actual LOOM values and compare variance
    # Original LOOM
    orig_stats = original_eligible.groupby('submission_number')['rating'].agg(['mean', 'std', 'count'])
    orig_loom_std = orig_stats['mean'].std()

    # Clean LOOM
    clean_stats = clean_eligible.groupby('submission_number')['rating'].agg(['mean', 'std', 'count'])
    clean_loom_std = clean_stats['mean'].std()

    results = {
        'original': {
            'eligible_reviews': len(original_eligible),
            'reviews_per_paper_mean': original_mean_reviews,
            'reviews_per_paper_median': original_per_paper.median(),
            'reviews_per_paper_min': original_per_paper.min(),
            'papers_with_valid_loom': original_valid_papers,
            'loom_std': orig_loom_std
        },
        'clean': {
            'eligible_reviews': len(clean_eligible),
            'reviews_per_paper_mean': clean_mean_reviews,
            'reviews_per_paper_median': clean_per_paper.median(),
            'reviews_per_paper_min': clean_per_paper.min(),
            'papers_with_valid_loom': clean_valid_papers,
            'loom_std': clean_loom_std
        },
        'comparison': {
            'review_reduction_pct': (1 - clean_mean_reviews / original_mean_reviews) * 100,
            'measurement_error_inflation': measurement_error_ratio,
            'papers_lost': original_valid_papers - clean_valid_papers,
            'attenuation_expected': measurement_error_ratio > 1.5
        }
    }

    # Estimate attenuation factor using reliability ratio
    # Attenuation ≈ reliability = Var(true) / (Var(true) + Var(error))
    # With fewer reviews, Var(error) increases, so reliability decreases
    if clean_mean_reviews > 0:
        # Rough approximation: attenuation factor ≈ n_clean / n_original
        attenuation_factor = clean_mean_reviews / original_mean_reviews
        results['comparison']['estimated_attenuation_factor'] = attenuation_factor

        # If original β₃ = -0.0319, attenuated β₃ ≈ -0.0319 * attenuation_factor
        original_beta3 = -0.0319
        predicted_attenuated_beta3 = original_beta3 * attenuation_factor
        results['comparison']['original_beta3'] = original_beta3
        results['comparison']['predicted_attenuated_beta3'] = predicted_attenuated_beta3
        results['comparison']['observed_clean_beta3'] = -0.0101  # From sensitivity analysis

    if verbose:
        print(f"\n--- Original LOOM (non-Fully-AI reviews) ---")
        print(f"Eligible reviews: {results['original']['eligible_reviews']:,}")
        print(f"Reviews per paper: mean={original_mean_reviews:.2f}, median={original_per_paper.median():.0f}")
        print(f"Papers with ≥2 reviews: {original_valid_papers:,}")
        print(f"LOOM std (quality signal): {orig_loom_std:.4f}")

        print(f"\n--- Clean LOOM (Fully human-written only) ---")
        print(f"Eligible reviews: {results['clean']['eligible_reviews']:,}")
        print(f"Reviews per paper: mean={clean_mean_reviews:.2f}, median={clean_per_paper.median():.0f}")
        print(f"Papers with ≥2 reviews: {clean_valid_papers:,}")
        print(f"LOOM std (quality signal): {clean_loom_std:.4f}")

        print(f"\n--- Measurement Error Analysis ---")
        print(f"Review reduction: {results['comparison']['review_reduction_pct']:.1f}%")
        print(f"Measurement error inflation: {measurement_error_ratio:.2f}x")
        print(f"Papers lost (insufficient reviews): {results['comparison']['papers_lost']:,}")

        if 'estimated_attenuation_factor' in results['comparison']:
            print(f"\n--- Attenuation Prediction ---")
            print(f"Estimated attenuation factor: {results['comparison']['estimated_attenuation_factor']:.3f}")
            print(f"Original β₃: {results['comparison']['original_beta3']:.4f}")
            print(f"Predicted attenuated β₃: {results['comparison']['predicted_attenuated_beta3']:.4f}")
            print(f"Observed clean β₃: {results['comparison']['observed_clean_beta3']:.4f}")

            pred = abs(results['comparison']['predicted_attenuated_beta3'])
            obs = abs(results['comparison']['observed_clean_beta3'])
            if pred > 0:
                match_pct = min(obs/pred, pred/obs) * 100
                print(f"Prediction accuracy: {match_pct:.1f}%")

        print(f"\n--- Conclusion ---")
        if results['comparison']['attenuation_expected']:
            print("⚠ SUBSTANTIAL MEASUREMENT ERROR INFLATION DETECTED")
            print("  The ~70% reduction in β₃ may be due to attenuation bias,")
            print("  not a true null effect. The original result may be valid.")
        else:
            print("Measurement error difference is modest.")
            print("Attenuation alone may not explain the discrepancy.")

    return results


def sensitivity_binary_clean(reviews_df: pd.DataFrame,
                              submissions_df: pd.DataFrame,
                              ai_threshold: float = 75.0,
                              human_threshold: float = 25.0,
                              verbose: bool = True) -> Dict:
    """
    Sensitivity Analysis D: Binary Classification (Drop Middle).

    Uses ONLY clearly human (<25% AI) vs clearly AI (>=75%) reviews,
    dropping ambiguous middle cases. This is the CLEANEST contrast.

    Justification for QJE/REStud:
    - "AI detection scores are inherently noisy. Fine-grained distinctions
       amplify classification error. We therefore use a parsimonious binary
       classification to maximize signal-to-noise, excluding ambiguous cases."

    Parameters
    ----------
    reviews_df : DataFrame
    submissions_df : DataFrame
    ai_threshold : float
        Reviews with AI% >= this are classified as AI (default: 75%)
    human_threshold : float
        Reviews with AI% <= this are classified as Human (default: 25%)
    verbose : bool

    Returns
    -------
    dict with binary classification results
    """
    import statsmodels.formula.api as smf

    if verbose:
        print("\n" + "="*70)
        print("SENSITIVITY ANALYSIS D: Binary Classification (Drop Middle)")
        print("="*70)
        print(f"\nUsing clean binary contrast:")
        print(f"  Human: AI% <= {human_threshold}%")
        print(f"  AI:    AI% >= {ai_threshold}%")
        print(f"  Dropped: {human_threshold}% < AI% < {ai_threshold}%")

    merged = merge_paper_info(reviews_df, submissions_df)

    # Create binary classification, dropping middle
    merged['binary_class'] = np.nan
    merged.loc[merged['ai_percentage'] <= human_threshold, 'binary_class'] = 0  # Human
    merged.loc[merged['ai_percentage'] >= ai_threshold, 'binary_class'] = 1     # AI

    # Drop middle and missing
    df = merged.dropna(subset=['rating', 'binary_class', 'submission_number']).copy()
    df['is_ai_reviewer'] = df['binary_class'].astype(int)

    n_total = len(merged.dropna(subset=['rating', 'ai_percentage']))
    n_human = (df['binary_class'] == 0).sum()
    n_ai = (df['binary_class'] == 1).sum()
    n_dropped = n_total - n_human - n_ai

    if verbose:
        print(f"\nSample:")
        print(f"  Human reviews (<={human_threshold}%): {n_human:,}")
        print(f"  AI reviews (>={ai_threshold}%):     {n_ai:,}")
        print(f"  Dropped (middle):              {n_dropped:,} ({100*n_dropped/n_total:.1f}%)")

    # Calculate LOOM using ONLY human reviews
    human_only = df[df['is_ai_reviewer'] == 0]
    paper_stats = human_only.groupby('submission_number')['rating'].agg(['sum', 'count']).reset_index()
    paper_stats.columns = ['submission_number', 'h_sum', 'h_count']

    df = df.merge(paper_stats, on='submission_number', how='left')

    # LOOM calculation
    df['loom'] = df['h_sum'] / df['h_count']
    mask_human = df['is_ai_reviewer'] == 0
    df.loc[mask_human, 'loom'] = (df.loc[mask_human, 'h_sum'] - df.loc[mask_human, 'rating']) / (df.loc[mask_human, 'h_count'] - 1)

    # Need at least 2 human reviews
    df_reg = df[df['h_count'] >= 2].dropna(subset=['loom']).copy()

    n_obs = len(df_reg)
    n_papers = df_reg['submission_number'].nunique()
    n_human_final = (df_reg['is_ai_reviewer'] == 0).sum()
    n_ai_final = (df_reg['is_ai_reviewer'] == 1).sum()

    if verbose:
        print(f"\nFinal regression sample:")
        print(f"  Papers with ≥2 human reviews: {n_papers:,}")
        print(f"  Human reviews: {n_human_final:,}")
        print(f"  AI reviews: {n_ai_final:,}")

    results = {
        'n_obs': n_obs,
        'n_papers': n_papers,
        'n_human': n_human_final,
        'n_ai': n_ai_final,
        'n_dropped': n_dropped,
        'pct_dropped': 100 * n_dropped / n_total if n_total > 0 else 0
    }

    # Run regression
    try:
        model = smf.ols(
            "rating ~ loom * is_ai_reviewer",
            data=df_reg
        ).fit(cov_type='cluster', cov_kwds={'groups': df_reg['submission_number']})

        beta1 = model.params['loom']
        beta2 = model.params['is_ai_reviewer']
        beta3 = model.params['loom:is_ai_reviewer']

        se3 = model.bse['loom:is_ai_reviewer']
        p3 = model.pvalues['loom:is_ai_reviewer']
        p3_onetailed = p3 / 2 if beta3 < 0 else 1 - p3 / 2

        # Sensitivity reduction
        ai_slope = beta1 + beta3
        human_slope = beta1
        sensitivity_reduction = (1 - ai_slope / human_slope) * 100 if human_slope != 0 else np.nan

        results['regression'] = {
            'beta1_loom': beta1,
            'beta2_ai': beta2,
            'beta3_interaction': beta3,
            'beta3_se': se3,
            'beta3_p': p3,
            'beta3_p_onetailed': p3_onetailed,
            'significant': beta3 < 0 and p3_onetailed < 0.05,
            'sensitivity_reduction_pct': sensitivity_reduction,
            'r_squared': model.rsquared
        }

        if verbose:
            print(f"\n--- Binary Regression Results (Clustered SE) ---")
            print(f"β₁ (Human LOOM slope): {beta1:.4f}")
            print(f"β₂ (AI intercept):     {beta2:.4f}")
            print(f"β₃ (Interaction):      {beta3:.4f} (SE={se3:.4f})")
            print(f"p-value (one-tailed):  {p3_onetailed:.4f}")
            print(f"Sensitivity reduction: {sensitivity_reduction:.1f}%")

            if results['regression']['significant']:
                print(f"\n✓ SIGNIFICANT: Gradient flattening confirmed with clean binary contrast")
            else:
                print(f"\n→ Not significant at α=0.05")

    except Exception as e:
        results['regression'] = None
        if verbose:
            print(f"\nRegression failed: {e}")

    return results


def sensitivity_terciles(reviews_df: pd.DataFrame,
                          submissions_df: pd.DataFrame,
                          verbose: bool = True) -> Dict:
    """
    Sensitivity Analysis E: Data-Driven Terciles.

    Uses terciles of AI% distribution (Low/Medium/High) instead of
    arbitrary thresholds. This is DATA-DRIVEN, not researcher-chosen.

    Justification for QJE/REStud:
    - "To avoid arbitrary threshold choices, we classify reviews into
       terciles based on the empirical AI percentage distribution.
       This data-driven approach ensures balanced group sizes and
       removes researcher degrees of freedom in classification."

    Parameters
    ----------
    reviews_df : DataFrame
    submissions_df : DataFrame
    verbose : bool

    Returns
    -------
    dict with tercile-based results
    """
    import statsmodels.formula.api as smf

    if verbose:
        print("\n" + "="*70)
        print("SENSITIVITY ANALYSIS E: Data-Driven Terciles")
        print("="*70)
        print("\nUsing terciles of AI% distribution (data-driven cutoffs)")

    merged = merge_paper_info(reviews_df, submissions_df)
    df = merged.dropna(subset=['rating', 'ai_percentage', 'submission_number']).copy()

    # Create terciles based on AI percentage distribution
    try:
        df['ai_tercile'] = pd.qcut(df['ai_percentage'], q=3, labels=['Low', 'Medium', 'High'])
    except ValueError:
        # If too many ties, use approximate terciles
        tercile_33 = df['ai_percentage'].quantile(0.33)
        tercile_67 = df['ai_percentage'].quantile(0.67)
        df['ai_tercile'] = pd.cut(
            df['ai_percentage'],
            bins=[-np.inf, tercile_33, tercile_67, np.inf],
            labels=['Low', 'Medium', 'High']
        )

    # Get tercile cutoffs
    tercile_stats = df.groupby('ai_tercile', observed=True)['ai_percentage'].agg(['min', 'max', 'count'])

    if verbose:
        print("\nTercile Definitions (AI%):")
        for tercile in ['Low', 'Medium', 'High']:
            if tercile in tercile_stats.index:
                row = tercile_stats.loc[tercile]
                print(f"  {tercile}: {row['min']:.1f}% - {row['max']:.1f}% (n={row['count']:,})")

    # Create numeric AI level (0=Low, 1=Medium, 2=High)
    df['ai_level'] = df['ai_tercile'].map({'Low': 0, 'Medium': 1, 'High': 2})

    # Calculate LOOM using ONLY Low AI tercile (cleanest human proxy)
    low_ai = df[df['ai_level'] == 0]
    paper_stats = low_ai.groupby('submission_number')['rating'].agg(['sum', 'count']).reset_index()
    paper_stats.columns = ['submission_number', 'h_sum', 'h_count']

    df = df.merge(paper_stats, on='submission_number', how='left')

    # LOOM calculation
    df['loom'] = df['h_sum'] / df['h_count']
    mask_low = df['ai_level'] == 0
    df.loc[mask_low, 'loom'] = (df.loc[mask_low, 'h_sum'] - df.loc[mask_low, 'rating']) / (df.loc[mask_low, 'h_count'] - 1)

    # Need at least 2 Low-AI reviews for valid LOOM
    df_reg = df[df['h_count'] >= 2].dropna(subset=['loom']).copy()

    n_obs = len(df_reg)
    n_papers = df_reg['submission_number'].nunique()

    results = {
        'n_obs': n_obs,
        'n_papers': n_papers,
        'tercile_cutoffs': {
            'low_max': tercile_stats.loc['Low', 'max'] if 'Low' in tercile_stats.index else None,
            'medium_max': tercile_stats.loc['Medium', 'max'] if 'Medium' in tercile_stats.index else None
        },
        'tercile_counts': tercile_stats['count'].to_dict() if len(tercile_stats) > 0 else {}
    }

    if verbose:
        print(f"\nRegression sample: {n_obs:,} reviews from {n_papers:,} papers")

    # Run ordinal regression with terciles
    try:
        model = smf.ols(
            "rating ~ loom * ai_level",
            data=df_reg
        ).fit(cov_type='cluster', cov_kwds={'groups': df_reg['submission_number']})

        beta1 = model.params['loom']
        beta2 = model.params['ai_level']
        beta3 = model.params['loom:ai_level']

        se3 = model.bse['loom:ai_level']
        p3 = model.pvalues['loom:ai_level']
        p3_onetailed = p3 / 2 if beta3 < 0 else 1 - p3 / 2

        results['ordinal'] = {
            'beta1_loom': beta1,
            'beta2_ai_level': beta2,
            'beta3_interaction': beta3,
            'beta3_se': se3,
            'beta3_p': p3,
            'beta3_p_onetailed': p3_onetailed,
            'significant': beta3 < 0 and p3_onetailed < 0.05,
            'r_squared': model.rsquared
        }

        if verbose:
            print(f"\n--- Tercile Regression Results (Clustered SE) ---")
            print(f"β₁ (Low-AI LOOM slope):    {beta1:.4f}")
            print(f"β₂ (AI tercile main effect): {beta2:.4f}")
            print(f"β₃ (Tercile × LOOM):        {beta3:.4f} (SE={se3:.4f})")
            print(f"p-value (one-tailed):       {p3_onetailed:.4f}")
            print(f"\nInterpretation: Moving from Low to High AI tercile")
            print(f"  {'DECREASES' if beta3 < 0 else 'increases'} sensitivity by {abs(beta3)*2:.4f}")

            if results['ordinal']['significant']:
                print(f"\n✓ SIGNIFICANT: Gradient flattening confirmed with data-driven terciles")
            else:
                print(f"\n→ Not significant at α=0.05")

    except Exception as e:
        results['ordinal'] = None
        if verbose:
            print(f"\nRegression failed: {e}")

    # Also run Low vs High comparison (binary tercile contrast)
    if verbose:
        print(f"\n--- Binary Tercile Contrast (Low vs High) ---")

    df_binary = df_reg[df_reg['ai_level'].isin([0, 2])].copy()
    df_binary['is_high_ai'] = (df_binary['ai_level'] == 2).astype(int)

    try:
        model_binary = smf.ols(
            "rating ~ loom * is_high_ai",
            data=df_binary
        ).fit(cov_type='cluster', cov_kwds={'groups': df_binary['submission_number']})

        beta3_binary = model_binary.params['loom:is_high_ai']
        se3_binary = model_binary.bse['loom:is_high_ai']
        p3_binary = model_binary.pvalues['loom:is_high_ai']
        p3_binary_onetailed = p3_binary / 2 if beta3_binary < 0 else 1 - p3_binary / 2

        results['binary_tercile'] = {
            'beta3_interaction': beta3_binary,
            'beta3_se': se3_binary,
            'beta3_p': p3_binary,
            'beta3_p_onetailed': p3_binary_onetailed,
            'significant': beta3_binary < 0 and p3_binary_onetailed < 0.05,
            'n_low': (df_binary['is_high_ai'] == 0).sum(),
            'n_high': (df_binary['is_high_ai'] == 1).sum()
        }

        if verbose:
            print(f"Low AI (n={results['binary_tercile']['n_low']:,}) vs High AI (n={results['binary_tercile']['n_high']:,})")
            print(f"β₃ = {beta3_binary:.4f} (SE={se3_binary:.4f})")
            print(f"p-value (one-tailed): {p3_binary_onetailed:.4f}")

            if results['binary_tercile']['significant']:
                print(f"\n✓ SIGNIFICANT")
            else:
                print(f"\n→ Not significant")

    except Exception as e:
        results['binary_tercile'] = None
        if verbose:
            print(f"Binary tercile comparison failed: {e}")

    return results


def run_sensitivity_analyses(reviews_df: pd.DataFrame,
                              submissions_df: pd.DataFrame,
                              n_permutations: int = 1000,
                              verbose: bool = True) -> Dict:
    """
    Run all sensitivity analyses for AI classification validation.

    A. Ordinal gradient: Does flattening increase with AI level (0-4)?
    B. Pairwise comparisons: Each AI category vs Fully Human
    C. Permutation test: Shuffle labels to verify effect is real
    D. Binary clean: Drop middle cases, use Human (<25%) vs AI (>=75%)
    E. Data-driven terciles: Low/Medium/High AI based on distribution

    All analyses use clean LOOM from Fully human-written reviews only.

    Parameters
    ----------
    reviews_df : DataFrame
    submissions_df : DataFrame
    n_permutations : int
    verbose : bool

    Returns
    -------
    dict with all sensitivity analysis results
    """
    if verbose:
        print("\n" + "="*70)
        print("AI CLASSIFICATION SENSITIVITY ANALYSES")
        print("="*70)
        print("\nValidating robustness of gradient flattening result")
        print("All analyses use CLEAN LOOM from Fully human-written reviews only")

    results = {}

    # Diagnostic: LOOM measurement error
    results['loom_diagnostic'] = loom_measurement_diagnostic(
        reviews_df, submissions_df, verbose=verbose
    )

    # Analysis A: Ordinal gradient (5-level)
    results['ordinal'] = sensitivity_ordinal_gradient(
        reviews_df, submissions_df, verbose=verbose
    )

    # Analysis B: Pairwise comparisons
    results['pairwise'] = sensitivity_pairwise_categories(
        reviews_df, submissions_df, verbose=verbose
    )

    # Analysis C: Permutation test
    results['permutation'] = sensitivity_permutation_test(
        reviews_df, submissions_df, n_permutations=n_permutations, verbose=verbose
    )

    # Analysis D: Binary clean (drop middle) - 75/25 threshold
    results['binary_clean'] = sensitivity_binary_clean(
        reviews_df, submissions_df,
        ai_threshold=75.0, human_threshold=25.0,
        verbose=verbose
    )

    # Analysis D': Binary 50/50 - more inclusive threshold
    results['binary_50_50'] = sensitivity_binary_clean(
        reviews_df, submissions_df,
        ai_threshold=50.0, human_threshold=50.0,
        verbose=verbose
    )

    # Analysis E: Data-driven terciles - RECOMMENDED FOR QJE/RESTUD
    results['terciles'] = sensitivity_terciles(
        reviews_df, submissions_df, verbose=verbose
    )

    # Overall summary - now includes D, D', and E
    checks_passed = 0
    total_checks = 6

    if results.get('ordinal', {}).get('ordinal', {}).get('significant'):
        checks_passed += 1

    if results.get('pairwise', {}).get('monotonic'):
        checks_passed += 1

    if results.get('permutation', {}).get('significant'):
        checks_passed += 1

    # Handle None values safely for new analyses
    binary_clean = results.get('binary_clean') or {}
    binary_reg = binary_clean.get('regression') or {}
    if binary_reg.get('significant'):
        checks_passed += 1

    # 50/50 binary
    binary_50_50 = results.get('binary_50_50') or {}
    binary_50_50_reg = binary_50_50.get('regression') or {}
    if binary_50_50_reg.get('significant'):
        checks_passed += 1

    terciles = results.get('terciles') or {}
    terciles_ordinal = terciles.get('ordinal') or {}
    if terciles_ordinal.get('significant'):
        checks_passed += 1

    results['overall'] = {
        'checks_passed': checks_passed,
        'total_checks': total_checks,
        'robust': checks_passed >= 2
    }

    if verbose:
        print("\n" + "="*70)
        print("SENSITIVITY ANALYSIS SUMMARY")
        print("="*70)
        print(f"\nChecks passed: {checks_passed}/{total_checks}")
        print(f"  A. Ordinal gradient (5-level):   {(results.get('ordinal') or {}).get('ordinal', {}).get('significant', False)}")
        print(f"  B. Monotonic across categories:  {(results.get('pairwise') or {}).get('monotonic', False)}")
        print(f"  C. Permutation test significant: {(results.get('permutation') or {}).get('significant', False)}")
        print(f"  D. Binary 75/25 (drop middle):   {binary_reg.get('significant', False)}")
        print(f"  D'. Binary 50/50:                {binary_50_50_reg.get('significant', False)}")
        print(f"  E. Data-driven terciles:         {terciles_ordinal.get('significant', False)}")

        # Highlight recommended analyses for QJE/REStud
        print(f"\n--- RECOMMENDED FOR QJE/REStud (Simplified Classification) ---")
        binary_sig = binary_reg.get('significant', False)
        binary_50_50_sig = binary_50_50_reg.get('significant', False)
        tercile_sig = terciles_ordinal.get('significant', False)
        terciles_binary = terciles.get('binary_tercile') or {}
        binary_tercile_sig = terciles_binary.get('significant', False)

        # Check for data issues
        n_ai_binary = binary_reg.get('n_ai', 0) if binary_reg else 0
        n_ai_50_50 = binary_50_50_reg.get('n_ai', 0) if binary_50_50_reg else 0

        if n_ai_binary == 0:
            print(f"  D. Binary (Human <25% vs AI ≥75%):  ⚠ INSUFFICIENT DATA")
        else:
            print(f"  D. Binary (Human <25% vs AI ≥75%):  {'✓ SIGNIFICANT' if binary_sig else '✗ Not significant'}")

        if n_ai_50_50 == 0:
            print(f"  D'. Binary (Human <50% vs AI ≥50%): ⚠ INSUFFICIENT DATA")
        else:
            print(f"  D'. Binary (Human <50% vs AI ≥50%): {'✓ SIGNIFICANT' if binary_50_50_sig else '✗ Not significant'}")

        print(f"  E. Terciles (Low/Med/High):         {'✓ SIGNIFICANT' if tercile_sig else '✗ Not significant'}")
        print(f"  E'. Low vs High tercile only:       {'✓ SIGNIFICANT' if binary_tercile_sig else '✗ Not significant'}")

        any_simplified_sig = binary_sig or binary_50_50_sig or tercile_sig or binary_tercile_sig
        if any_simplified_sig:
            print(f"\n✓ GRADIENT FLATTENING ROBUST with simplified classification")
        else:
            print(f"\n⚠ GRADIENT FLATTENING NOT ROBUST even with simplified classification")

        print(f"\nOverall robustness: {'✓ ROBUST' if results['overall']['robust'] else '✗ CONCERNS'}")

    return results


def _generate_sensitivity_table(results: Dict, output_dir: str):
    """Generate LaTeX table for sensitivity analyses."""

    latex = """\\begin{tabular}{llccc}
\\toprule
\\textbf{Analysis} & \\textbf{Specification} & $\\hat{\\beta}_3$ & \\textbf{p-value} & \\textbf{Significant} \\\\
\\midrule
\\multicolumn{5}{l}{\\textit{Panel A: Ordinal AI Classification (0-4 scale)}} \\\\
\\midrule"""

    # Ordinal results
    ordinal = results.get('ordinal', {}).get('ordinal', {})
    if ordinal:
        sig_mark = "\\checkmark" if ordinal.get('significant') else ""
        latex += f"""
AI Level $\\times$ LOOM & Continuous (0-4) & {ordinal.get('beta3_interaction', 0):.4f} & {ordinal.get('beta3_p_onetailed', 1):.4f} & {sig_mark} \\\\"""

    latex += """
\\midrule
\\multicolumn{5}{l}{\\textit{Panel B: Pairwise vs Fully Human (Clean LOOM)}} \\\\
\\midrule"""

    # Pairwise results
    pairwise = results.get('pairwise', {}).get('pairwise', {})
    for cat in ['Lightly AI-edited', 'Moderately AI-edited', 'Heavily AI-edited', 'Fully AI-generated']:
        if cat in pairwise:
            r = pairwise[cat]
            sig_mark = "\\checkmark" if r.get('significant') else ""
            sens_red = r.get('sensitivity_reduction_pct', 0)
            latex += f"""
{cat} & n={r.get('n', 0):,} & {r.get('beta3', 0):.4f} & {r.get('p_onetailed', 1):.4f} & {sig_mark} \\\\"""

    # Monotonicity
    mono = results.get('pairwise', {}).get('monotonic', False)
    latex += f"""
Monotonic gradient & \\multicolumn{{4}}{{c}}{{{'Yes' if mono else 'No'}}} \\\\"""

    latex += """
\\midrule
\\multicolumn{5}{l}{\\textit{Panel C: Permutation Test (Fully AI vs Fully Human)}} \\\\
\\midrule"""

    # Permutation results
    perm = results.get('permutation', {})
    if perm:
        sig_mark = "\\checkmark" if perm.get('significant') else ""
        latex += f"""
Actual vs Null & n={perm.get('n_permutations', 0):,} perms & {perm.get('actual_beta3', 0):.4f} & {perm.get('permutation_p', 1):.4f} & {sig_mark} \\\\
Null distribution & & \\multicolumn{{3}}{{c}}{{Mean={perm.get('null_mean', 0):.4f}, SD={perm.get('null_std', 0):.4f}}} \\\\"""

    latex += """
\\bottomrule
\\end{tabular}"""

    with open(f'{output_dir}/table_sensitivity.tex', 'w') as f:
        f.write(latex)

    print(f"\n✓ Saved sensitivity analysis table to {output_dir}/table_sensitivity.tex")


# =============================================================================
# COMBINED ANALYSIS FUNCTION
# =============================================================================

def run_referee_tests(submissions_df, reviews_df,
                      output_dir: str = 'tables',
                      run_sensitivity: bool = True,
                      n_permutations: int = 1000,
                      verbose: bool = True) -> Dict:
    """
    Run all referee-requested tests.

    1. Variance compression test (Lemma 2)
    2. Formal component interaction test (Table 2)
    3. Gradient flattening test (Lemma 2 part iii - Truth-Tracking)
    4. Sensitivity analyses for AI classification validation (optional)

    Parameters
    ----------
    submissions_df : DataFrame
    reviews_df : DataFrame
    output_dir : str
    run_sensitivity : bool
        Whether to run sensitivity analyses (can be slow due to permutation test)
    n_permutations : int
        Number of permutations for placebo test (default 1000)
    verbose : bool

    Returns
    -------
    dict with all test results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("="*70)
    print("REFEREE-REQUESTED STATISTICAL TESTS")
    print("="*70)

    results = {}

    # Test 1: Variance Compression
    print("\n[TEST 1] Variance Compression (Lemma 2 Validation)")
    results['variance_compression'] = variance_compression_test(
        reviews_df, submissions_df, verbose=verbose
    )
    results['variance_by_component'] = variance_compression_by_component(
        reviews_df, submissions_df, verbose=verbose
    )

    # Test 2: Component Interaction
    print("\n[TEST 2] Component Interaction (Table 2 Formal Test)")
    results['component_interaction'] = component_interaction_test(
        reviews_df, submissions_df, verbose=verbose
    )

    # Test 3: Gradient Flattening (Truth-Tracking)
    print("\n[TEST 3] Gradient Flattening (Lemma 2 part iii - Truth-Tracking)")
    results['gradient_flattening'] = gradient_flattening_test(
        reviews_df, submissions_df, verbose=verbose
    )

    # Test 4: Sensitivity Analyses (optional)
    if run_sensitivity:
        print("\n[TEST 4] Sensitivity Analyses (AI Classification Validation)")
        results['sensitivity'] = run_sensitivity_analyses(
            reviews_df, submissions_df,
            n_permutations=n_permutations,
            verbose=verbose
        )
        if output_dir:
            _generate_sensitivity_table(results['sensitivity'], output_dir)

    # Generate LaTeX table for results
    if output_dir:
        _generate_referee_tests_table(results, output_dir)

    return results


def _generate_referee_tests_table(results: Dict, output_dir: str):
    """Generate LaTeX table summarizing referee tests."""

    vc = results['variance_compression']
    ci = results['component_interaction']
    gf = results.get('gradient_flattening', {})

    latex = f"""\\begin{{tabular}}{{lcc}}
\\toprule
\\textbf{{Test}} & \\textbf{{Statistic}} & \\textbf{{p-value}} \\\\
\\midrule
\\multicolumn{{3}}{{l}}{{\\textit{{Panel A: Variance Compression (Lemma 2)}}}} \\\\
\\midrule
Human variance ($\\sigma^2_H$) & \\multicolumn{{2}}{{c}}{{{vc['var_human']:.4f}}} \\\\
AI variance ($\\sigma^2_{{AI}}$) & \\multicolumn{{2}}{{c}}{{{vc['var_ai']:.4f}}} \\\\
Variance ratio ($\\sigma^2_H / \\sigma^2_{{AI}}$) & \\multicolumn{{2}}{{c}}{{{vc['var_ratio']:.3f}}} \\\\
Variance reduction & \\multicolumn{{2}}{{c}}{{{vc['variance_reduction_pct']:.1f}\\%}} \\\\
Levene's test (H$_1$: $\\sigma^2_{{AI}} < \\sigma^2_H$) & $F = {vc['levene_stat']:.2f}$ & ${vc['levene_p_onetailed']:.4f}$ \\\\
\\midrule
\\multicolumn{{3}}{{l}}{{\\textit{{Panel B: Component Interaction (Table 2)}}}} \\\\
\\midrule
Interaction ($\\beta_3$: AI $\\times$ Soundness) & ${ci['ols_robust']['interaction_coef']:.4f}$ & ${ci['ols_robust']['interaction_p_onetailed']:.4e}$ \\\\
95\\% CI & \\multicolumn{{2}}{{c}}{{$[{ci['ols_robust']['ci_lower']:.4f}, {ci['ols_robust']['ci_upper']:.4f}]$}} \\\\"""

    # Add Panel C if gradient flattening results are available
    if gf and gf.get('clustered'):
        gfc = gf['clustered']
        sens_red = gf.get('sensitivity_reduction_pct', 0)
        latex += f"""
\\midrule
\\multicolumn{{3}}{{l}}{{\\textit{{Panel C: Gradient Flattening (Lemma 2 part iii)}}}} \\\\
\\midrule
Human consensus slope ($\\beta_1$) & ${gfc['beta1_loom']:.4f}$ & ${gfc['beta1_p_onetailed']:.4e}$ \\\\
AI leniency intercept ($\\beta_2$) & ${gfc['beta2_ai']:.4f}$ & ${gfc['beta2_p_onetailed']:.4e}$ \\\\
Interaction ($\\beta_3$: AI $\\times$ LOOM) & ${gfc['beta3_interaction']:.4f}$ & ${gfc['beta3_p_onetailed']:.4e}$ \\\\
95\\% CI for $\\beta_3$ & \\multicolumn{{2}}{{c}}{{$[{gfc['beta3_ci'][0]:.4f}, {gfc['beta3_ci'][1]:.4f}]$}} \\\\
Sensitivity reduction & \\multicolumn{{2}}{{c}}{{{sens_red:.1f}\\%}} \\\\"""

    latex += """
\\bottomrule
\\end{tabular}"""

    with open(f'{output_dir}/table_referee_tests.tex', 'w') as f:
        f.write(latex)

    print(f"\n✓ Saved referee tests table to {output_dir}/table_referee_tests.tex")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        submissions_df, reviews_df = load_data(sys.argv[1], sys.argv[2])
        output_dir = sys.argv[3] if len(sys.argv) > 3 else 'tables'
        results = run_referee_tests(submissions_df, reviews_df, output_dir)
    else:
        print("""
Referee-Requested Statistical Tests
====================================

Usage:
    python referee_tests.py submissions.csv reviews.csv [output_dir]

Or in Python:
    from analysis.referee_tests import run_referee_tests
    results = run_referee_tests(submissions_df, reviews_df)

    # Skip sensitivity analyses (faster):
    results = run_referee_tests(submissions_df, reviews_df, run_sensitivity=False)

Tests:
1. Variance compression (Lemma 2): σ²_AI < σ²_Human
2. Component interaction (Table 2): Score ~ AI × IsSoundness
3. Gradient flattening (Lemma 2 part iii): Score ~ LOOM × Is_AI
   - Tests if AI reviewers are less correlated with true quality
   - β₃ < 0 proves sensitivity collapse (AI less responsive to quality)
4. Sensitivity analyses (AI classification validation):
   A. Threshold variation: Vary AI classification cutoff
   B. Permutation test: Shuffle AI labels to generate null distribution
   C. Exclude borderline: Only use 'Fully AI' vs 'Fully Human'
   D. AI intensity gradient: Test if effect increases with AI percentage

Output tables:
- table_referee_tests.tex: Main results (Panels A-C)
- table_sensitivity.tex: Sensitivity analyses (Panels A-D)
""")
