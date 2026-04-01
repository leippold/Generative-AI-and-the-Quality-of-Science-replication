"""
Reviewer-Side Diagnostics
=========================

Five diagnostics to stress-test the within-paper AI-reviewer effect:

1. Review-level controls in paper-FE regression (confidence, component scores)
2. Balanced paired-subsample check (exactly 1 AI + ≥1 human per paper)
3. Permutation inference (shuffle AI labels within paper)
4. Predictive-validity check (which reviewer type better predicts avg_rating)
5. Summary robustness table

Uses within-transformation (demeaning by paper) instead of explicit dummy
variables for paper fixed effects.  This is numerically equivalent to
C(submission_number) but scales to any number of papers without creating
a massive design matrix.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Optional
import warnings
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loading import load_data, merge_paper_info
from src.stats_utils import bootstrap_ci, cohens_d
from src.constants import N_PERMUTATIONS, RANDOM_SEED


# =============================================================================
# HELPERS
# =============================================================================

def _prepare_diagnostic_data(reviews_df: pd.DataFrame,
                              submissions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare review-level data with binary AI indicator and paper info.

    Returns a DataFrame restricted to papers that have BOTH human and AI
    reviews, with columns: submission_number, rating, confidence, soundness,
    presentation, contribution, reviewer_AI (0/1), ai_percentage.
    """
    merged = merge_paper_info(reviews_df, submissions_df)

    # Binary reviewer indicator (only fully human vs fully AI)
    reviewer_map = {
        'Fully human-written': 0,
        'Fully AI-generated': 1,
    }
    merged['reviewer_AI'] = merged['ai_classification'].map(reviewer_map)
    merged = merged.dropna(subset=['reviewer_AI', 'rating']).copy()
    merged['reviewer_AI'] = merged['reviewer_AI'].astype(int)

    # Keep only papers with BOTH reviewer types
    paper_types = merged.groupby('submission_number')['reviewer_AI'].nunique()
    both_papers = paper_types[paper_types == 2].index
    merged = merged[merged['submission_number'].isin(both_papers)].copy()

    return merged


def _demean(df: pd.DataFrame, cols: list, group: str = 'submission_number'):
    """
    Within-transform: subtract group means from each column.
    Equivalent to absorbing paper fixed effects.
    """
    out = df.copy()
    group_means = df.groupby(group)[cols].transform('mean')
    for c in cols:
        out[f'{c}_dm'] = df[c] - group_means[c]
    return out


def _fe_ols(df, y_col, x_cols, group='submission_number'):
    """
    OLS on demeaned data with clustered SEs — equivalent to paper-FE regression.

    Returns dict with coeff, se, pvalue for each x_col plus model metadata.
    """
    import statsmodels.api as sm

    all_cols = [y_col] + x_cols
    clean = df.dropna(subset=all_cols + [group]).copy()

    # Demean
    clean = _demean(clean, all_cols, group)

    y = clean[f'{y_col}_dm'].values
    X = clean[[f'{c}_dm' for c in x_cols]].values

    # OLS on demeaned data (no constant — absorbed by FE)
    model = sm.OLS(y, X).fit(
        cov_type='cluster',
        cov_kwds={'groups': clean[group].values}
    )

    results = {
        'n': len(clean),
        'n_groups': clean[group].nunique(),
        'model': model,
    }
    for i, col in enumerate(x_cols):
        results[col] = {
            'coeff': model.params[i],
            'se': model.bse[i],
            'pvalue': model.pvalues[i],
        }
    return results


def _fe_coeff(df, y_col='rating', x_col='reviewer_AI',
              group='submission_number'):
    """Quick helper: return just the FE coefficient for x_col."""
    res = _fe_ols(df, y_col, [x_col], group)
    return res[x_col]['coeff']


# =============================================================================
# DIAGNOSTIC 1: Paper-FE with review-level controls
# =============================================================================

def diagnostic_review_controls(merged: pd.DataFrame,
                                verbose: bool = True) -> Dict:
    """
    Paper-FE regression with and without review-level controls.

    Controls added: reviewer confidence, soundness, presentation, contribution.
    Uses within-transformation + clustered SEs at the paper level.
    """
    results = {}

    if verbose:
        print("\n" + "=" * 70)
        print("DIAGNOSTIC 1: Review-Level Controls in Paper-FE Regression")
        print("=" * 70)

    n_papers = merged['submission_number'].nunique()
    n_reviews = len(merged)
    if verbose:
        print(f"\nSample: {n_reviews} reviews across {n_papers} papers "
              f"(papers with both AI and human reviews)")

    # --- Baseline: paper FE only ---
    base_res = _fe_ols(merged, 'rating', ['reviewer_AI'])
    b = base_res['reviewer_AI']

    results['baseline'] = {
        'coeff': b['coeff'],
        'se': b['se'],
        'pvalue': b['pvalue'],
        'n': base_res['n'],
        'n_papers': base_res['n_groups'],
    }

    if verbose:
        r = results['baseline']
        print(f"\n  Baseline (paper FE only):")
        print(f"    AI Reviewer coeff = {r['coeff']:.4f} "
              f"(SE = {r['se']:.4f}, p = {r['pvalue']:.4e})")

    # --- With review-level controls ---
    controls = []
    if 'confidence' in merged.columns and merged['confidence'].notna().sum() > 50:
        controls.append('confidence')
    for col in ['soundness', 'presentation', 'contribution']:
        if col in merged.columns and merged[col].notna().sum() > 50:
            controls.append(col)

    if controls:
        ctrl_res = _fe_ols(merged, 'rating', ['reviewer_AI'] + controls)
        c = ctrl_res['reviewer_AI']

        ctrl_info = {}
        for ctrl in controls:
            if ctrl in ctrl_res:
                ctrl_info[ctrl] = {
                    'coeff': ctrl_res[ctrl]['coeff'],
                    'pvalue': ctrl_res[ctrl]['pvalue'],
                }

        results['controlled'] = {
            'coeff': c['coeff'],
            'se': c['se'],
            'pvalue': c['pvalue'],
            'controls': controls,
            'n': ctrl_res['n'],
            'control_coeffs': ctrl_info,
        }

        if verbose:
            r = results['controlled']
            print(f"\n  With controls ({', '.join(controls)}):")
            print(f"    AI Reviewer coeff = {r['coeff']:.4f} "
                  f"(SE = {r['se']:.4f}, p = {r['pvalue']:.4e})")
            for ctrl, info in ctrl_info.items():
                print(f"    {ctrl}: coeff = {info['coeff']:.4f} "
                      f"(p = {info['pvalue']:.4e})")
    else:
        results['controlled'] = None
        if verbose:
            print("\n  No review-level controls available — skipping.")

    return results


# =============================================================================
# DIAGNOSTIC 2: Balanced paired subsample
# =============================================================================

def diagnostic_balanced_pairs(merged: pd.DataFrame,
                               verbose: bool = True) -> Dict:
    """
    Restrict to papers with exactly 1 AI-flagged and ≥1 human review,
    then re-run the paper-FE regression.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("DIAGNOSTIC 2: Balanced Paired-Subsample Check")
        print("=" * 70)

    counts = merged.groupby('submission_number')['reviewer_AI'].agg(
        n_ai='sum', n_total='count'
    )
    counts['n_human'] = counts['n_total'] - counts['n_ai']

    balanced_papers = counts[(counts['n_ai'] == 1) &
                             (counts['n_human'] >= 1)].index
    balanced_df = merged[merged['submission_number'].isin(balanced_papers)]

    n_balanced = len(balanced_papers)
    n_total = merged['submission_number'].nunique()

    if verbose:
        print(f"\n  Balanced subsample: {n_balanced} papers "
              f"(of {n_total} with both types)")
        print(f"  Reviews in subsample: {len(balanced_df)}")

    results = {
        'n_balanced_papers': n_balanced,
        'n_total_papers': n_total,
        'n_reviews': len(balanced_df),
    }

    if n_balanced < 10:
        if verbose:
            print("  Too few balanced papers for regression.")
        results['coeff'] = np.nan
        results['se'] = np.nan
        results['pvalue'] = np.nan
        return results

    res = _fe_ols(balanced_df, 'rating', ['reviewer_AI'])
    r = res['reviewer_AI']
    results['coeff'] = r['coeff']
    results['se'] = r['se']
    results['pvalue'] = r['pvalue']

    if verbose:
        print(f"\n  AI Reviewer coeff = {results['coeff']:.4f} "
              f"(SE = {results['se']:.4f}, p = {results['pvalue']:.4e})")

    return results


# =============================================================================
# DIAGNOSTIC 3: Permutation inference
# =============================================================================

def diagnostic_permutation(merged: pd.DataFrame,
                            n_perms: int = None,
                            verbose: bool = True) -> Dict:
    """
    Permute AI-reviewer labels within each paper to build a null distribution.
    Reports a permutation p-value for the AI-reviewer coefficient.

    Fully vectorized: pre-computes group structure as numpy arrays,
    then shuffles within groups using pure numpy — no pandas groupby
    in the loop.  ~5,000 perms on 27k reviews finishes in seconds.
    """
    if n_perms is None:
        n_perms = min(N_PERMUTATIONS, 5000)

    if verbose:
        print("\n" + "=" * 70)
        print(f"DIAGNOSTIC 3: Permutation Inference ({n_perms:,} permutations)")
        print("=" * 70)

    rng = np.random.RandomState(RANDOM_SEED)

    # Observed coefficient
    observed_coeff = _fe_coeff(merged)

    if verbose:
        print(f"\n  Observed AI Reviewer coeff = {observed_coeff:.4f}")
        print(f"  Permuting AI labels within paper...")

    # --- Pre-compute group structure as numpy arrays ---
    # Sort by paper so each group is contiguous
    sorted_df = merged.sort_values('submission_number').reset_index(drop=True)
    rating = sorted_df['rating'].values.astype(float)
    ai_label = sorted_df['reviewer_AI'].values.astype(float)
    groups = sorted_df['submission_number'].values

    # Find group boundaries (start index, length) for each paper
    _, group_starts, group_counts = np.unique(
        groups, return_index=True, return_counts=True
    )

    # Pre-demean rating (paper means are fixed across permutations)
    rating_dm = rating.copy()
    for s, c in zip(group_starts, group_counts):
        rating_dm[s:s+c] -= rating[s:s+c].mean()

    # --- Vectorized permutation loop ---
    perm_coeffs = np.empty(n_perms)
    perm_ai = ai_label.copy()

    for i in range(n_perms):
        # Shuffle AI labels within each paper (contiguous slices)
        for s, c in zip(group_starts, group_counts):
            perm_ai[s:s+c] = rng.permutation(ai_label[s:s+c])

        # Demean the permuted AI label
        perm_ai_dm = perm_ai.copy()
        for s, c in zip(group_starts, group_counts):
            perm_ai_dm[s:s+c] -= perm_ai[s:s+c].mean()

        # OLS coefficient = cov(y_dm, x_dm) / var(x_dm)
        var_x = np.dot(perm_ai_dm, perm_ai_dm)
        if var_x > 0:
            perm_coeffs[i] = np.dot(rating_dm, perm_ai_dm) / var_x
        else:
            perm_coeffs[i] = 0.0

    perm_p = np.mean(np.abs(perm_coeffs) >= np.abs(observed_coeff))

    results = {
        'observed_coeff': observed_coeff,
        'perm_pvalue': perm_p,
        'null_mean': perm_coeffs.mean(),
        'null_std': perm_coeffs.std(),
        'null_ci_low': np.percentile(perm_coeffs, 2.5),
        'null_ci_high': np.percentile(perm_coeffs, 97.5),
        'n_perms': n_perms,
    }

    if verbose:
        print(f"  Permutation p-value (two-sided): {perm_p:.4f}")
        print(f"  Null distribution: mean = {results['null_mean']:.4f}, "
              f"SD = {results['null_std']:.4f}")
        print(f"  95% null CI: [{results['null_ci_low']:.4f}, "
              f"{results['null_ci_high']:.4f}]")

    return results


# =============================================================================
# DIAGNOSTIC 4: Predictive validity
# =============================================================================

def diagnostic_predictive_validity(merged: pd.DataFrame,
                                    verbose: bool = True) -> Dict:
    """
    Compare whether human vs AI-flagged review means better predict
    the paper's average rating (proxy for editorial consensus).

    Uses Pearson/Spearman correlation between reviewer-type mean and
    the paper-level avg_rating.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("DIAGNOSTIC 4: Predictive Validity")
        print("=" * 70)

    human_means = (merged[merged['reviewer_AI'] == 0]
                   .groupby('submission_number')['rating'].mean()
                   .rename('human_mean'))
    ai_means = (merged[merged['reviewer_AI'] == 1]
                .groupby('submission_number')['rating'].mean()
                .rename('ai_mean'))

    paper_df = pd.DataFrame({'human_mean': human_means, 'ai_mean': ai_means}).dropna()

    if 'avg_rating' in merged.columns:
        avg_ratings = (merged.groupby('submission_number')['avg_rating']
                       .first().rename('avg_rating'))
        paper_df = paper_df.join(avg_ratings).dropna()
    else:
        overall = (merged.groupby('submission_number')['rating']
                   .mean().rename('avg_rating'))
        paper_df = paper_df.join(overall).dropna()

    results = {'n_papers': len(paper_df)}

    if len(paper_df) < 10:
        if verbose:
            print("  Too few papers for predictive validity check.")
        return results

    r_human, p_human = stats.pearsonr(paper_df['human_mean'],
                                       paper_df['avg_rating'])
    r_ai, p_ai = stats.pearsonr(paper_df['ai_mean'],
                                  paper_df['avg_rating'])

    rho_human, _ = stats.spearmanr(paper_df['human_mean'],
                                    paper_df['avg_rating'])
    rho_ai, _ = stats.spearmanr(paper_df['ai_mean'],
                                  paper_df['avg_rating'])

    results.update({
        'pearson_human': r_human,
        'pearson_human_p': p_human,
        'pearson_ai': r_ai,
        'pearson_ai_p': p_ai,
        'spearman_human': rho_human,
        'spearman_ai': rho_ai,
    })

    if verbose:
        print(f"\n  Papers with both reviewer types: {len(paper_df)}")
        print(f"\n  Correlation with paper avg_rating:")
        print(f"    Human reviews (Pearson r): {r_human:.4f} (p = {p_human:.4e})")
        print(f"    AI reviews    (Pearson r): {r_ai:.4f} (p = {p_ai:.4e})")
        print(f"    Human reviews (Spearman rho): {rho_human:.4f}")
        print(f"    AI reviews    (Spearman rho): {rho_ai:.4f}")

        if r_human > r_ai:
            print(f"\n  -> Human reviews are more predictive of consensus "
                  f"(delta_r = {r_human - r_ai:.4f})")
        else:
            print(f"\n  -> AI reviews are more predictive of consensus "
                  f"(delta_r = {r_ai - r_human:.4f})")

    return results


# =============================================================================
# DIAGNOSTIC 5: Summary robustness table
# =============================================================================

def diagnostic_summary_table(diag1: Dict, diag2: Dict,
                              diag3: Dict, diag4: Dict,
                              output_dir: Optional[str] = None,
                              verbose: bool = True) -> pd.DataFrame:
    """
    Compile all diagnostic results into a summary DataFrame and optionally
    save as LaTeX.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("DIAGNOSTIC 5: Summary Robustness Table")
        print("=" * 70)

    rows = []

    if 'baseline' in diag1:
        b = diag1['baseline']
        rows.append({
            'Specification': 'Baseline paper-FE',
            'AI Reviewer coeff': b['coeff'],
            'SE': b['se'],
            'p-value': b['pvalue'],
            'N': b.get('n', np.nan),
        })

    if diag1.get('controlled') is not None:
        c = diag1['controlled']
        rows.append({
            'Specification': f"+ Review controls ({', '.join(c['controls'])})",
            'AI Reviewer coeff': c['coeff'],
            'SE': c['se'],
            'p-value': c['pvalue'],
            'N': c.get('n', np.nan),
        })

    if not np.isnan(diag2.get('coeff', np.nan)):
        rows.append({
            'Specification': f"Balanced pairs (n={diag2['n_balanced_papers']})",
            'AI Reviewer coeff': diag2['coeff'],
            'SE': diag2['se'],
            'p-value': diag2['pvalue'],
            'N': diag2.get('n_reviews', np.nan),
        })

    rows.append({
        'Specification': f"Permutation test ({diag3['n_perms']:,} perms)",
        'AI Reviewer coeff': diag3['observed_coeff'],
        'SE': diag3['null_std'],
        'p-value': diag3['perm_pvalue'],
        'N': np.nan,
    })

    summary = pd.DataFrame(rows)

    if verbose:
        print(f"\n{summary.to_string(index=False, float_format='{:.4f}'.format)}")
        print(f"\nPredictive validity — "
              f"Human r = {diag4.get('pearson_human', np.nan):.4f}, "
              f"AI r = {diag4.get('pearson_ai', np.nan):.4f}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        tex_path = os.path.join(output_dir, 'reviewer_diagnostics.tex')
        latex = summary.to_latex(
            index=False,
            float_format='%.4f',
            caption='Reviewer-side diagnostic regressions',
            label='tab:reviewer_diagnostics',
            column_format='lcccc',
        )
        with open(tex_path, 'w') as f:
            f.write(latex)
        if verbose:
            print(f"\n  LaTeX table saved to: {tex_path}")

    return summary


# =============================================================================
# MASTER RUNNER
# =============================================================================

def run_reviewer_diagnostics(submissions_df: pd.DataFrame,
                              reviews_df: pd.DataFrame,
                              output_dir: str = 'output',
                              n_perms: int = None,
                              verbose: bool = True) -> Dict:
    """
    Run all 5 reviewer-side diagnostics.

    Parameters
    ----------
    submissions_df : DataFrame
        Paper-level data (submission_number, ai_percentage, avg_rating, ...)
    reviews_df : DataFrame
        Review-level data (submission_number, rating, ai_classification, ...)
    output_dir : str
        Directory for LaTeX output
    n_perms : int, optional
        Number of permutations (default: min(N_PERMUTATIONS, 5000))
    verbose : bool
        Print results

    Returns
    -------
    dict with keys: merged_data, diag1_controls, diag2_balanced,
                    diag3_permutation, diag4_validity, summary_table
    """
    if verbose:
        print("\n" + "#" * 70)
        print("#  REVIEWER-SIDE DIAGNOSTICS")
        print("#" * 70)

    merged = _prepare_diagnostic_data(reviews_df, submissions_df)

    if verbose:
        n_ai = (merged['reviewer_AI'] == 1).sum()
        n_human = (merged['reviewer_AI'] == 0).sum()
        print(f"\nPrepared data: {len(merged)} reviews "
              f"({n_human} human, {n_ai} AI) "
              f"across {merged['submission_number'].nunique()} papers")

    if len(merged) < 20:
        warnings.warn("Too few reviews with both AI and human reviewers.")
        return {'error': 'insufficient_data'}

    diag1 = diagnostic_review_controls(merged, verbose=verbose)
    diag2 = diagnostic_balanced_pairs(merged, verbose=verbose)
    diag3 = diagnostic_permutation(merged, n_perms=n_perms, verbose=verbose)
    diag4 = diagnostic_predictive_validity(merged, verbose=verbose)
    summary = diagnostic_summary_table(
        diag1, diag2, diag3, diag4,
        output_dir=output_dir, verbose=verbose
    )

    return {
        'merged_data': merged,
        'diag1_controls': diag1,
        'diag2_balanced': diag2,
        'diag3_permutation': diag3,
        'diag4_validity': diag4,
        'summary_table': summary,
    }
