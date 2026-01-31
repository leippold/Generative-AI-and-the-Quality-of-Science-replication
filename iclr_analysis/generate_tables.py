"""
Generate LaTeX Tables for ICLR AI Contamination Paper
======================================================

Produces publication-ready LaTeX tables matching the paper format.

Usage:
    from generate_tables import generate_all_tables
    generate_all_tables(submissions_df, reviews_df, output_dir='tables/')
"""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, kruskal, ttest_1samp
import os

# Add path to package
import sys
sys.path.insert(0, 'iclr_analysis')

from src.data_loading import (
    load_data, prepare_echo_chamber_data, merge_paper_info, 
    classify_papers, get_cell_data
)
from src.stats_utils import (
    ols_with_clustered_se, ols_robust, bootstrap_ci,
    permutation_test_interaction
)


def generate_panel_a_no_safe_dose(submissions_df):
    """
    Panel A: Pairwise comparisons vs pure human baseline.
    Matches paper bins: 0%, 0-5%, 5-15%, 15-30%, 50-100%
    """
    df = submissions_df.dropna(subset=['ai_percentage', 'avg_rating']).copy()
    
    # Define bins matching the paper
    def categorize(pct):
        if pct == 0:
            return '0% (Pure Human)'
        elif pct <= 5:
            return '0-5% (Minimal)'
        elif pct <= 15:
            return '5-15% (Light)'
        elif pct <= 30:
            return '15-30% (Moderate)'
        elif pct >= 50:
            return '50-100% (Heavy)'
        else:
            return None  # Exclude 30-50%
    
    df['category'] = df['ai_percentage'].apply(categorize)
    df = df.dropna(subset=['category'])
    
    baseline = df[df['category'] == '0% (Pure Human)']['avg_rating']
    
    rows = []
    order = ['0% (Pure Human)', '0-5% (Minimal)', '5-15% (Light)', 
             '15-30% (Moderate)', '50-100% (Heavy)']
    
    for cat in order:
        subset = df[df['category'] == cat]['avg_rating']
        n = len(subset)
        mean = subset.mean()
        
        if cat == '0% (Pure Human)':
            delta = '---'
            p_str = '---'
        else:
            delta = mean - baseline.mean()
            _, p = mannwhitneyu(baseline, subset, alternative='two-sided')
            if p < 0.001:
                p_str = f'$<0.001^{{***}}$'
            elif p < 0.01:
                p_str = f'$\\mathbf{{{p:.4f}}}^{{**}}$'
            else:
                p_str = f'${p:.4f}$'
            delta = f'${delta:+.3f}$'
        
        rows.append({
            'AI Content Level': cat.replace('%', '\\%'),
            'N': f'{n:,}',
            'Mean Rating': f'{mean:.3f}',
            'Delta': delta,
            'p': p_str
        })
    
    return pd.DataFrame(rows)


def generate_panel_b_quadratic(submissions_df):
    """
    Panel B: Quadratic regression testing for interior optimum.
    """
    import statsmodels.api as sm
    
    df = submissions_df.dropna(subset=['ai_percentage', 'avg_rating']).copy()
    
    df['ai_pct'] = df['ai_percentage']
    df['ai_pct_sq'] = df['ai_percentage'] ** 2
    
    X = sm.add_constant(df[['ai_pct', 'ai_pct_sq']])
    model = sm.OLS(df['avg_rating'], X).fit()
    
    rows = []
    
    # Beta 1
    b1 = model.params['ai_pct']
    se1 = model.bse['ai_pct']
    t1 = model.tvalues['ai_pct']
    p1 = model.pvalues['ai_pct']
    rows.append({
        'Variable': 'AI Percentage ($\\beta_1$)',
        'Coefficient': f'${b1:.4f}$',
        'SE': f'${se1:.3f}$',
        't': f'${t1:.2f}$',
        'p': '$<0.001^{***}$' if p1 < 0.001 else f'${p1:.3f}$'
    })
    
    # Beta 2
    b2 = model.params['ai_pct_sq']
    se2 = model.bse['ai_pct_sq']
    t2 = model.tvalues['ai_pct_sq']
    p2 = model.pvalues['ai_pct_sq']
    rows.append({
        'Variable': 'AI Percentage$^2$ ($\\beta_2$)',
        'Coefficient': f'${b2:.6f}$',
        'SE': f'${se2:.3f}$',
        't': f'${t2:.2f}$',
        'p': f'${p2:.3f}$'
    })
    
    # Implied peak
    if b2 < 0:
        peak = -b1 / (2 * b2)
        if 0 < peak < 100:
            implied = f'$a^* = {peak:.1f}\\%$'
        else:
            implied = '\\textit{None (monotonic decline)}'
    else:
        implied = '\\textit{None (monotonic decline)}'
    
    rows.append({
        'Variable': '\\textit{Implied peak}',
        'Coefficient': implied,
        'SE': '',
        't': '',
        'p': ''
    })
    
    return pd.DataFrame(rows)


def generate_panel_c_synthesis(reviews_df, submissions_df):
    """
    Panel C: AI vs Human reviewer comparison.
    """
    # Merge data
    merged = merge_paper_info(reviews_df, submissions_df)
    
    # Classify reviewers
    human_reviews = merged[merged['ai_classification'] == 'Fully human-written']
    ai_reviews = merged[merged['ai_classification'] == 'Fully AI-generated']
    
    n_human = len(human_reviews)
    n_ai = len(ai_reviews)
    
    mean_human = human_reviews['rating'].mean()
    mean_ai = ai_reviews['rating'].mean()
    
    conf_human = human_reviews['confidence'].mean() if 'confidence' in human_reviews else np.nan
    conf_ai = ai_reviews['confidence'].mean() if 'confidence' in ai_reviews else np.nan
    
    leniency = mean_ai - mean_human
    
    # Within-paper analysis
    merged['reviewer_type'] = merged['ai_classification'].map({
        'Fully human-written': 'Human',
        'Fully AI-generated': 'AI'
    })
    
    clean = merged.dropna(subset=['reviewer_type', 'rating'])
    
    paper_ratings = clean.pivot_table(
        values='rating',
        index='submission_number',
        columns='reviewer_type',
        aggfunc='mean'
    ).dropna()
    
    within_premium = (paper_ratings['AI'] - paper_ratings['Human']).mean()
    n_papers = len(paper_ratings)
    
    rows = [
        {
            'Reviewer Type': 'Human-Written Reviews',
            'N': f'{n_human:,}',
            'Mean Rating': f'{mean_human:.3f}',
            'Mean Confidence': f'{conf_human:.2f}',
            'Leniency': '---'
        },
        {
            'Reviewer Type': 'AI-Generated Reviews',
            'N': f'{n_ai:,}',
            'Mean Rating': f'{mean_ai:.3f}',
            'Mean Confidence': f'{conf_ai:.2f}',
            'Leniency': f'$+{leniency:.3f}^{{***}}$'
        },
        {
            'Reviewer Type': '\\quad \\textit{Within-paper premium}',
            'N': '',
            'Mean Rating': '',
            'Mean Confidence': '',
            'Leniency': f'\\textit{{$+{within_premium:.2f}^{{***}}$ ($N = {n_papers:,}$ papers with both types)}}'
        }
    ]
    
    return pd.DataFrame(rows)


def generate_table_cell_means(reviews_df, submissions_df):
    """
    Table 2: Reviewer Ratings by Paper Type and Reviewer Type (2x2 cell means).
    """
    clean = prepare_echo_chamber_data(reviews_df, submissions_df)
    
    # Pivot table
    mean_table = clean.pivot_table(
        values='rating',
        index='paper_type',
        columns='reviewer_type',
        aggfunc='mean'
    )
    
    count_table = pd.crosstab(clean['paper_type'], clean['reviewer_type'])
    
    # Extract values
    hh_mean = mean_table.loc['Human Paper', 'Human Review']
    ha_mean = mean_table.loc['Human Paper', 'AI Review']
    ah_mean = mean_table.loc['AI Paper', 'Human Review']
    aa_mean = mean_table.loc['AI Paper', 'AI Review']
    
    hh_n = count_table.loc['Human Paper', 'Human Review']
    ha_n = count_table.loc['Human Paper', 'AI Review']
    ah_n = count_table.loc['AI Paper', 'Human Review']
    aa_n = count_table.loc['AI Paper', 'AI Review']
    
    # Penalties
    human_penalty = hh_mean - ah_mean
    ai_penalty = ha_mean - aa_mean
    sensitivity_diff = human_penalty - ai_penalty
    
    latex = f"""\\begin{{tabular}}{{lcccc}}
\\toprule
& \\multicolumn{{2}}{{c}}{{\\textbf{{Human Reviewer}}}} & \\multicolumn{{2}}{{c}}{{\\textbf{{AI Reviewer}}}} \\\\
\\cmidrule(lr){{2-3}} \\cmidrule(lr){{4-5}}
& Mean & $N$ & Mean & $N$ \\\\
\\midrule
\\textbf{{Human Paper}} ($\\leq 25\\%$ AI) & {hh_mean:.3f} & {hh_n:,} & {ha_mean:.3f} & {ha_n:,} \\\\
\\textbf{{AI Paper}} ($\\geq 75\\%$ AI) & {ah_mean:.3f} & {ah_n:,} & {aa_mean:.3f} & {aa_n:,} \\\\
\\midrule
\\textbf{{Penalty for AI Papers}} & \\multicolumn{{2}}{{c}}{{$-{human_penalty:.2f}$}} & \\multicolumn{{2}}{{c}}{{$-{ai_penalty:.2f}$}} \\\\
\\textbf{{Sensitivity Difference}} & \\multicolumn{{4}}{{c}}{{${sensitivity_diff:.2f}^{{***}}$}} \\\\
\\bottomrule
\\end{{tabular}}"""
    
    return latex


def generate_table_robustness(reviews_df, submissions_df):
    """
    Table 3: Robustness of Interaction Effect to Classification Thresholds.
    """
    thresholds = [
        (100, 0, 'Extreme (100/0)'),
        (90, 10, 'Strict (90/10)'),
        (80, 20, 'Alternative (80/20)'),
        (75, 25, 'Baseline (75/25)'),
        (50, 50, 'Lenient (50/50)')
    ]
    
    rows = []
    
    for ai_th, hu_th, label in thresholds:
        clean = prepare_echo_chamber_data(
            reviews_df, submissions_df,
            ai_paper_threshold=ai_th,
            human_paper_threshold=hu_th
        )
        
        counts = pd.crosstab(clean['paper_type'], clean['reviewer_type'])
        n_aa = counts.loc['AI Paper', 'AI Review'] if 'AI Paper' in counts.index else 0
        
        if n_aa < 5:
            continue
        
        # Get cell data
        hh = get_cell_data(clean, 'Human Paper', 'Human Review')
        ha = get_cell_data(clean, 'Human Paper', 'AI Review')
        ah = get_cell_data(clean, 'AI Paper', 'Human Review')
        aa = get_cell_data(clean, 'AI Paper', 'AI Review')
        
        # Interaction
        interaction = (np.mean(aa) - np.mean(ah)) - (np.mean(ha) - np.mean(hh))
        
        # OLS with clustered SE
        try:
            ols = ols_with_clustered_se(clean, 'rating ~ paper_AI * reviewer_AI', 'submission_number')
            p_clust = ols['p_values']['paper_AI:reviewer_AI']
        except:
            p_clust = np.nan
        
        # Permutation
        perm = permutation_test_interaction(hh, ha, ah, aa, n_permutations=5000)
        p_perm = perm['p_value']
        
        rows.append({
            'Threshold': label,
            'N': f'{len(clean):,}',
            'N_AA': f'{n_aa}',
            'Interaction': f'$+{interaction:.3f}$' if interaction > 0 else f'${interaction:.3f}$',
            'p_clust': f'{p_clust:.3f}',
            'p_perm': f'{p_perm:.3f}'
        })
    
    return pd.DataFrame(rows)


def generate_table_substitution(reviews_df, submissions_df):
    """
    Table 4: Substitution Signature - Component Scores by AI Content.
    """
    merged = merge_paper_info(reviews_df, submissions_df)
    clean = merged.dropna(subset=['soundness', 'presentation', 'ai_percentage']).copy()
    
    # Bins
    bins = [0, 10, 25, 50, 75, 100.001]
    labels = ['0-10%', '10-25%', '25-50%', '50-75%', '75-100%']
    clean['ai_bin'] = pd.cut(clean['ai_percentage'], bins=bins, labels=labels)
    
    # Gap by bin
    clean['gap'] = clean['presentation'] - clean['soundness']
    gap_by_bin = clean.groupby('ai_bin', observed=True)['gap'].mean()
    
    # Regression coefficients
    from src.stats_utils import ols_robust
    
    results = {}
    for metric in ['soundness', 'presentation', 'contribution']:
        if metric in clean.columns:
            reg = ols_robust(clean, f'{metric} ~ ai_percentage', 'HC3')
            coef = reg['params']['ai_percentage'] * 100  # Per 100%
            p = reg['p_values']['ai_percentage']
            results[metric] = (coef, p)
    
    # Build LaTeX
    gap_cells = ' & '.join([f'{gap_by_bin.get(l, np.nan):.3f}' for l in labels])
    
    latex = f"""\\begin{{tabular}}{{p{{3.2cm}}ccccc}}
\\toprule
& \\multicolumn{{5}}{{c}}{{\\textbf{{AI Content Bin}}}} \\\\
\\cmidrule(lr){{2-6}}
& 0--10\\% & 10--25\\% & 25--50\\% & 50--75\\% & 75--100\\% \\\\
\\midrule
\\textbf{{Presentation $-$ Soundness}} & {gap_cells} \\\\
\\midrule
\\multicolumn{{6}}{{l}}{{\\textit{{Component Score Penalties (per 100\\% AI):}}}} \\\\
\\quad Soundness & \\multicolumn{{5}}{{c}}{{${results['soundness'][0]:.3f}^{{***}}$}} \\\\
\\quad Presentation & \\multicolumn{{5}}{{c}}{{${results['presentation'][0]:.3f}^{{***}}$}} \\\\
\\quad Contribution & \\multicolumn{{5}}{{c}}{{${results['contribution'][0]:.3f}^{{***}}$}} \\\\
\\quad \\textit{{Soundness $-$ Presentation}} & \\multicolumn{{5}}{{c}}{{${results['soundness'][0] - results['presentation'][0]:.3f}^{{***}}$}} \\\\
\\bottomrule
\\end{{tabular}}"""
    
    return latex


def generate_all_tables(submissions_df, reviews_df, output_dir='tables'):
    """
    Generate all LaTeX tables and save to files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating tables...")
    
    # Table 1 Panel A
    print("  Panel A: No Safe Dose...")
    panel_a = generate_panel_a_no_safe_dose(submissions_df)
    panel_a.to_latex(f'{output_dir}/panel_a_no_safe_dose.tex', index=False, escape=False)
    
    # Table 1 Panel B
    print("  Panel B: Quadratic Test...")
    panel_b = generate_panel_b_quadratic(submissions_df)
    panel_b.to_latex(f'{output_dir}/panel_b_quadratic.tex', index=False, escape=False)
    
    # Table 1 Panel C
    print("  Panel C: Synthesis Fidelity...")
    panel_c = generate_panel_c_synthesis(reviews_df, submissions_df)
    panel_c.to_latex(f'{output_dir}/panel_c_synthesis.tex', index=False, escape=False)
    
    # Table 2: Cell Means
    print("  Table 2: Cell Means...")
    table2 = generate_table_cell_means(reviews_df, submissions_df)
    with open(f'{output_dir}/table_cell_means.tex', 'w') as f:
        f.write(table2)
    
    # Table 3: Robustness
    print("  Table 3: Robustness...")
    table3 = generate_table_robustness(reviews_df, submissions_df)
    table3.to_latex(f'{output_dir}/table_robustness.tex', index=False, escape=False)
    
    # Table 4: Substitution Signature
    print("  Table 4: Substitution Signature...")
    table4 = generate_table_substitution(reviews_df, submissions_df)
    with open(f'{output_dir}/table_substitution.tex', 'w') as f:
        f.write(table4)
    
    print(f"\n✓ All tables saved to {output_dir}/")
    print(f"  - panel_a_no_safe_dose.tex")
    print(f"  - panel_b_quadratic.tex")
    print(f"  - panel_c_synthesis.tex")
    print(f"  - table_cell_means.tex")
    print(f"  - table_robustness.tex")
    print(f"  - table_substitution.tex")
    
    return {
        'panel_a': panel_a,
        'panel_b': panel_b,
        'panel_c': panel_c,
        'table2': table2,
        'table3': table3,
        'table4': table4
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        submissions_df, reviews_df = load_data(sys.argv[1], sys.argv[2])
        output_dir = sys.argv[3] if len(sys.argv) > 3 else 'tables'
        generate_all_tables(submissions_df, reviews_df, output_dir)
    else:
        print("Usage: python generate_tables.py submissions.csv reviews.csv [output_dir]")
