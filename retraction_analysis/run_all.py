"""
Run All Analyses + Generate LaTeX Tables
=========================================
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure the retraction_analysis directory is on the path for imports
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from retraction_src.data_loading import load_data, define_ai_cohorts, get_cohort_summary
from retraction_analysis_modules.survival import (
    kaplan_meier_analysis, cox_regression,
    cox_with_time_interaction, plot_hazard_ratio_by_era,
    analyze_high_freq_escalation, matched_cohort_analysis,
    restricted_cohort_analysis
)
from retraction_analysis_modules.descriptive import (
    analyze_retraction_reasons, analyze_subject_areas, analyze_temporal_trends,
    analyze_citation_contamination
)


def run_all_analyses(retraction_path, problematic_path,
                     output_dir='output', start_year=2005,
                     matched_start_year=None, matched_end_year=None,
                     escalation_start_year=2019,
                     single_figures=True):
    """
    Run complete retraction analysis pipeline.

    Parameters
    ----------
    retraction_path : str
        Path to Retraction Watch CSV
    problematic_path : str
        Path to Problematic Paper Screener CSV
    output_dir : str
        Directory for figures and tables
    start_year : int
        Filter papers from this year onwards (default: 2005)
    matched_start_year : int, optional
        Start year for matched cohort analysis
    matched_end_year : int, optional
        End year for matched cohort analysis
    escalation_start_year : int
        Start year for high-frequency escalation (default: 2019)
    single_figures : bool
        If True, save individual figures for LaTeX (default: True)

    Returns
    -------
    dict with all results
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/figures', exist_ok=True)
    os.makedirs(f'{output_dir}/tables', exist_ok=True)

    # Directory for individual figures
    fig_dir = f'{output_dir}/figures' if single_figures else None
    
    print("="*70)
    print("RETRACTION ANALYSIS PIPELINE")
    print("="*70)
    
    # Load data
    rw_df, prob_df = load_data(retraction_path, problematic_path, start_year)
    
    # Define cohorts
    df = define_ai_cohorts(rw_df, prob_df)
    
    results = {'data': df}
    
    # Cohort summary
    summary = get_cohort_summary(df)
    print("\nCohort Summary:")
    print(summary)
    results['cohort_summary'] = summary
    
    # 1. Kaplan-Meier
    print("\n" + "#"*70)
    print("# SURVIVAL ANALYSIS")
    print("#"*70)
    
    km_results = kaplan_meier_analysis(
        df, save_path=f'{output_dir}/figures/fig_kaplan_meier.png'
    )
    results['kaplan_meier'] = km_results
    
    # 2. Cox Regression
    cox_results = cox_regression(df)
    results['cox'] = cox_results
    
    # 3. Cox by Era
    era_results = cox_with_time_interaction(df)
    results['cox_by_era'] = era_results
    
    if era_results:
        plot_hazard_ratio_by_era(
            era_results, 
            save_path=f'{output_dir}/figures/fig_hazard_by_era.png'
        )
    
    # 4. HIGH-FREQUENCY ESCALATION ANALYSIS (THE KEY FIGURE!)
    print("\n" + "#"*70)
    print("# HIGH-FREQUENCY ESCALATION ANALYSIS")
    print("#"*70)
    
    escalation_results = analyze_high_freq_escalation(
        df,
        start_year=escalation_start_year,
        save_path=f'{output_dir}/figures/fig_high_freq_escalation.png'
    )
    results['escalation'] = escalation_results
    
    # 5. MATCHED COHORT ANALYSIS
    print("\n" + "#"*70)
    print("# MATCHED COHORT ANALYSIS")
    print("#"*70)

    matched_results = matched_cohort_analysis(
        df,
        start_year=matched_start_year,
        end_year=matched_end_year,
        save_path=f'{output_dir}/figures/fig_matched_cohort.png'
    )
    results['matched_cohort'] = matched_results

    # 5b. RESTRICTED COHORT ANALYSIS (Option C - for Table 5 and η calibration)
    print("\n" + "#"*70)
    print("# RESTRICTED COHORT ANALYSIS (2018-2021, 2+ years follow-up)")
    print("#"*70)

    restricted_results = restricted_cohort_analysis(
        df,
        pub_year_start=2018,
        pub_year_end=2021,
        min_followup=2.0,
        verbose=True
    )
    results['restricted_cohort'] = restricted_results
    
    # 6. Retraction Reasons
    print("\n" + "#"*70)
    print("# REASON ANALYSIS")
    print("#"*70)

    reason_results = analyze_retraction_reasons(
        df, save_path=f'{output_dir}/figures/fig_reasons.png',
        save_dir=fig_dir
    )
    results['reasons'] = reason_results
    
    # 7. Subject Areas
    print("\n" + "#"*70)
    print("# SUBJECT ANALYSIS")
    print("#"*70)

    subject_results = analyze_subject_areas(
        df, save_path=f'{output_dir}/figures/fig_subjects.png',
        save_dir=fig_dir
    )
    results['subjects'] = subject_results

    # 8. Temporal Trends
    print("\n" + "#"*70)
    print("# TEMPORAL TRENDS")
    print("#"*70)

    temporal_results = analyze_temporal_trends(
        df, save_path=f'{output_dir}/figures/fig_temporal.png',
        save_dir=fig_dir
    )
    results['temporal'] = temporal_results

    # 9. CITATION CONTAMINATION (Finding 5)
    print("\n" + "#"*70)
    print("# CITATION CONTAMINATION (Finding 5)")
    print("#"*70)

    if 'Citations' in df.columns:
        citation_results = analyze_citation_contamination(
            df, save_path=f'{output_dir}/figures/fig_citations.png',
            save_dir=fig_dir
        )
        results['citations'] = citation_results
    else:
        print("Skipping: No citation data available")
        results['citations'] = None
    
    # Generate LaTeX tables
    print("\n" + "#"*70)
    print("# GENERATING LATEX TABLES")
    print("#"*70)

    generate_latex_tables(results, output_dir=f'{output_dir}/tables')

    # Save individual figures for LaTeX
    if single_figures:
        print("\n" + "#"*70)
        print("# SAVING INDIVIDUAL FIGURES FOR LATEX")
        print("#"*70)
        save_individual_figures(results, output_dir=f'{output_dir}/figures')

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nOutputs saved to: {output_dir}/")

    return results


def save_individual_figures(results, output_dir='figures'):
    """
    Extract and save individual figure panels for LaTeX.
    """
    import matplotlib.pyplot as plt
    os.makedirs(output_dir, exist_ok=True)

    print("  Saving individual figures...")

    # The key figures are already saved individually by the analysis functions
    # Here we list what was generated
    expected_figures = [
        'fig_kaplan_meier.png',
        'fig_hazard_by_era.png',
        'fig_high_freq_escalation.png',
        'fig_matched_cohort.png',
        'fig_reasons.png',
        'fig_subjects.png',
        'fig_temporal.png',
        'fig_citations.png',
        'reasons_by_cohort.png',
        'reasons_difference.png'
    ]

    found = []
    for fig_name in expected_figures:
        path = f'{output_dir}/{fig_name}'
        if os.path.exists(path):
            found.append(fig_name)

    print(f"  Generated {len(found)} figure files:")
    for f in found:
        print(f"    - {f}")


def generate_latex_tables(results, output_dir='tables'):
    """
    Generate publication-ready LaTeX tables.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    df = results['data']
    
    # Table 1: Cohort Summary
    print("  Generating Table 1: Cohort Summary...")
    
    human = df[df['is_ai'] == 0]
    ai = df[df['is_ai'] == 1]
    
    table1 = f"""\\begin{{tabular}}{{lcc}}
\\toprule
& \\textbf{{Human (Control)}} & \\textbf{{AI (Treatment)}} \\\\
\\midrule
N & {len(human):,} & {len(ai):,} \\\\
Mean GIGO (years) & {human['GIGO_Years'].mean():.2f} & {ai['GIGO_Years'].mean():.2f} \\\\
Median GIGO (years) & {human['GIGO_Years'].median():.2f} & {ai['GIGO_Years'].median():.2f} \\\\
Publication Year Range & {int(human['pub_year'].min())}--{int(human['pub_year'].max())} & {int(ai['pub_year'].min())}--{int(ai['pub_year'].max())} \\\\
\\bottomrule
\\end{{tabular}}"""
    
    with open(f'{output_dir}/table_cohort_summary.tex', 'w') as f:
        f.write(table1)
    
    # Table 2: Survival Analysis Results
    print("  Generating Table 2: Survival Results...")
    
    km = results['kaplan_meier']
    cox = results['cox']
    
    table2 = f"""\\begin{{tabular}}{{lcc}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Human}} & \\textbf{{AI}} \\\\
\\midrule
N & {km[0]['n']:,} & {km[1]['n']:,} \\\\
Median Survival (years) & {km[0]['median_survival']:.2f} & {km[1]['median_survival']:.2f} \\\\
Mean Survival (years) & {km[0]['mean_survival']:.2f} & {km[1]['mean_survival']:.2f} \\\\
\\midrule
\\multicolumn{{3}}{{l}}{{\\textbf{{Statistical Tests}}}} \\\\
Log-rank $\\chi^2$ & \\multicolumn{{2}}{{c}}{{{km['logrank_stat']:.2f}}} \\\\
Log-rank $p$-value & \\multicolumn{{2}}{{c}}{{{km['logrank_p']:.4e}}} \\\\
Cox Hazard Ratio (AI) & \\multicolumn{{2}}{{c}}{{{cox['hazard_ratio']:.4f}}} \\\\
Cox 95\\% CI & \\multicolumn{{2}}{{c}}{{[{cox['ci_lower']:.4f}, {cox['ci_upper']:.4f}]}} \\\\
\\bottomrule
\\end{{tabular}}"""
    
    with open(f'{output_dir}/table_survival.tex', 'w') as f:
        f.write(table2)
    
    # Table 3: Cox by Era
    print("  Generating Table 3: Hazard Ratios by Era...")
    
    era = results.get('cox_by_era', {})
    
    if era:
        rows = []
        for period, data in era.items():
            sig = '***' if data['p'] < 0.001 else '**' if data['p'] < 0.01 else '*' if data['p'] < 0.05 else ''
            rows.append(f"{period} & {data['n']:,} & {data['n_ai']:,} & {data['hr']:.3f}{sig} & {data['p']:.4f}")
        
        table3 = f"""\\begin{{tabular}}{{lcccc}}
\\toprule
\\textbf{{Era}} & \\textbf{{N}} & \\textbf{{N (AI)}} & \\textbf{{Hazard Ratio}} & \\textbf{{$p$-value}} \\\\
\\midrule
{chr(10).join([r + ' \\\\' for r in rows])}
\\bottomrule
\\end{{tabular}}"""
        
        with open(f'{output_dir}/table_era_hazard.tex', 'w') as f:
            f.write(table3)
    
    # Table 4: Escalation Analysis
    print("  Generating Table 4: Escalation Summary...")
    
    if results.get('escalation') and results['escalation'].get('results_df') is not None:
        esc = results['escalation']
        esc_df = esc['results_df']
        
        # Get first, last, and trend
        first_ratio = esc_df['ratio'].iloc[0]
        last_ratio = esc_df['ratio'].iloc[-1]
        slope = esc.get('slope', 0)
        
        table4 = f"""\\begin{{tabular}}{{lc}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Value}} \\\\
\\midrule
First Period Ratio & {first_ratio:.3f} \\\\
Latest Period Ratio & {last_ratio:.3f} \\\\
Change & {last_ratio - first_ratio:+.3f} \\\\
Trend (per quarter) & {slope:.4f} \\\\
\\midrule
\\multicolumn{{2}}{{l}}{{\\textit{{Ratio $<$ 1 = AI harder to detect}}}} \\\\
\\bottomrule
\\end{{tabular}}"""
        
        with open(f'{output_dir}/table_escalation.tex', 'w') as f:
            f.write(table4)
    
    # Table 5: Restricted Cohort Results (Option C: 2018-2021, 2+ years follow-up)
    print("  Generating Table 5: Restricted Cohort Results (2018-2021)...")

    if results.get('restricted_cohort'):
        rc = results['restricted_cohort']

        # Calculate percentage gap for display
        pct_gap = rc['percentage_gap']

        table5 = f"""\\begin{{tabular}}{{lcc}}
\\toprule
& \\textbf{{AI Papers}} & \\textbf{{Human Papers}} \\\\
\\midrule
\\multicolumn{{3}}{{l}}{{\\textit{{Restricted cohort: {rc['pub_year_range']}, $\\geq${rc['min_followup']:.0f} years follow-up}}}} \\\\
\\midrule
N & {rc['n_ai']:,} & {rc['n_human']:,} \\\\
Median Survival (years) & {rc['ai_median']:.2f} & {rc['human_median']:.2f} \\\\
Difference & \\multicolumn{{2}}{{c}}{{{rc['ai_median'] - rc['human_median']:+.2f} years ({pct_gap:.0f}\\% longer)}} \\\\
$\\eta$ (persistence gap) & \\multicolumn{{2}}{{c}}{{{rc['persistence_gap_eta']:.2f}}} \\\\
\\midrule
Log-rank $\\chi^2$ & \\multicolumn{{2}}{{c}}{{{rc['logrank_stat']:.2f}}} \\\\
Log-rank $p$-value & \\multicolumn{{2}}{{c}}{{{rc['logrank_p']:.4e}}} \\\\
\\bottomrule
\\end{{tabular}}"""

        with open(f'{output_dir}/table_matched_cohort.tex', 'w') as f:
            f.write(table5)

    # Fallback to old matched cohort if restricted not available
    elif results.get('matched_cohort'):
        mc = results['matched_cohort']

        table5 = f"""\\begin{{tabular}}{{lcc}}
\\toprule
& \\textbf{{AI Papers}} & \\textbf{{Human Papers}} \\\\
\\midrule
Median Survival (years) & {mc['ai_median']:.2f} & {mc['human_median']:.2f} \\\\
Difference & \\multicolumn{{2}}{{c}}{{{mc['ai_median'] - mc['human_median']:+.2f} years}} \\\\
\\midrule
Log-rank $\\chi^2$ & \\multicolumn{{2}}{{c}}{{{mc['logrank_stat']:.2f}}} \\\\
Log-rank $p$-value & \\multicolumn{{2}}{{c}}{{{mc['logrank_p']:.4e}}} \\\\
\\bottomrule
\\end{{tabular}}"""

        with open(f'{output_dir}/table_matched_cohort.tex', 'w') as f:
            f.write(table5)
    
    # Table 6: Citation Contamination (Finding 5)
    print("  Generating Table 6: Citation Contamination...")
    
    if results.get('citations') and results['citations'] is not None:
        cit = results['citations']
        
        table6 = f"""\\begin{{tabular}}{{lcc}}
\\toprule
& \\textbf{{AI Papers}} & \\textbf{{Human Papers}} \\\\
\\midrule
\\multicolumn{{3}}{{l}}{{\\textbf{{Citations at Retraction}}}} \\\\
\\quad N & {cit['n_ai']:,} & {cit['n_human']:,} \\\\
\\quad Mean & {cit['ai_citations_mean']:.1f} & {cit['human_citations_mean']:.1f} \\\\
\\quad Median & {cit['ai_citations_median']:.1f} & {cit['human_citations_median']:.1f} \\\\
\\quad Total & {cit['ai_citations_total']:,.0f} & {cit['human_citations_total']:,.0f} \\\\
\\midrule
\\multicolumn{{3}}{{l}}{{\\textbf{{Contamination Rate (cites/year)}}}} \\\\
\\quad Mean & {cit['ai_rate_mean']:.2f} & {cit['human_rate_mean']:.2f} \\\\
\\quad Median & {cit['ai_rate_median']:.2f} & {cit['human_rate_median']:.2f} \\\\
\\midrule
\\multicolumn{{3}}{{l}}{{\\textbf{{Statistical Tests}}}} \\\\
\\quad Citations $p$-value & \\multicolumn{{2}}{{c}}{{{cit.get('citations_mann_whitney_p', 'N/A'):.4e}}} \\\\
\\quad Rate $p$-value & \\multicolumn{{2}}{{c}}{{{cit.get('rate_mann_whitney_p', 'N/A'):.4e}}} \\\\
\\midrule
\\multicolumn{{3}}{{l}}{{\\textbf{{High-Citation Retractions (top 10\\%)}}}} \\\\
\\quad Threshold & \\multicolumn{{2}}{{c}}{{$\\geq {cit['high_cite_threshold']:.0f}$ citations}} \\\\
\\quad \\% AI in high-cite & \\multicolumn{{2}}{{c}}{{{cit['high_cite_pct_ai']:.1f}\\%}} \\\\
\\quad \\% AI overall & \\multicolumn{{2}}{{c}}{{{cit['overall_pct_ai']:.1f}\\%}} \\\\
\\bottomrule
\\end{{tabular}}"""
        
        with open(f'{output_dir}/table_citations.tex', 'w') as f:
            f.write(table6)
    
    # Table 4: Retraction Reasons
    print("  Generating Table 4: Retraction Reasons...")
    
    if results.get('reasons') and results['reasons'].get('crosstab') is not None:
        crosstab = results['reasons']['crosstab'].round(1)
        
        rows = []
        for reason in crosstab.index:
            human_pct = crosstab.loc[reason, 'Human']
            ai_pct = crosstab.loc[reason, 'AI']
            diff = ai_pct - human_pct
            sign = '+' if diff > 0 else ''
            rows.append(f"{reason} & {human_pct:.1f}\\% & {ai_pct:.1f}\\% & {sign}{diff:.1f}\\%")
        
        table4 = f"""\\begin{{tabular}}{{lccc}}
\\toprule
\\textbf{{Reason}} & \\textbf{{Human \\%}} & \\textbf{{AI \\%}} & \\textbf{{Difference}} \\\\
\\midrule
{chr(10).join([r + ' \\\\' for r in rows])}
\\bottomrule
\\end{{tabular}}"""
        
        with open(f'{output_dir}/table_reasons.tex', 'w') as f:
            f.write(table4)
    
    print(f"\n  ✓ Tables saved to {output_dir}/")


# Convenience function for notebook use
def quick_analysis(retraction_path, problematic_path, output_dir='output'):
    """
    Quick wrapper for notebook use.
    """
    return run_all_analyses(retraction_path, problematic_path, output_dir)


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        run_all_analyses(sys.argv[1], sys.argv[2], 
                        sys.argv[3] if len(sys.argv) > 3 else 'output')
    else:
        print("""
Retraction Analysis Pipeline
============================

Usage:
    python run_all.py retraction_watch.csv problematic_papers.csv [output_dir]

Or in Python/Notebook:
    from run_all import run_all_analyses
    results = run_all_analyses('retraction_watch.csv', 'problematic_papers.csv')
""")
