"""
Survival Analysis for Retraction Data
======================================

Kaplan-Meier curves and Cox regression comparing AI vs Human papers.

The "GIGO Window" = time from publication to retraction.
- Longer survival = harder to detect (paper sat undetected longer)
- Shorter survival = easier to detect (caught quickly)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional

try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import logrank_test
    from lifelines.plotting import add_at_risk_counts
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False
    print("Warning: lifelines not installed. Run: pip install lifelines")


def kaplan_meier_analysis(df, time_col='GIGO_Years', event_col=None, 
                          group_col='is_ai', max_time=15,
                          labels=None, colors=None,
                          save_path=None, verbose=True):
    """
    Kaplan-Meier survival analysis comparing AI vs Human papers.
    
    Parameters
    ----------
    df : DataFrame
    time_col : str
        Time to event (GIGO_Years)
    event_col : str, optional
        Event indicator (if None, all are events=1)
    group_col : str
        Column distinguishing cohorts
    max_time : float
        Maximum time for plot
    labels : dict, optional
        Labels for each group value
    colors : dict, optional
        Colors for each group
    save_path : str, optional
    
    Returns
    -------
    dict with KM results
    """
    if not HAS_LIFELINES:
        raise ImportError("lifelines required: pip install lifelines")
    
    if labels is None:
        labels = {0: 'Human (Control)', 1: 'AI (Treatment)'}
    
    if colors is None:
        colors = {0: '#2ca02c', 1: '#d62728'}
    
    # All records are "events" (retractions) in this dataset
    if event_col is None:
        df = df.copy()
        df['event'] = 1
        event_col = 'event'
    
    # Cap time for visualization
    df_plot = df.copy()
    df_plot[time_col] = df_plot[time_col].clip(upper=max_time)
    
    results = {}
    
    # Fit KM for each group
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for group_val in sorted(df_plot[group_col].unique()):
        mask = df_plot[group_col] == group_val
        subset = df_plot[mask]
        
        kmf = KaplanMeierFitter()
        kmf.fit(
            subset[time_col],
            event_observed=subset[event_col],
            label=labels.get(group_val, f'Group {group_val}')
        )
        
        kmf.plot_survival_function(
            ax=ax, 
            color=colors.get(group_val, 'gray'),
            ci_show=True,
            linewidth=2
        )
        
        results[group_val] = {
            'median_survival': kmf.median_survival_time_,
            'mean_survival': subset[time_col].mean(),
            'n': len(subset)
        }
    
    # Log-rank test
    human = df_plot[df_plot[group_col] == 0]
    ai = df_plot[df_plot[group_col] == 1]
    
    lr_result = logrank_test(
        human[time_col], ai[time_col],
        event_observed_A=human[event_col],
        event_observed_B=ai[event_col]
    )
    
    results['logrank_p'] = lr_result.p_value
    results['logrank_stat'] = lr_result.test_statistic
    
    # Format plot
    ax.set_xlabel('Years Since Publication', fontsize=12)
    ax.set_ylabel('Survival Probability\n(Not Yet Retracted)', fontsize=12)
    ax.set_title('Time to Retraction: AI vs Human Papers', fontsize=14, fontweight='bold')
    ax.set_xlim(0, max_time)
    ax.set_ylim(0, 1)
    
    # Add log-rank p-value
    p_text = f"Log-rank p = {lr_result.p_value:.2e}" if lr_result.p_value < 0.001 else f"Log-rank p = {lr_result.p_value:.4f}"
    ax.text(0.95, 0.95, p_text, transform=ax.transAxes, ha='right', va='top',
            fontsize=11, bbox=dict(facecolor='white', alpha=0.8))
    
    ax.legend(loc='lower left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if verbose:
        print("\n" + "="*60)
        print("KAPLAN-MEIER ANALYSIS")
        print("="*60)
        print(f"\nHuman papers: n={results[0]['n']:,}, median survival={results[0]['median_survival']:.2f} years")
        print(f"AI papers:    n={results[1]['n']:,}, median survival={results[1]['median_survival']:.2f} years")
        print(f"\nLog-rank test: χ² = {lr_result.test_statistic:.2f}, p = {lr_result.p_value:.4e}")
        
        if results[1]['median_survival'] > results[0]['median_survival']:
            print("\n→ AI papers take LONGER to be retracted (harder to detect)")
        else:
            print("\n→ AI papers are retracted FASTER (easier to detect)")
    
    results['figure'] = fig
    return results


def cox_regression(df, time_col='GIGO_Years', event_col=None,
                   covariates=None, verbose=True):
    """
    Cox proportional hazards regression.
    
    Parameters
    ----------
    df : DataFrame
    time_col : str
    event_col : str, optional
    covariates : list, optional
        Additional covariates beyond is_ai (e.g., ['pub_year'])
    
    Returns
    -------
    dict with hazard ratios and model
    """
    if not HAS_LIFELINES:
        raise ImportError("lifelines required")
    
    df = df.copy()
    
    if event_col is None:
        df['event'] = 1
        event_col = 'event'
    
    # Prepare covariates
    if covariates is None:
        covariates = ['is_ai']
    else:
        covariates = ['is_ai'] + [c for c in covariates if c != 'is_ai']
    
    # Filter to needed columns
    cols = [time_col, event_col] + covariates
    df_cox = df[cols].dropna()
    
    # Fit model
    cph = CoxPHFitter()
    cph.fit(df_cox, duration_col=time_col, event_col=event_col)
    
    # Extract results
    hr_ai = np.exp(cph.params_['is_ai'])
    ci_lower = np.exp(cph.confidence_intervals_.loc['is_ai', '95% lower-bound'])
    ci_upper = np.exp(cph.confidence_intervals_.loc['is_ai', '95% upper-bound'])
    p_value = cph.summary.loc['is_ai', 'p']
    
    results = {
        'hazard_ratio': hr_ai,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_value,
        'model': cph
    }
    
    if verbose:
        print("\n" + "="*60)
        print("COX PROPORTIONAL HAZARDS REGRESSION")
        print("="*60)
        print(f"\nHazard Ratio (AI vs Human): {hr_ai:.4f}")
        print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"p-value: {p_value:.4e}")
        
        if hr_ai < 1:
            reduction = (1 - hr_ai) * 100
            print(f"\n→ AI papers are {reduction:.1f}% LESS likely to be retracted at any given time")
            print("  Interpretation: AI contamination takes LONGER to detect")
        else:
            increase = (hr_ai - 1) * 100
            print(f"\n→ AI papers are {increase:.1f}% MORE likely to be retracted at any given time")
            print("  Interpretation: AI contamination is detected FASTER")
    
    return results


def cox_with_time_interaction(df, time_col='GIGO_Years', 
                               verbose=True):
    """
    Cox model with time-varying effect to test if AI detectability changed over time.
    
    Stratifies by publication era to see if AI papers were harder/easier to detect
    in different periods.
    """
    if not HAS_LIFELINES:
        raise ImportError("lifelines required")
    
    df = df.copy()
    df['event'] = 1
    
    # Define eras
    def get_era(year):
        if year < 2018:
            return 'Pre-2018'
        elif year < 2021:
            return '2018-2020'
        elif year < 2023:
            return '2021-2022'
        else:
            return '2023+'
    
    df['era'] = df['pub_year'].apply(get_era)
    
    results = {}
    
    print("\n" + "="*60)
    print("COX REGRESSION BY ERA")
    print("="*60)
    
    for era in ['Pre-2018', '2018-2020', '2021-2022', '2023+']:
        subset = df[df['era'] == era]
        
        if len(subset) < 50 or subset['is_ai'].sum() < 10:
            print(f"\n{era}: Insufficient data")
            continue
        
        try:
            cph = CoxPHFitter()
            cph.fit(subset[['GIGO_Years', 'event', 'is_ai']], 
                   duration_col='GIGO_Years', event_col='event')
            
            hr = np.exp(cph.params_['is_ai'])
            p = cph.summary.loc['is_ai', 'p']
            
            results[era] = {'hr': hr, 'p': p, 'n': len(subset), 'n_ai': subset['is_ai'].sum()}
            
            print(f"\n{era} (n={len(subset):,}, AI={subset['is_ai'].sum():,}):")
            print(f"  Hazard Ratio: {hr:.3f}, p={p:.4f}")
            
            if hr < 1:
                print(f"  → AI papers {(1-hr)*100:.1f}% less likely to be retracted (HARDER to detect)")
            else:
                print(f"  → AI papers {(hr-1)*100:.1f}% more likely to be retracted (EASIER to detect)")
                
        except Exception as e:
            print(f"\n{era}: Model failed - {e}")
    
    return results


def analyze_high_freq_escalation(df, start_year=2019, window_quarters=4,
                                   save_path=None, verbose=True):
    """
    High-Frequency Escalation Analysis (Quarterly).
    
    Shows how AI detection difficulty has evolved over time.
    
    The Detection Ratio = Median Human GIGO / Median AI GIGO
    - Ratio > 1: AI detected faster (humans stay longer)
    - Ratio < 1: AI harder to detect (AI stays longer)
    - Downward trend = "Escalating Trap"
    
    Parameters
    ----------
    df : DataFrame
        Must have 'is_ai', 'GIGO_Years', 'OriginalPaperDate'
    start_year : int
        Start of analysis window (default: 2019)
    window_quarters : int
        Rolling window size in quarters (default: 4 = 1 year)
    save_path : str, optional
    
    Returns
    -------
    dict with quarterly results and figure
    """
    # Handle column name variations
    ai_col = 'is_ai' if 'is_ai' in df.columns else 'is_ai_contaminated'
    date_col = 'OriginalPaperDate' if 'OriginalPaperDate' in df.columns else 'pub_date'
    
    # Ensure datetime
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    # Filter to recent era
    recent_df = df[df[date_col].dt.year >= start_year].copy()
    
    if len(recent_df) < 100:
        print(f"Warning: Only {len(recent_df)} records after {start_year}")
    
    # Create Year-Quarter period
    recent_df['Period'] = recent_df[date_col].dt.to_period('Q')
    
    results = []
    labels = []
    periods = sorted(recent_df['Period'].unique())
    
    if verbose:
        print("\n" + "="*60)
        print("HIGH-FREQUENCY ESCALATION ANALYSIS (Quarterly)")
        print("="*60)
        print(f"Analysis period: {start_year} onwards")
        print(f"Rolling window: {window_quarters} quarters")
    
    # Rolling window analysis
    for i in range(len(periods) - window_quarters + 1):
        window_periods = periods[i : i + window_quarters]
        window_label = str(window_periods[-1])  # Label by end quarter
        
        sub_df = recent_df[recent_df['Period'].isin(window_periods)]
        
        n_ai = sub_df[ai_col].sum()
        n_human = len(sub_df) - n_ai
        
        if n_ai < 3 or n_human < 3:
            continue
        
        ai_med = sub_df[sub_df[ai_col] == 1]['GIGO_Years'].median()
        hu_med = sub_df[sub_df[ai_col] == 0]['GIGO_Years'].median()
        
        if ai_med > 0 and hu_med > 0:
            ratio = hu_med / ai_med
            results.append({
                'period': window_label,
                'ratio': ratio,
                'n_ai': n_ai,
                'n_human': n_human,
                'ai_median': ai_med,
                'human_median': hu_med
            })
            labels.append(window_label)
            
            if verbose:
                print(f"  {window_label}: Ratio={ratio:.2f} (AI n={n_ai}, median={ai_med:.2f})")
    
    if not results:
        print("Insufficient quarterly data for analysis.")
        return None
    
    results_df = pd.DataFrame(results)
    ratios = results_df['ratio'].values
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # 1. Raw data (light)
    ax.plot(labels, ratios, marker='o', color='purple', linewidth=1, 
            alpha=0.4, label='Raw Quarterly Ratio')
    
    # 2. Smoothed trend (4-quarter rolling average)
    smoothed = pd.Series(ratios).rolling(window=4, center=True).mean()
    ax.plot(labels, smoothed, color='purple', linewidth=4, 
            label='Smoothed Trend (4Q)')
    
    # 3. Linear regression trend
    x_idx = np.arange(len(labels))
    valid_mask = ~np.isnan(ratios)
    if valid_mask.sum() > 2:
        z = np.polyfit(x_idx[valid_mask], ratios[valid_mask], 1)
        p = np.poly1d(z)
        ax.plot(labels, p(x_idx), "r--", alpha=0.8, linewidth=2, 
                label=f'Linear Fit (slope={z[0]:.3f}/Q)')
    
    # Reference line
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, 
               label='Equal Detection Time')
    
    # Annotation if downward trend
    if len(ratios) > 5 and ratios[-1] < ratios[0]:
        ax.annotate('AI becoming harder to detect', 
                   xy=(len(labels)-1, ratios[-1]),
                   xytext=(len(labels)-5, ratios[-1] + 0.4),
                   fontsize=11, color='red',
                   arrowprops=dict(facecolor='red', shrink=0.05, width=2))
    
    ax.set_xticks(range(0, len(labels), max(1, len(labels)//10)))
    ax.set_xticklabels([labels[i] for i in range(0, len(labels), max(1, len(labels)//10))], 
                       rotation=45, ha='right')
    
    ax.set_title("The Escalating Trap: Detection Difficulty Over Time", 
                fontsize=14, fontweight='bold')
    ax.set_ylabel("Detection Ratio (Human / AI)\n(<1 = AI Harder to Detect)", fontsize=11)
    ax.set_xlabel("Quarter (Rolling Window End)", fontsize=11)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Set y-axis limits (lower bound 0.4 for better visualization)
    ax.set_ylim(0.4, None)

    # Color regions
    ax.axhspan(0.4, 1, alpha=0.1, color='red', label='_')  # Danger zone
    ax.axhspan(1, ax.get_ylim()[1], alpha=0.1, color='green', label='_')  # Safe zone

    plt.tight_layout()

    if save_path:
        # Save standard version (default 14x7 aspect - same as other figures)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

        # Save compact/square version (8x8) for alternative layout
        fig_square, ax_square = plt.subplots(figsize=(10, 10))

        # Recreate the plot in square format
        ax_square.plot(labels, ratios, marker='o', color='purple', linewidth=1,
                alpha=0.4, label='Raw Quarterly Ratio')
        ax_square.plot(labels, smoothed, color='purple', linewidth=4,
                label='Smoothed Trend (4Q)')
        if valid_mask.sum() > 2:
            ax_square.plot(labels, p(x_idx), "r--", alpha=0.8, linewidth=2,
                    label=f'Linear Fit (slope={z[0]:.3f}/Q)')
        ax_square.axhline(y=1.0, color='gray', linestyle='--', linewidth=2,
                   label='Equal Detection Time')
        ax_square.set_ylim(0.4, None)
        ax_square.axhspan(0.4, 1, alpha=0.1, color='red', label='_')
        ax_square.axhspan(1, ax_square.get_ylim()[1], alpha=0.1, color='green', label='_')
        ax_square.set_xticks(range(0, len(labels), max(1, len(labels)//8)))
        ax_square.set_xticklabels([labels[i] for i in range(0, len(labels), max(1, len(labels)//8))],
                           rotation=45, ha='right')
        ax_square.set_title("The Escalating Trap: Detection Difficulty Over Time",
                    fontsize=14, fontweight='bold')
        ax_square.set_ylabel("Detection Ratio (Human / AI)\n(<1 = AI Harder to Detect)", fontsize=11)
        ax_square.set_xlabel("Quarter (Rolling Window End)", fontsize=11)
        ax_square.legend(loc='upper left')
        ax_square.grid(True, alpha=0.3)
        plt.tight_layout()

        square_path = save_path.replace('.png', '_square.png')
        fig_square.savefig(square_path, dpi=300, bbox_inches='tight')
        plt.close(fig_square)
        print(f"Saved: {square_path}")
    
    if verbose:
        print(f"\nTrend: {z[0]:.4f} per quarter")
        if z[0] < 0:
            print("→ DOWNWARD TREND: AI papers becoming HARDER to detect over time")
        else:
            print("→ UPWARD TREND: AI papers becoming EASIER to detect over time")
        
        print(f"\nLatest ratio: {ratios[-1]:.2f}")
        if ratios[-1] < 1:
            print("→ Currently in DANGER ZONE (ratio < 1)")
    
    return {
        'results_df': results_df,
        'slope': z[0] if valid_mask.sum() > 2 else None,
        'latest_ratio': ratios[-1],
        'figure': fig
    }


def matched_cohort_analysis(df, start_year=None, end_year=None, max_time=15,
                            save_path=None, verbose=True):
    """
    Matched Cohort Survival Analysis.
    
    Matches AI and Human papers by publication year to control for 
    temporal confounding (detection infrastructure improved over time).
    
    Parameters
    ----------
    df : DataFrame
    start_year : int, optional
        Start of analysis window
    end_year : int, optional
        End of analysis window
    max_time : float
        Maximum time for KM plot
    save_path : str, optional
    
    Returns
    -------
    dict with matched analysis results
    """
    if not HAS_LIFELINES:
        raise ImportError("lifelines required")
    
    # Handle column variations
    ai_col = 'is_ai' if 'is_ai' in df.columns else 'is_ai_contaminated'
    date_col = 'OriginalPaperDate' if 'OriginalPaperDate' in df.columns else 'pub_date'
    
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df['pub_year'] = df[date_col].dt.year
    
    # Filter time horizon
    if start_year:
        df = df[df['pub_year'] >= start_year]
    if end_year:
        df = df[df['pub_year'] <= end_year]
    
    if verbose:
        print("\n" + "="*60)
        print("MATCHED COHORT SURVIVAL ANALYSIS")
        print("="*60)
        year_range = f"{start_year or df['pub_year'].min()}-{end_year or df['pub_year'].max()}"
        print(f"Time horizon: {year_range}")
    
    ai_papers = df[df[ai_col] == 1]
    human_papers = df[df[ai_col] == 0]
    
    if verbose:
        print(f"AI papers: {len(ai_papers):,}")
        print(f"Human papers: {len(human_papers):,}")
    
    # Match by year
    matched_data = []
    match_summary = []
    
    for year in sorted(ai_papers['pub_year'].unique()):
        ai_subset = ai_papers[ai_papers['pub_year'] == year]
        human_subset = human_papers[human_papers['pub_year'] == year]
        
        n_ai = len(ai_subset)
        n_human = len(human_subset)
        
        if n_human > 0 and n_ai > 0:
            matched_data.append(ai_subset)
            
            # Sample with replacement if needed
            n_sample = min(n_ai, n_human) if n_human >= n_ai else n_ai
            sampled_human = human_subset.sample(
                n=n_ai, 
                replace=(n_human < n_ai), 
                random_state=42
            )
            matched_data.append(sampled_human)
            
            match_summary.append({
                'year': year, 
                'n_ai': n_ai, 
                'n_human_available': n_human,
                'n_matched': n_ai
            })
    
    if not matched_data:
        print("No matched data available")
        return None
    
    matched_df = pd.concat(matched_data, ignore_index=True)
    
    if verbose:
        print(f"\nMatched sample: {len(matched_df):,} papers")
        print(f"  AI: {matched_df[ai_col].sum():,}")
        print(f"  Human: {len(matched_df) - matched_df[ai_col].sum():,}")
    
    # Cap time
    matched_df['GIGO_Years_capped'] = matched_df['GIGO_Years'].clip(upper=max_time)
    matched_df['event'] = 1
    
    # Fit KM
    fig, ax = plt.subplots(figsize=(10, 7))
    
    kmf_ai = KaplanMeierFitter()
    ai_data = matched_df[matched_df[ai_col] == 1]
    kmf_ai.fit(ai_data['GIGO_Years_capped'], event_observed=ai_data['event'],
               label='AI Papers (Treatment)')
    kmf_ai.plot_survival_function(ax=ax, ci_show=True, color='#D62728', linewidth=2)
    
    kmf_human = KaplanMeierFitter()
    human_data = matched_df[matched_df[ai_col] == 0]
    kmf_human.fit(human_data['GIGO_Years_capped'], event_observed=human_data['event'],
                  label='Human Papers (Control)')
    kmf_human.plot_survival_function(ax=ax, ci_show=True, color='#2ca02c', linewidth=2)
    
    # Add at-risk counts
    add_at_risk_counts(kmf_ai, kmf_human, ax=ax)
    
    # Log-rank test
    lr = logrank_test(
        ai_data['GIGO_Years_capped'], human_data['GIGO_Years_capped'],
        event_observed_A=ai_data['event'], event_observed_B=human_data['event']
    )
    
    # Annotations
    ax.set_xlabel('Years Since Publication', fontsize=12)
    ax.set_ylabel('Survival Probability (Not Yet Retracted)', fontsize=12)
    ax.set_title(f'Matched Cohort Survival Analysis ({year_range})', 
                fontsize=14, fontweight='bold')
    ax.set_xlim(0, max_time)
    
    p_text = f"Log-rank p = {lr.p_value:.2e}" if lr.p_value < 0.001 else f"Log-rank p = {lr.p_value:.4f}"
    ax.text(0.95, 0.95, p_text, transform=ax.transAxes, ha='right', va='top',
            fontsize=11, bbox=dict(facecolor='white', alpha=0.8))
    
    ax.legend(loc='lower left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    # Results
    results = {
        'matched_df': matched_df,
        'match_summary': pd.DataFrame(match_summary),
        'ai_median': kmf_ai.median_survival_time_,
        'human_median': kmf_human.median_survival_time_,
        'logrank_p': lr.p_value,
        'logrank_stat': lr.test_statistic,
        'figure': fig
    }
    
    if verbose:
        print(f"\nResults:")
        print(f"  AI median survival: {results['ai_median']:.2f} years")
        print(f"  Human median survival: {results['human_median']:.2f} years")
        print(f"  Difference: {results['ai_median'] - results['human_median']:.2f} years")
        print(f"  Log-rank p = {lr.p_value:.4e}")
        
        if results['ai_median'] > results['human_median']:
            print("\n→ AI papers survive LONGER before retraction (harder to detect)")
        else:
            print("\n→ AI papers are retracted FASTER (easier to detect)")
    
    return results


def plot_hazard_ratio_by_era(era_results, save_path=None):
    """
    Forest plot of hazard ratios by publication era.
    """
    eras = list(era_results.keys())
    hrs = [era_results[e]['hr'] for e in eras]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    y_pos = range(len(eras))
    colors = ['#2ca02c' if hr < 1 else '#d62728' for hr in hrs]
    
    ax.barh(y_pos, hrs, color=colors, alpha=0.7, edgecolor='black')
    ax.axvline(x=1, color='black', linestyle='--', linewidth=2, label='HR = 1 (no difference)')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(eras)
    ax.set_xlabel('Hazard Ratio')
    ax.set_title('Detection Difficulty Over Time\n(HR < 1 = Harder to Detect)', fontweight='bold')
    
    # Add values
    for i, (era, hr) in enumerate(zip(eras, hrs)):
        ax.text(hr + 0.05, i, f'{hr:.2f}', va='center', fontsize=10)
    
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# COX PROPORTIONAL HAZARDS DIAGNOSTICS
# =============================================================================

def cox_ph_diagnostics(df, time_col='GIGO_Years', event_col=None,
                       covariates=None, save_path=None, verbose=True):
    """
    Comprehensive diagnostics for Cox Proportional Hazards assumption.

    Tests and visualizations:
    1. Schoenfeld residuals test (statistical test)
    2. Scaled Schoenfeld residuals plot (visual inspection)
    3. Log-log survival plot (visual check)
    4. Martingale residuals (functional form)

    Parameters
    ----------
    df : DataFrame
    time_col : str
    event_col : str, optional
    covariates : list, optional
    save_path : str, optional
        Path to save diagnostic plots

    Returns
    -------
    dict with test results and figures

    Notes
    -----
    The proportional hazards assumption requires that the hazard ratio
    between groups remains constant over time. Violations suggest
    time-varying effects that should be modeled explicitly.

    References
    ----------
    Grambsch, P. M., & Therneau, T. M. (1994). Proportional hazards tests
    and diagnostics based on weighted residuals. Biometrika, 81(3), 515-526.
    """
    if not HAS_LIFELINES:
        raise ImportError("lifelines required")

    df = df.copy()

    if event_col is None:
        df['event'] = 1
        event_col = 'event'

    if covariates is None:
        covariates = ['is_ai']
    else:
        covariates = ['is_ai'] + [c for c in covariates if c != 'is_ai']

    # Prepare data
    cols = [time_col, event_col] + covariates
    df_cox = df[cols].dropna()

    # Fit Cox model
    cph = CoxPHFitter()
    cph.fit(df_cox, duration_col=time_col, event_col=event_col)

    results = {'model': cph}

    if verbose:
        print("\n" + "="*70)
        print("COX PROPORTIONAL HAZARDS DIAGNOSTICS")
        print("="*70)

    # =========================================================================
    # 1. Schoenfeld Residuals Test
    # =========================================================================
    try:
        # lifelines provides check_assumptions method
        ph_test = cph.check_assumptions(df_cox, p_value_threshold=0.05, show_plots=False)

        results['schoenfeld_test'] = {
            'passed': True,  # Will be updated below
            'details': ph_test
        }

        if verbose:
            print("\n--- Schoenfeld Residuals Test ---")
            print("Tests whether log(HR) varies with time")
            print("\nNull hypothesis: Hazard ratio is constant over time")

            # Parse test results
            if hasattr(ph_test, 'summary'):
                print(ph_test.summary)

    except Exception as e:
        if verbose:
            print(f"\nSchoenfeld test failed: {e}")
        results['schoenfeld_test'] = {'passed': None, 'error': str(e)}

    # =========================================================================
    # 2. Visual Diagnostics
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel A: Log-log survival plot
    ax1 = axes[0, 0]
    try:
        ai_col = 'is_ai' if 'is_ai' in df_cox.columns else 'is_ai_contaminated'

        for group_val, label, color in [(0, 'Human', '#2ca02c'), (1, 'AI', '#d62728')]:
            subset = df_cox[df_cox[ai_col] == group_val]
            if len(subset) > 10:
                kmf = KaplanMeierFitter()
                kmf.fit(subset[time_col], event_observed=subset[event_col])

                # Log-log transformation: log(-log(S(t))) vs log(t)
                survival = kmf.survival_function_.iloc[:, 0]
                times = survival.index

                # Filter valid values
                valid = (survival > 0) & (survival < 1) & (times > 0)
                if valid.sum() > 5:
                    log_log_s = np.log(-np.log(survival[valid]))
                    log_t = np.log(times[valid])
                    ax1.plot(log_t, log_log_s, label=label, color=color, linewidth=2)

        ax1.set_xlabel('log(Time)')
        ax1.set_ylabel('log(-log(S(t)))')
        ax1.set_title('A. Log-Log Survival Plot\n(Parallel = PH satisfied)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        results['log_log_plot'] = 'Generated'
    except Exception as e:
        ax1.text(0.5, 0.5, f'Failed: {e}', ha='center', va='center')
        results['log_log_plot'] = str(e)

    # Panel B: Scaled Schoenfeld residuals over time
    ax2 = axes[0, 1]
    try:
        # Get Schoenfeld residuals
        schoenfeld = cph.compute_residuals(df_cox, kind='schoenfeld')

        if 'is_ai' in schoenfeld.columns:
            resid = schoenfeld['is_ai'].values
            event_times = df_cox[df_cox[event_col] == 1][time_col].values

            if len(resid) == len(event_times):
                ax2.scatter(event_times, resid, alpha=0.5, s=20)

                # Add LOWESS smooth
                try:
                    from statsmodels.nonparametric.smoothers_lowess import lowess
                    smoothed = lowess(resid, event_times, frac=0.3)
                    ax2.plot(smoothed[:, 0], smoothed[:, 1], 'r-', linewidth=2,
                            label='LOWESS smooth')
                except:
                    pass

                ax2.axhline(y=0, color='gray', linestyle='--')
                ax2.set_xlabel('Time')
                ax2.set_ylabel('Scaled Schoenfeld Residual (is_ai)')
                ax2.set_title('B. Schoenfeld Residuals vs Time\n(Flat = PH satisfied)')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

        results['schoenfeld_plot'] = 'Generated'
    except Exception as e:
        ax2.text(0.5, 0.5, f'Failed: {e}', ha='center', va='center', transform=ax2.transAxes)
        results['schoenfeld_plot'] = str(e)

    # Panel C: Martingale residuals (functional form check)
    ax3 = axes[1, 0]
    try:
        martingale = cph.compute_residuals(df_cox, kind='martingale')

        if len(martingale) > 0:
            ax3.scatter(df_cox[time_col], martingale.values.flatten(),
                       alpha=0.3, s=15)
            ax3.axhline(y=0, color='red', linestyle='--')

            # LOWESS smooth
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                smoothed = lowess(martingale.values.flatten(),
                                 df_cox[time_col].values, frac=0.3)
                ax3.plot(smoothed[:, 0], smoothed[:, 1], 'r-', linewidth=2)
            except:
                pass

            ax3.set_xlabel('Time')
            ax3.set_ylabel('Martingale Residual')
            ax3.set_title('C. Martingale Residuals\n(Centered at 0 = good fit)')
            ax3.grid(True, alpha=0.3)

        results['martingale_plot'] = 'Generated'
    except Exception as e:
        ax3.text(0.5, 0.5, f'Failed: {e}', ha='center', va='center', transform=ax3.transAxes)
        results['martingale_plot'] = str(e)

    # Panel D: Deviance residuals (outlier detection)
    ax4 = axes[1, 1]
    try:
        deviance = cph.compute_residuals(df_cox, kind='deviance')

        if len(deviance) > 0:
            ax4.scatter(range(len(deviance)), deviance.values.flatten(),
                       alpha=0.5, s=15)
            ax4.axhline(y=0, color='gray', linestyle='--')
            ax4.axhline(y=2, color='red', linestyle=':', label='±2 threshold')
            ax4.axhline(y=-2, color='red', linestyle=':')

            ax4.set_xlabel('Observation Index')
            ax4.set_ylabel('Deviance Residual')
            ax4.set_title('D. Deviance Residuals\n(|r| > 2 = potential outliers)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            # Count outliers
            n_outliers = np.sum(np.abs(deviance.values) > 2)
            results['n_outliers'] = n_outliers
            if verbose:
                print(f"\nDeviance residual outliers (|r| > 2): {n_outliers}")

        results['deviance_plot'] = 'Generated'
    except Exception as e:
        ax4.text(0.5, 0.5, f'Failed: {e}', ha='center', va='center', transform=ax4.transAxes)
        results['deviance_plot'] = str(e)

    plt.suptitle('Cox PH Model Diagnostics', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"\nSaved diagnostics: {save_path}")

    results['figure'] = fig

    # =========================================================================
    # Summary
    # =========================================================================
    if verbose:
        print("\n" + "-"*70)
        print("INTERPRETATION GUIDE")
        print("-"*70)
        print("""
A. Log-Log Plot: Lines should be approximately parallel over time.
   - Non-parallel lines suggest time-varying hazard ratios.

B. Schoenfeld Residuals: Should show no systematic pattern with time.
   - Trend suggests HR changes over follow-up period.

C. Martingale Residuals: Should be centered around 0.
   - Systematic pattern suggests incorrect functional form.

D. Deviance Residuals: Most should be within ±2.
   - Outliers may indicate influential observations.
""")

    return results


# =============================================================================
# E-VALUE SENSITIVITY ANALYSIS
# =============================================================================

def compute_e_value(effect_estimate, ci_lower=None, effect_type='HR'):
    """
    Compute E-value for sensitivity analysis of unmeasured confounding.

    The E-value quantifies the minimum strength of association that an
    unmeasured confounder would need to have with both the treatment
    and outcome to explain away the observed effect.

    Parameters
    ----------
    effect_estimate : float
        Point estimate (HR, RR, or OR)
    ci_lower : float, optional
        Lower bound of confidence interval
    effect_type : str
        'HR' (hazard ratio), 'RR' (risk ratio), or 'OR' (odds ratio)

    Returns
    -------
    dict with e_value, e_value_ci, interpretation

    Notes
    -----
    E-value interpretation:
    - E = 2: Confounder needs 2x association with both treatment and outcome
    - E = 4: Confounder needs 4x association (stronger confounding needed)
    - Higher E-value = more robust to unmeasured confounding

    References
    ----------
    VanderWeele, T. J., & Ding, P. (2017). Sensitivity analysis in
    observational research: Introducing the E-value. Annals of Internal
    Medicine, 167(4), 268-274.
    """
    def _e_value_from_rr(rr):
        """Calculate E-value from risk/rate ratio."""
        if rr < 1:
            rr = 1 / rr  # Take reciprocal for protective effects

        e_val = rr + np.sqrt(rr * (rr - 1))
        return e_val

    # Convert OR to approximate RR if needed (rare outcome assumption)
    if effect_type == 'OR':
        # For rare outcomes, OR ≈ RR
        # For common outcomes, use square root transformation
        rr = np.sqrt(effect_estimate)
    else:
        rr = effect_estimate

    # E-value for point estimate
    e_value = _e_value_from_rr(rr)

    # E-value for CI bound (if provided)
    e_value_ci = None
    if ci_lower is not None:
        if effect_type == 'OR':
            rr_ci = np.sqrt(ci_lower)
        else:
            rr_ci = ci_lower

        # Only compute if CI doesn't cross 1
        if (effect_estimate > 1 and ci_lower > 1) or (effect_estimate < 1 and ci_lower < 1):
            if ci_lower > 1:
                e_value_ci = _e_value_from_rr(ci_lower)
            else:
                e_value_ci = _e_value_from_rr(1/ci_lower)
        else:
            e_value_ci = 1.0  # CI crosses null

    # Interpretation
    if e_value < 1.5:
        strength = "very weak"
    elif e_value < 2.0:
        strength = "weak"
    elif e_value < 3.0:
        strength = "moderate"
    elif e_value < 5.0:
        strength = "strong"
    else:
        strength = "very strong"

    return {
        'e_value': e_value,
        'e_value_ci': e_value_ci,
        'effect_estimate': effect_estimate,
        'effect_type': effect_type,
        'robustness': strength,
        'interpretation': f"An unmeasured confounder would need to be associated with "
                         f"both the treatment and outcome by a risk ratio of at least "
                         f"{e_value:.2f} to explain away the observed effect."
    }


def restricted_cohort_analysis(df, time_col='GIGO_Years',
                                pub_year_start=2018, pub_year_end=2021,
                                min_followup=2.0, verbose=True):
    """
    Compute statistics for restricted cohort analysis (Option C).

    This function calculates the median survival times for a restricted
    time window (e.g., 2018-2021) with minimum follow-up requirements.
    These statistics are used for:
    - Table 5 (Matched Cohort Results)
    - η calibration in Section 5 quantitative analysis

    Parameters
    ----------
    df : DataFrame
    time_col : str
    pub_year_start, pub_year_end : int
        Publication year range (default: 2018-2021)
    min_followup : float
        Minimum years of follow-up (default: 2.0)
    verbose : bool

    Returns
    -------
    dict with:
        - ai_median: Median survival for AI papers
        - human_median: Median survival for Human papers
        - persistence_gap_eta: η = 1 - (human_median / ai_median)
        - percentage_gap: (ai_median - human_median) / human_median * 100
        - n_ai, n_human: Sample sizes
        - logrank_p, logrank_stat: Log-rank test results
    """
    if not HAS_LIFELINES:
        raise ImportError("lifelines required")

    df = df.copy()
    df['event'] = 1

    # Handle column variations
    ai_col = 'is_ai' if 'is_ai' in df.columns else 'is_ai_contaminated'

    # Filter to publication year range
    df = df[(df['pub_year'] >= pub_year_start) & (df['pub_year'] <= pub_year_end)]

    # Apply minimum follow-up requirement
    current_year = 2024  # Approximate
    df['max_possible_followup'] = current_year - df['pub_year']
    df = df[df['max_possible_followup'] >= min_followup]

    # Split cohorts
    human = df[df[ai_col] == 0]
    ai = df[df[ai_col] == 1]

    if len(human) < 20 or len(ai) < 20:
        if verbose:
            print(f"Warning: Small sample sizes (Human: {len(human)}, AI: {len(ai)})")

    # Fit Kaplan-Meier for each cohort
    kmf_human = KaplanMeierFitter()
    kmf_human.fit(human[time_col], event_observed=human['event'])

    kmf_ai = KaplanMeierFitter()
    kmf_ai.fit(ai[time_col], event_observed=ai['event'])

    human_median = kmf_human.median_survival_time_
    ai_median = kmf_ai.median_survival_time_

    # Calculate persistence gap (η) and percentage gap
    persistence_gap_eta = 1 - (human_median / ai_median) if ai_median > 0 else np.nan
    percentage_gap = ((ai_median - human_median) / human_median * 100) if human_median > 0 else np.nan

    # Log-rank test
    lr_result = logrank_test(
        human[time_col], ai[time_col],
        event_observed_A=human['event'], event_observed_B=ai['event']
    )

    results = {
        'ai_median': ai_median,
        'human_median': human_median,
        'persistence_gap_eta': persistence_gap_eta,
        'percentage_gap': percentage_gap,
        'n_ai': len(ai),
        'n_human': len(human),
        'logrank_p': lr_result.p_value,
        'logrank_stat': lr_result.test_statistic,
        'pub_year_range': f'{pub_year_start}-{pub_year_end}',
        'min_followup': min_followup
    }

    if verbose:
        print("\n" + "="*60)
        print("RESTRICTED COHORT ANALYSIS (Option C)")
        print("="*60)
        print(f"Publication years: {pub_year_start}-{pub_year_end}")
        print(f"Minimum follow-up: {min_followup} years")
        print(f"\nSample sizes:")
        print(f"  Human: {len(human):,}")
        print(f"  AI:    {len(ai):,}")
        print(f"\nMedian survival times:")
        print(f"  Human: {human_median:.2f} years")
        print(f"  AI:    {ai_median:.2f} years")
        print(f"\nDerived parameters:")
        print(f"  η (persistence gap): {persistence_gap_eta:.3f}")
        print(f"  Percentage longer:   {percentage_gap:.1f}%")
        print(f"\nLog-rank test: χ²={lr_result.test_statistic:.2f}, p={lr_result.p_value:.4e}")

    return results


def e_value_analysis(df, time_col='GIGO_Years', event_col=None,
                     verbose=True, save_path=None):
    """
    Complete E-value sensitivity analysis for Cox regression results.

    Parameters
    ----------
    df : DataFrame
    time_col : str
    event_col : str, optional
    verbose : bool
    save_path : str, optional

    Returns
    -------
    dict with E-values and sensitivity plot
    """
    if not HAS_LIFELINES:
        raise ImportError("lifelines required")

    df = df.copy()
    if event_col is None:
        df['event'] = 1
        event_col = 'event'

    # Fit Cox model
    cph = CoxPHFitter()
    df_cox = df[['is_ai', time_col, event_col]].dropna()
    cph.fit(df_cox, duration_col=time_col, event_col=event_col)

    # Extract HR and CI
    hr = np.exp(cph.params_['is_ai'])
    ci_lower = np.exp(cph.confidence_intervals_.loc['is_ai', '95% lower-bound'])
    ci_upper = np.exp(cph.confidence_intervals_.loc['is_ai', '95% upper-bound'])

    # Compute E-values
    e_results = compute_e_value(hr, ci_lower, effect_type='HR')

    if verbose:
        print("\n" + "="*70)
        print("E-VALUE SENSITIVITY ANALYSIS")
        print("="*70)
        print(f"\nObserved Hazard Ratio: {hr:.4f}")
        print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"\nE-value (point estimate): {e_results['e_value']:.2f}")
        if e_results['e_value_ci']:
            print(f"E-value (CI limit): {e_results['e_value_ci']:.2f}")
        print(f"\nRobustness: {e_results['robustness']}")
        print(f"\n{e_results['interpretation']}")

    # Create sensitivity plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Contour plot of confounding scenarios
    rr_ue = np.linspace(1, 6, 100)  # RR of confounder-exposure
    rr_ud = np.linspace(1, 6, 100)  # RR of confounder-outcome

    RR_UE, RR_UD = np.meshgrid(rr_ue, rr_ud)

    # Maximum bias factor
    bias_factor = (RR_UE * RR_UD) / (RR_UE + RR_UD - 1)

    # Could the confounding explain away the effect?
    explained_away = bias_factor >= hr

    ax.contourf(RR_UE, RR_UD, explained_away.astype(int),
                levels=[-0.5, 0.5, 1.5], colors=['#90EE90', '#FFB6C1'], alpha=0.5)
    ax.contour(RR_UE, RR_UD, bias_factor, levels=[hr], colors='red', linewidths=2)

    # Mark E-value point
    ax.scatter([e_results['e_value']], [e_results['e_value']],
               s=200, c='red', marker='*', zorder=5, label=f"E-value = {e_results['e_value']:.2f}")

    ax.set_xlabel('RR of Confounder-Treatment Association', fontsize=12)
    ax.set_ylabel('RR of Confounder-Outcome Association', fontsize=12)
    ax.set_title(f"E-value Sensitivity Analysis\nHR = {hr:.2f}", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')

    # Add interpretation text
    ax.text(0.05, 0.95, 'Green: Effect NOT explained away\nPink: Effect explained away',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlim(1, 6)
    ax.set_ylim(1, 6)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"\nSaved: {save_path}")

    e_results['figure'] = fig
    e_results['hr'] = hr
    e_results['ci'] = (ci_lower, ci_upper)

    return e_results
