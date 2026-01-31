"""
Retraction Reason and Subject Area Analysis
=============================================
Compare AI vs Human papers by retraction reasons and subject distribution.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from typing import Dict, Optional


def analyze_retraction_reasons(df, top_n=10, save_path=None, save_dir=None, verbose=True):
    """
    Compare retraction reasons between AI and Human papers.

    Parameters
    ----------
    df : DataFrame
        Must have 'is_ai' and 'Reason' columns
    top_n : int
        Number of top reasons to show
    save_path : str, optional
        Path for combined multi-panel figure
    save_dir : str, optional
        Directory for individual figures (for LaTeX)

    Returns
    -------
    dict with reason analysis
    """
    if 'Reason' not in df.columns:
        print("Warning: 'Reason' column not found")
        return None
    
    df = df.copy()
    df['Reason'] = df['Reason'].fillna('Unknown')
    
    # Define reason categories
    reason_map = {
        'Fabrication/Falsification': ['fabricat', 'falsif', 'fake', 'manipulat'],
        'Plagiarism': ['plagiar', 'duplicate', 'overlap'],
        'AI/Paper Mill': ['paper mill', 'tortured', 'generated', 'ChatGPT', 'AI', 'LLM'],
        'Data Issues': ['data', 'error', 'mistake', 'unreliable'],
        'Image Manipulation': ['image', 'figure', 'duplication'],
        'Authorship Issues': ['author', 'consent', 'permission'],
        'Peer Review Fraud': ['peer review', 'review process'],
        'Ethical Issues': ['ethic', 'IRB', 'consent'],
        'Publisher Error': ['publisher', 'administrative'],
        'Other': []
    }
    
    def categorize_reason(reason_text):
        reason_lower = str(reason_text).lower()
        for category, keywords in reason_map.items():
            if any(kw in reason_lower for kw in keywords):
                return category
        return 'Other'
    
    df['reason_category'] = df['Reason'].apply(categorize_reason)
    
    # Cross-tabulation
    crosstab = pd.crosstab(
        df['reason_category'], 
        df['is_ai'].map({0: 'Human', 1: 'AI'}),
        normalize='columns'
    ) * 100
    
    # Chi-square test
    contingency = pd.crosstab(df['reason_category'], df['is_ai'])
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    
    results = {
        'crosstab': crosstab,
        'crosstab_counts': contingency,
        'chi2': chi2,
        'p_value': p_value,
        'dof': dof
    }
    
    if verbose:
        print("\n" + "="*60)
        print("RETRACTION REASON ANALYSIS")
        print("="*60)
        print("\nReason Distribution (% within cohort):")
        print(crosstab.round(1).to_string())
        print(f"\nChi-square test: χ² = {chi2:.2f}, p = {p_value:.4e}")
    
    # Save individual figures for LaTeX if save_dir provided
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)

        # Figure 1: Reasons by Cohort
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        crosstab.plot(kind='barh', ax=ax1, color=['#2ca02c', '#d62728'])
        ax1.set_xlabel('Percentage')
        ax1.set_title('Retraction Reasons by Cohort')
        ax1.legend(title='Cohort')
        plt.tight_layout()
        fig1.savefig(f'{save_dir}/reasons_by_cohort.png', dpi=300, bbox_inches='tight')
        plt.close(fig1)

        # Figure 2: Difference
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        diff = crosstab['AI'] - crosstab['Human']
        colors = ['#d62728' if d > 0 else '#2ca02c' for d in diff]
        diff.plot(kind='barh', ax=ax2, color=colors)
        ax2.axvline(x=0, color='black', linestyle='--')
        ax2.set_xlabel('AI % - Human %')
        ax2.set_title('Difference (Positive = More Common in AI)')
        plt.tight_layout()
        fig2.savefig(f'{save_dir}/reasons_difference.png', dpi=300, bbox_inches='tight')
        plt.close(fig2)

        print(f"Saved: {save_dir}/reasons_by_cohort.png")
        print(f"Saved: {save_dir}/reasons_difference.png")
        results['figures'] = [f'{save_dir}/reasons_by_cohort.png', f'{save_dir}/reasons_difference.png']

    # Combined figure (for backward compatibility)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Stacked comparison
    ax1 = axes[0]
    crosstab.plot(kind='barh', ax=ax1, color=['#2ca02c', '#d62728'])
    ax1.set_xlabel('Percentage')
    ax1.set_title('A. Retraction Reasons by Cohort')
    ax1.legend(title='Cohort')

    # Panel B: Difference
    ax2 = axes[1]
    diff = crosstab['AI'] - crosstab['Human']
    colors = ['#d62728' if d > 0 else '#2ca02c' for d in diff]
    diff.plot(kind='barh', ax=ax2, color=colors)
    ax2.axvline(x=0, color='black', linestyle='--')
    ax2.set_xlabel('AI % - Human %')
    ax2.set_title('B. Difference (Positive = More Common in AI)')

    plt.suptitle('Retraction Reasons: AI vs Human Papers', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    results['figure'] = fig
    return results


def analyze_subject_areas(df, top_n=15, save_path=None, save_dir=None, verbose=True):
    """
    Compare subject area distribution between AI and Human papers.

    Parameters
    ----------
    df : DataFrame
        Must have 'is_ai' and 'Subject' columns
    save_dir : str, optional
        Directory for individual figures (for LaTeX)
    """
    subject_col = None
    for col in ['Subject', 'subject', 'SubjectArea', 'subject_area']:
        if col in df.columns:
            subject_col = col
            break
    
    if subject_col is None:
        print("Warning: Subject column not found")
        return None
    
    df = df.copy()
    df['subject'] = df[subject_col].fillna('Unknown')
    
    # Get top subjects
    top_subjects = df['subject'].value_counts().head(top_n).index.tolist()
    df_top = df[df['subject'].isin(top_subjects)]
    
    # Cross-tabulation
    crosstab = pd.crosstab(
        df_top['subject'],
        df_top['is_ai'].map({0: 'Human', 1: 'AI'}),
        normalize='columns'
    ) * 100
    
    # Chi-square
    contingency = pd.crosstab(df_top['subject'], df_top['is_ai'])
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    
    # AI concentration ratio
    ai_pct = df.groupby('subject')['is_ai'].mean() * 100
    ai_pct = ai_pct.sort_values(ascending=False)
    
    results = {
        'crosstab': crosstab,
        'ai_concentration': ai_pct,
        'chi2': chi2,
        'p_value': p_value
    }
    
    if verbose:
        print("\n" + "="*60)
        print("SUBJECT AREA ANALYSIS")
        print("="*60)
        print(f"\nTop {top_n} subjects by volume:")
        print(crosstab.round(1).to_string())
        print(f"\nChi-square test: χ² = {chi2:.2f}, p = {p_value:.4e}")

        print("\n\nSubjects with HIGHEST AI concentration:")
        print(ai_pct.head(10).round(1).to_string())

    # Save individual figures for LaTeX if save_dir provided
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)

        # Figure 1: Subject distribution by cohort
        fig1, ax1 = plt.subplots(figsize=(8, 8))
        crosstab.plot(kind='barh', ax=ax1, color=['#2ca02c', '#d62728'])
        ax1.set_xlabel('Percentage')
        ax1.set_title('Subject Distribution by Cohort')
        plt.tight_layout()
        fig1.savefig(f'{save_dir}/subjects_distribution.png', dpi=300, bbox_inches='tight')
        plt.close(fig1)

        # Figure 2: AI concentration by subject
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        top_ai = ai_pct.head(15)
        colors = plt.cm.Reds(top_ai / top_ai.max())
        top_ai.plot(kind='barh', ax=ax2, color=colors)
        ax2.axvline(x=df['is_ai'].mean() * 100, color='black', linestyle='--',
                    label=f'Overall AI %: {df["is_ai"].mean()*100:.1f}%')
        ax2.set_xlabel('% AI Papers')
        ax2.set_title('AI Concentration by Subject')
        ax2.legend()
        plt.tight_layout()
        fig2.savefig(f'{save_dir}/subjects_ai_concentration.png', dpi=300, bbox_inches='tight')
        plt.close(fig2)

        print(f"Saved: {save_dir}/subjects_*.png (2 files)")
        results['figures'] = [
            f'{save_dir}/subjects_distribution.png',
            f'{save_dir}/subjects_ai_concentration.png'
        ]

    # Visualization (combined for backward compatibility)
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    
    # Panel A: Distribution
    ax1 = axes[0]
    crosstab.plot(kind='barh', ax=ax1, color=['#2ca02c', '#d62728'])
    ax1.set_xlabel('Percentage')
    ax1.set_title('A. Subject Distribution by Cohort')
    
    # Panel B: AI concentration
    ax2 = axes[1]
    top_ai = ai_pct.head(15)
    colors = plt.cm.Reds(top_ai / top_ai.max())
    top_ai.plot(kind='barh', ax=ax2, color=colors)
    ax2.axvline(x=df['is_ai'].mean() * 100, color='black', linestyle='--', 
                label=f'Overall AI %: {df["is_ai"].mean()*100:.1f}%')
    ax2.set_xlabel('% AI Papers')
    ax2.set_title('B. AI Concentration by Subject')
    ax2.legend()
    
    plt.suptitle('Subject Area Analysis: AI vs Human Papers', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    results['figure'] = fig
    return results


def analyze_temporal_trends(df, save_path=None, save_dir=None, verbose=True):
    """
    Analyze how AI contamination changed over time.

    Parameters
    ----------
    save_dir : str, optional
        Directory for individual figures (for LaTeX)
    """
    df = df.copy()
    
    # AI papers by publication year
    yearly = df.groupby('pub_year').agg({
        'is_ai': ['sum', 'count', 'mean']
    }).round(4)
    yearly.columns = ['n_ai', 'n_total', 'pct_ai']
    yearly['pct_ai'] *= 100
    
    # GIGO window by year and cohort
    gigo_by_year = df.groupby(['pub_year', 'is_ai'])['GIGO_Years'].mean().unstack()
    gigo_by_year.columns = ['Human', 'AI']
    
    # Detection ratio (AI / Human GIGO)
    gigo_by_year['ratio'] = gigo_by_year['AI'] / gigo_by_year['Human']
    
    results = {
        'yearly_ai': yearly,
        'gigo_by_year': gigo_by_year
    }
    
    if verbose:
        print("\n" + "="*60)
        print("TEMPORAL TRENDS")
        print("="*60)
        print("\nAI Papers by Publication Year:")
        print(yearly.tail(10).to_string())

        print("\n\nMean GIGO Window by Year (Ratio > 1 = AI harder to detect):")
        print(gigo_by_year.tail(10).round(2).to_string())

    # Save individual figures for LaTeX if save_dir provided
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)

        # Figure 1: AI paper count over time
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        yearly['n_ai'].plot(ax=ax1, color='#d62728', linewidth=2, marker='o')
        ax1.set_xlabel('Publication Year')
        ax1.set_ylabel('Number of AI Papers')
        ax1.set_title('AI Paper Retractions Over Time')
        plt.tight_layout()
        fig1.savefig(f'{save_dir}/temporal_ai_count.png', dpi=300, bbox_inches='tight')
        plt.close(fig1)

        # Figure 2: AI contamination rate
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        yearly['pct_ai'].plot(ax=ax2, color='#d62728', linewidth=2, marker='o')
        ax2.set_xlabel('Publication Year')
        ax2.set_ylabel('% AI Papers')
        ax2.set_title('AI Contamination Rate Over Time')
        plt.tight_layout()
        fig2.savefig(f'{save_dir}/temporal_ai_rate.png', dpi=300, bbox_inches='tight')
        plt.close(fig2)

        # Figure 3: Detection speed
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        gigo_by_year[['Human', 'AI']].plot(ax=ax3, linewidth=2, marker='o',
                                           color=['#2ca02c', '#d62728'])
        ax3.set_xlabel('Publication Year')
        ax3.set_ylabel('Mean Years to Retraction')
        ax3.set_title('Detection Speed Over Time')
        ax3.legend()
        plt.tight_layout()
        fig3.savefig(f'{save_dir}/temporal_detection_speed.png', dpi=300, bbox_inches='tight')
        plt.close(fig3)

        # Figure 4: Detection ratio
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        ratio = gigo_by_year['ratio'].dropna()
        colors = ['#d62728' if r > 1 else '#2ca02c' for r in ratio]
        ratio.plot(kind='bar', ax=ax4, color=colors, alpha=0.7, edgecolor='black')
        ax4.axhline(y=1, color='black', linestyle='--', linewidth=2)
        ax4.set_xlabel('Publication Year')
        ax4.set_ylabel('Ratio (AI / Human GIGO)')
        ax4.set_title('Detection Difficulty Ratio (>1 = AI Harder to Detect)')
        ax4.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        fig4.savefig(f'{save_dir}/temporal_detection_ratio.png', dpi=300, bbox_inches='tight')
        plt.close(fig4)

        print(f"Saved: {save_dir}/temporal_*.png (4 files)")
        results['figures'] = [
            f'{save_dir}/temporal_ai_count.png',
            f'{save_dir}/temporal_ai_rate.png',
            f'{save_dir}/temporal_detection_speed.png',
            f'{save_dir}/temporal_detection_ratio.png'
        ]

    # Visualization (combined for backward compatibility)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel A: AI paper count over time
    ax1 = axes[0, 0]
    yearly['n_ai'].plot(ax=ax1, color='#d62728', linewidth=2, marker='o')
    ax1.set_xlabel('Publication Year')
    ax1.set_ylabel('Number of AI Papers')
    ax1.set_title('A. AI Paper Retractions Over Time')
    
    # Panel B: AI percentage
    ax2 = axes[0, 1]
    yearly['pct_ai'].plot(ax=ax2, color='#d62728', linewidth=2, marker='o')
    ax2.set_xlabel('Publication Year')
    ax2.set_ylabel('% AI Papers')
    ax2.set_title('B. AI Contamination Rate Over Time')
    
    # Panel C: GIGO window
    ax3 = axes[1, 0]
    gigo_by_year[['Human', 'AI']].plot(ax=ax3, linewidth=2, marker='o',
                                        color=['#2ca02c', '#d62728'])
    ax3.set_xlabel('Publication Year')
    ax3.set_ylabel('Mean Years to Retraction')
    ax3.set_title('C. Detection Speed Over Time')
    ax3.legend()
    
    # Panel D: Detection ratio
    ax4 = axes[1, 1]
    ratio = gigo_by_year['ratio'].dropna()
    colors = ['#d62728' if r > 1 else '#2ca02c' for r in ratio]
    ratio.plot(kind='bar', ax=ax4, color=colors, alpha=0.7, edgecolor='black')
    ax4.axhline(y=1, color='black', linestyle='--', linewidth=2)
    ax4.set_xlabel('Publication Year')
    ax4.set_ylabel('Ratio (AI / Human GIGO)')
    ax4.set_title('D. Detection Difficulty Ratio\n(>1 = AI Harder to Detect)')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.suptitle('Temporal Analysis of AI Contamination', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    results['figure'] = fig
    return results


def analyze_citation_contamination(df, save_path=None, save_dir=None, verbose=True):
    """
    Citation Contamination Analysis (Finding 5).

    Tests: Do AI papers accumulate more citations before retraction?
    This measures the "GIGO window" in terms of scientific influence.

    Metrics:
    - Citations at retraction: Total "damage" from bad science
    - Citations per GIGO year: Contamination rate (spread velocity)
    - High-citation retractions: "Super-spreader" papers

    Parameters
    ----------
    df : DataFrame
        Must have 'is_ai', 'Citations', and 'GIGO_Years'
    save_path : str, optional
    
    Returns
    -------
    dict with citation analysis results
    """
    from scipy.stats import mannwhitneyu
    
    # Handle column variations
    ai_col = 'is_ai' if 'is_ai' in df.columns else 'is_ai_contaminated'
    
    if 'Citations' not in df.columns:
        print("Warning: 'Citations' column not found in data")
        print("Make sure citation data was merged from problematic papers")
        return None
    
    df = df.copy()
    df['Citations'] = pd.to_numeric(df['Citations'], errors='coerce')
    
    # Filter valid data
    clean = df.dropna(subset=['Citations', ai_col, 'GIGO_Years'])
    clean = clean[clean['Citations'] >= 0]
    
    if len(clean) < 50:
        print(f"Warning: Only {len(clean)} papers with citation data")
    
    if verbose:
        print("\n" + "="*60)
        print("CITATION CONTAMINATION ANALYSIS (Finding 5)")
        print("="*60)
        print(f"Papers with citation data: {len(clean):,}")
    
    # Calculate contamination rate (citations per GIGO year)
    clean['contamination_rate'] = clean['Citations'] / clean['GIGO_Years'].clip(lower=0.1)
    
    ai_df = clean[clean[ai_col] == 1]
    human_df = clean[clean[ai_col] == 0]
    
    results = {
        'n_ai': len(ai_df),
        'n_human': len(human_df),
        
        # Raw citations
        'ai_citations_mean': ai_df['Citations'].mean(),
        'ai_citations_median': ai_df['Citations'].median(),
        'ai_citations_total': ai_df['Citations'].sum(),
        
        'human_citations_mean': human_df['Citations'].mean(),
        'human_citations_median': human_df['Citations'].median(),
        'human_citations_total': human_df['Citations'].sum(),
        
        # Contamination rate (citations / GIGO year)
        'ai_rate_mean': ai_df['contamination_rate'].mean(),
        'ai_rate_median': ai_df['contamination_rate'].median(),
        
        'human_rate_mean': human_df['contamination_rate'].mean(),
        'human_rate_median': human_df['contamination_rate'].median(),
    }
    
    # Statistical tests
    if len(ai_df) >= 10 and len(human_df) >= 10:
        # Test 1: Citations at retraction
        u1, p1 = mannwhitneyu(ai_df['Citations'], human_df['Citations'], alternative='two-sided')
        results['citations_mann_whitney_p'] = p1
        
        # Test 2: Contamination rate
        u2, p2 = mannwhitneyu(ai_df['contamination_rate'], human_df['contamination_rate'], alternative='two-sided')
        results['rate_mann_whitney_p'] = p2
        
        # Effect sizes
        results['citations_diff'] = results['ai_citations_median'] - results['human_citations_median']
        results['citations_ratio'] = results['ai_citations_median'] / max(results['human_citations_median'], 0.1)
        results['rate_diff'] = results['ai_rate_median'] - results['human_rate_median']
        results['rate_ratio'] = results['ai_rate_median'] / max(results['human_rate_median'], 0.1)
    
    # High-citation retractions (top 10%)
    threshold = clean['Citations'].quantile(0.9)
    high_cite = clean[clean['Citations'] >= threshold]
    results['high_cite_threshold'] = threshold
    results['high_cite_n'] = len(high_cite)
    results['high_cite_pct_ai'] = high_cite[ai_col].mean() * 100
    results['overall_pct_ai'] = clean[ai_col].mean() * 100
    
    if verbose:
        print(f"\n--- Citations at Retraction ---")
        print(f"AI papers:    mean={results['ai_citations_mean']:.1f}, median={results['ai_citations_median']:.1f}, total={results['ai_citations_total']:,.0f}")
        print(f"Human papers: mean={results['human_citations_mean']:.1f}, median={results['human_citations_median']:.1f}, total={results['human_citations_total']:,.0f}")
        print(f"Mann-Whitney p = {results.get('citations_mann_whitney_p', 'N/A'):.4e}")
        
        print(f"\n--- Contamination Rate (Citations / GIGO Year) ---")
        print(f"AI papers:    mean={results['ai_rate_mean']:.2f}, median={results['ai_rate_median']:.2f} cites/year")
        print(f"Human papers: mean={results['human_rate_mean']:.2f}, median={results['human_rate_median']:.2f} cites/year")
        print(f"Mann-Whitney p = {results.get('rate_mann_whitney_p', 'N/A'):.4e}")
        
        print(f"\n--- High-Citation Retractions (top 10%, ≥{threshold:.0f} citations) ---")
        print(f"N = {results['high_cite_n']}")
        print(f"% AI in high-citation: {results['high_cite_pct_ai']:.1f}%")
        print(f"% AI overall: {results['overall_pct_ai']:.1f}%")
        
        if results['high_cite_pct_ai'] > results['overall_pct_ai']:
            print("→ AI papers OVER-REPRESENTED in high-citation retractions")
        else:
            print("→ AI papers UNDER-REPRESENTED in high-citation retractions")

    # Save individual figures for LaTeX if save_dir provided
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)

        # Figure 1: Citation distribution
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        data_to_plot = [human_df['Citations'].values, ai_df['Citations'].values]
        bp = ax1.boxplot(data_to_plot, labels=['Human', 'AI'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#2ca02c')
        bp['boxes'][1].set_facecolor('#d62728')
        for box in bp['boxes']:
            box.set_alpha(0.7)
        ax1.set_ylabel('Citations at Retraction')
        ax1.set_title('Citation Distribution by Cohort')
        ax1.set_yscale('log')
        means = [human_df['Citations'].mean(), ai_df['Citations'].mean()]
        ax1.scatter([1, 2], means, color='black', s=100, zorder=5, marker='D', label='Mean')
        ax1.legend()
        plt.tight_layout()
        fig1.savefig(f'{save_dir}/citations_distribution.png', dpi=300, bbox_inches='tight')
        plt.close(fig1)

        # Figure 2: Contamination rate
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        data_rate = [human_df['contamination_rate'].values, ai_df['contamination_rate'].values]
        bp2 = ax2.boxplot(data_rate, labels=['Human', 'AI'], patch_artist=True)
        bp2['boxes'][0].set_facecolor('#2ca02c')
        bp2['boxes'][1].set_facecolor('#d62728')
        for box in bp2['boxes']:
            box.set_alpha(0.7)
        ax2.set_ylabel('Citations per GIGO Year')
        ax2.set_title('Contamination Rate (Citations / Years Before Detection)')
        ax2.set_yscale('log')
        plt.tight_layout()
        fig2.savefig(f'{save_dir}/citations_contamination_rate.png', dpi=300, bbox_inches='tight')
        plt.close(fig2)

        # Figure 3: Scatter plot
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        ax3.scatter(human_df['GIGO_Years'], human_df['Citations'], alpha=0.3,
                    color='#2ca02c', label='Human', s=20)
        ax3.scatter(ai_df['GIGO_Years'], ai_df['Citations'], alpha=0.5,
                    color='#d62728', label='AI', s=20)
        ax3.set_xlabel('GIGO Window (Years)')
        ax3.set_ylabel('Citations')
        ax3.set_title('Detection Time vs. Citation Impact')
        ax3.set_yscale('log')
        ax3.legend()
        plt.tight_layout()
        fig3.savefig(f'{save_dir}/citations_vs_gigo.png', dpi=300, bbox_inches='tight')
        plt.close(fig3)

        # Figure 4: Citation quantiles
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        clean['cite_quantile'] = pd.qcut(clean['Citations'].rank(method='first'), q=5,
                                          labels=['Q1\n(Lowest)', 'Q2', 'Q3', 'Q4', 'Q5\n(Highest)'])
        quantile_ai = clean.groupby('cite_quantile', observed=True)[ai_col].mean() * 100
        colors = ['#2ca02c', '#7cb342', '#fdd835', '#ff9800', '#d62728']
        quantile_ai.plot(kind='bar', ax=ax4, color=colors, edgecolor='black', alpha=0.7)
        ax4.axhline(y=results['overall_pct_ai'], color='black', linestyle='--',
                    linewidth=2, label=f'Overall AI %: {results["overall_pct_ai"]:.1f}%')
        ax4.set_ylabel('% AI Papers')
        ax4.set_xlabel('Citation Quantile')
        ax4.set_title('AI Concentration by Citation Level')
        ax4.legend()
        ax4.tick_params(axis='x', rotation=0)
        plt.tight_layout()
        fig4.savefig(f'{save_dir}/citations_quantiles.png', dpi=300, bbox_inches='tight')
        plt.close(fig4)

        print(f"Saved: {save_dir}/citations_*.png (4 files)")
        results['figures'] = [
            f'{save_dir}/citations_distribution.png',
            f'{save_dir}/citations_contamination_rate.png',
            f'{save_dir}/citations_vs_gigo.png',
            f'{save_dir}/citations_quantiles.png'
        ]

    # Visualization (combined for backward compatibility)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel A: Citation distribution by cohort
    ax1 = axes[0, 0]
    data_to_plot = [human_df['Citations'].values, ai_df['Citations'].values]
    bp = ax1.boxplot(data_to_plot, labels=['Human', 'AI'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#2ca02c')
    bp['boxes'][1].set_facecolor('#d62728')
    for box in bp['boxes']:
        box.set_alpha(0.7)
    ax1.set_ylabel('Citations at Retraction')
    ax1.set_title('A. Citation Distribution by Cohort')
    ax1.set_yscale('log')
    
    # Add mean markers
    means = [human_df['Citations'].mean(), ai_df['Citations'].mean()]
    ax1.scatter([1, 2], means, color='black', s=100, zorder=5, marker='D', label='Mean')
    ax1.legend()
    
    # Panel B: Contamination rate
    ax2 = axes[0, 1]
    data_rate = [human_df['contamination_rate'].values, ai_df['contamination_rate'].values]
    bp2 = ax2.boxplot(data_rate, labels=['Human', 'AI'], patch_artist=True)
    bp2['boxes'][0].set_facecolor('#2ca02c')
    bp2['boxes'][1].set_facecolor('#d62728')
    for box in bp2['boxes']:
        box.set_alpha(0.7)
    ax2.set_ylabel('Citations per GIGO Year')
    ax2.set_title('B. Contamination Rate\n(Citations / Years Before Detection)')
    ax2.set_yscale('log')
    
    # Panel C: Citations vs GIGO window (scatter)
    ax3 = axes[1, 0]
    ax3.scatter(human_df['GIGO_Years'], human_df['Citations'], alpha=0.3, 
                color='#2ca02c', label='Human', s=20)
    ax3.scatter(ai_df['GIGO_Years'], ai_df['Citations'], alpha=0.5, 
                color='#d62728', label='AI', s=20)
    ax3.set_xlabel('GIGO Window (Years)')
    ax3.set_ylabel('Citations')
    ax3.set_title('C. Detection Time vs. Citation Impact')
    ax3.set_yscale('log')
    ax3.legend()
    
    # Panel D: High-citation composition
    ax4 = axes[1, 1]
    
    # Bar chart: % AI in each citation quantile
    clean['cite_quantile'] = pd.qcut(clean['Citations'].rank(method='first'), q=5, 
                                      labels=['Q1\n(Lowest)', 'Q2', 'Q3', 'Q4', 'Q5\n(Highest)'])
    quantile_ai = clean.groupby('cite_quantile', observed=True)[ai_col].mean() * 100
    
    colors = ['#2ca02c', '#7cb342', '#fdd835', '#ff9800', '#d62728']
    quantile_ai.plot(kind='bar', ax=ax4, color=colors, edgecolor='black', alpha=0.7)
    ax4.axhline(y=results['overall_pct_ai'], color='black', linestyle='--', 
                linewidth=2, label=f'Overall AI %: {results["overall_pct_ai"]:.1f}%')
    ax4.set_ylabel('% AI Papers')
    ax4.set_xlabel('Citation Quantile')
    ax4.set_title('D. AI Concentration by Citation Level')
    ax4.legend()
    ax4.tick_params(axis='x', rotation=0)
    
    plt.suptitle('Citation Contamination: The Spread of AI Errors', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved: {save_path}")
    
    results['figure'] = fig
    return results
