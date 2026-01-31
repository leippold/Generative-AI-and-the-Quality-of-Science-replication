"""
Sample Construction and Reconciliation Module
==============================================

This module provides comprehensive documentation of the sample construction
process for the retraction analysis. It addresses referee concerns about
sample size consistency by providing a complete audit trail.

Key Features:
- Step-by-step sample flow with counts at each stage
- Exclusion criteria documentation
- AI cohort definition audit
- LaTeX-ready reconciliation tables
- Consistency checks across analyses

Usage:
    from sample_construction import SampleConstructionAudit
    audit = SampleConstructionAudit(retraction_path, problematic_path)
    audit.generate_full_report(output_dir)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re
import os


class SampleConstructionAudit:
    """
    Comprehensive audit trail for sample construction.

    Tracks every step of data processing to ensure reproducibility
    and address concerns about sample size discrepancies.
    """

    def __init__(self, retraction_path, problematic_path,
                 start_year=2005, end_year=2025,
                 min_gigo_days=36, ai_definition='standard'):
        """
        Initialize audit with data paths and parameters.

        Parameters
        ----------
        retraction_path : str
            Path to Retraction Watch CSV
        problematic_path : str
            Path to Problematic Paper Screener CSV
        start_year : int
            Start of analysis window (default: 2005)
        end_year : int
            End of analysis window (default: 2025)
        min_gigo_days : int
            Minimum days between publication and retraction (default: 36)
        ai_definition : str
            'standard' or 'strict' AI classification
        """
        self.retraction_path = retraction_path
        self.problematic_path = problematic_path
        self.start_year = start_year
        self.end_year = end_year
        self.min_gigo_days = min_gigo_days
        self.ai_definition = ai_definition

        # Audit trail
        self.flow_steps = []
        self.exclusion_reasons = {}
        self.parameters = {
            'start_year': start_year,
            'end_year': end_year,
            'min_gigo_days': min_gigo_days,
            'ai_definition': ai_definition,
            'extraction_date': datetime.now().strftime('%Y-%m-%d')
        }

        # Data
        self.raw_rw = None
        self.raw_prob = None
        self.final_df = None

    def _log_step(self, step_name, n_before, n_after, description):
        """Log a processing step."""
        excluded = n_before - n_after
        self.flow_steps.append({
            'step': step_name,
            'n_before': n_before,
            'n_after': n_after,
            'excluded': excluded,
            'pct_excluded': 100 * excluded / n_before if n_before > 0 else 0,
            'description': description
        })

    def load_and_process(self, verbose=True):
        """
        Load data with complete audit trail.

        Returns
        -------
        DataFrame : Final analysis sample with is_ai indicator
        """
        if verbose:
            print("="*70)
            print("SAMPLE CONSTRUCTION AUDIT")
            print("="*70)
            print(f"\nParameters:")
            for k, v in self.parameters.items():
                print(f"  {k}: {v}")
            print()

        # =====================================================================
        # STEP 1: Load raw data
        # =====================================================================
        if verbose:
            print("STEP 1: Loading raw data files")

        self.raw_rw = pd.read_csv(self.retraction_path, encoding='latin-1', low_memory=False)
        self.raw_prob = pd.read_csv(self.problematic_path, encoding='latin-1', low_memory=False)

        n_raw_rw = len(self.raw_rw)
        n_raw_prob = len(self.raw_prob)

        self._log_step(
            'Load Retraction Watch',
            n_raw_rw, n_raw_rw,
            f"Raw extraction from Retraction Watch database"
        )

        if verbose:
            print(f"  Retraction Watch: {n_raw_rw:,} records")
            print(f"  Problematic Papers: {n_raw_prob:,} records")

        df = self.raw_rw.copy()

        # =====================================================================
        # STEP 2: Parse dates
        # =====================================================================
        if verbose:
            print("\nSTEP 2: Parsing dates")

        n_before = len(df)
        df['RetractionDate'] = pd.to_datetime(df['RetractionDate'], errors='coerce')
        df['OriginalPaperDate'] = pd.to_datetime(df['OriginalPaperDate'], errors='coerce')

        # Track missing dates separately
        missing_retraction_date = df['RetractionDate'].isna().sum()
        missing_pub_date = df['OriginalPaperDate'].isna().sum()
        missing_either = (df['RetractionDate'].isna() | df['OriginalPaperDate'].isna()).sum()

        self.exclusion_reasons['missing_retraction_date'] = missing_retraction_date
        self.exclusion_reasons['missing_publication_date'] = missing_pub_date

        df = df.dropna(subset=['RetractionDate', 'OriginalPaperDate'])
        n_after = len(df)

        self._log_step(
            'Valid dates required',
            n_before, n_after,
            f"Exclude records with missing retraction or publication date"
        )

        if verbose:
            print(f"  Missing retraction date: {missing_retraction_date:,}")
            print(f"  Missing publication date: {missing_pub_date:,}")
            print(f"  Excluded (missing either): {n_before - n_after:,}")

        # =====================================================================
        # STEP 3: Apply time horizon filter
        # =====================================================================
        if verbose:
            print(f"\nSTEP 3: Time horizon filter ({self.start_year}-{self.end_year})")

        n_before = len(df)
        df['pub_year'] = df['OriginalPaperDate'].dt.year

        before_start = (df['pub_year'] < self.start_year).sum()
        after_end = (df['pub_year'] > self.end_year).sum()

        self.exclusion_reasons['before_start_year'] = before_start
        self.exclusion_reasons['after_end_year'] = after_end

        df = df[(df['pub_year'] >= self.start_year) & (df['pub_year'] <= self.end_year)]
        n_after = len(df)

        self._log_step(
            f'Time horizon ({self.start_year}-{self.end_year})',
            n_before, n_after,
            f"Restrict to papers published {self.start_year}-{self.end_year}"
        )

        if verbose:
            print(f"  Before {self.start_year}: {before_start:,}")
            print(f"  After {self.end_year}: {after_end:,}")
            print(f"  Remaining: {n_after:,}")

        # =====================================================================
        # STEP 4: Calculate GIGO window and filter
        # =====================================================================
        if verbose:
            print(f"\nSTEP 4: GIGO window calculation (min {self.min_gigo_days} days)")

        n_before = len(df)
        df['GIGO_Days'] = (df['RetractionDate'] - df['OriginalPaperDate']).dt.days
        df['GIGO_Years'] = df['GIGO_Days'] / 365.25

        # Negative GIGO (retraction before publication - data error)
        negative_gigo = (df['GIGO_Days'] < 0).sum()
        # Too fast (< min_gigo_days)
        too_fast = ((df['GIGO_Days'] >= 0) & (df['GIGO_Days'] < self.min_gigo_days)).sum()

        self.exclusion_reasons['negative_gigo'] = negative_gigo
        self.exclusion_reasons['gigo_too_short'] = too_fast

        df = df[df['GIGO_Days'] >= self.min_gigo_days]
        n_after = len(df)

        self._log_step(
            f'Valid GIGO (≥{self.min_gigo_days} days)',
            n_before, n_after,
            f"Exclude negative or implausibly short retraction windows"
        )

        if verbose:
            print(f"  Negative GIGO (data error): {negative_gigo:,}")
            print(f"  GIGO < {self.min_gigo_days} days: {too_fast:,}")
            print(f"  Remaining: {n_after:,}")

        # =====================================================================
        # STEP 5: Define AI cohorts
        # =====================================================================
        if verbose:
            print("\nSTEP 5: AI cohort classification")

        n_before = len(df)
        df = self._classify_ai_cohorts(df, verbose=verbose)
        n_after = len(df)

        self._log_step(
            'AI classification',
            n_before, n_after,
            f"Classify papers as AI or Human based on {self.ai_definition} definition"
        )

        # =====================================================================
        # STEP 6: Filter to complete covariates (for survival analysis)
        # =====================================================================
        if verbose:
            print("\nSTEP 6: Complete covariates filter")

        n_before = len(df)
        df['retraction_year'] = df['RetractionDate'].dt.year

        # Required columns for survival analysis
        required_cols = ['is_ai', 'GIGO_Years', 'pub_year']
        n_missing = df[required_cols].isna().any(axis=1).sum()

        df = df.dropna(subset=required_cols)
        n_after = len(df)

        self.exclusion_reasons['missing_survival_covariates'] = n_before - n_after

        self._log_step(
            'Complete covariates',
            n_before, n_after,
            f"Exclude records with missing survival covariates"
        )

        if verbose:
            print(f"  Missing covariates: {n_before - n_after:,}")
            print(f"  Remaining: {n_after:,}")

        # =====================================================================
        # FINAL: Store results
        # =====================================================================
        self.final_df = df

        n_ai = df['is_ai'].sum()
        n_human = len(df) - n_ai

        if verbose:
            print("\n" + "="*70)
            print("FINAL SAMPLE")
            print("="*70)
            print(f"\n  Total: N = {len(df):,}")
            print(f"  AI Cohort: N = {n_ai:,} ({100*n_ai/len(df):.1f}%)")
            print(f"  Human Cohort: N = {n_human:,} ({100*n_human/len(df):.1f}%)")
            print(f"\n  Year range: {df['pub_year'].min()}-{df['pub_year'].max()}")
            print(f"  Median GIGO: {df['GIGO_Years'].median():.2f} years")

        return df

    def _classify_ai_cohorts(self, df, verbose=True):
        """
        Classify papers into AI vs Human cohorts.

        Classification criteria (standard definition):
        1. Flagged by AI-detection tools (tortured phrases, SCIgen, Seek&Blastn)
        2. Retraction reason contains AI-related keywords

        Classification criteria (strict definition):
        - Only detector flags, no keyword matching
        """
        df = df.copy()

        # Normalize titles for matching
        df['title_norm'] = df['Title'].apply(self._normalize_title)

        prob_df = self.raw_prob.copy()
        prob_df['title_norm'] = prob_df['Title'].apply(self._normalize_title)

        # Merge detector info
        merge_cols = ['title_norm', 'Detectors']
        if 'Citations' in prob_df.columns:
            merge_cols.append('Citations')

        df = df.merge(
            prob_df[merge_cols].drop_duplicates(subset=['title_norm']),
            on='title_norm',
            how='left'
        )

        df['Detectors'] = df['Detectors'].fillna('')
        df['Reason'] = df['Reason'].fillna('')

        # AI Detection Criteria
        target_detectors = ['tortured', 'scigen', 'Seek&Blastn']
        ai_keywords = [
            'generated', 'ChatGPT', 'LLM', 'AI', 'hallucination',
            'fake', 'paper mill', 'tortured phrases', 'fabricat'
        ]

        # Flag 1: Detector-based
        has_ai_detector = df['Detectors'].apply(
            lambda x: any(d.lower() in str(x).lower() for d in target_detectors)
        )

        # Flag 2: Keyword-based
        has_ai_reason = df['Reason'].str.contains(
            '|'.join(ai_keywords), case=False, na=False
        )

        # Track breakdown
        n_detector_only = (has_ai_detector & ~has_ai_reason).sum()
        n_keyword_only = (~has_ai_detector & has_ai_reason).sum()
        n_both = (has_ai_detector & has_ai_reason).sum()

        self.ai_classification_breakdown = {
            'detector_only': n_detector_only,
            'keyword_only': n_keyword_only,
            'both': n_both,
            'detector_total': has_ai_detector.sum(),
            'keyword_total': has_ai_reason.sum()
        }

        if verbose:
            print(f"  Detector flags: {has_ai_detector.sum():,}")
            print(f"  Keyword matches: {has_ai_reason.sum():,}")
            print(f"  Both: {n_both:,}")

        # Apply definition
        if self.ai_definition == 'strict':
            df['is_ai'] = has_ai_detector.astype(int)
            if verbose:
                print(f"  Using STRICT definition (detector only)")
        else:
            df['is_ai'] = (has_ai_detector | has_ai_reason).astype(int)
            if verbose:
                print(f"  Using STANDARD definition (detector OR keyword)")

        # Clean up
        df = df.drop(columns=['title_norm'], errors='ignore')

        # Handle citations
        if 'Citations' in df.columns:
            df['Citations'] = pd.to_numeric(df['Citations'], errors='coerce')

        return df

    @staticmethod
    def _normalize_title(title):
        """Normalize title for matching."""
        if not isinstance(title, str):
            return ""
        return re.sub(r'[^a-z0-9]', '', title.lower())

    def get_flow_table(self):
        """
        Get sample flow as DataFrame.

        Returns
        -------
        DataFrame with step-by-step sample counts
        """
        if not self.flow_steps:
            raise ValueError("Run load_and_process() first")

        return pd.DataFrame(self.flow_steps)

    def get_exclusion_summary(self):
        """
        Get summary of exclusion reasons.

        Returns
        -------
        DataFrame with exclusion counts
        """
        if not self.exclusion_reasons:
            raise ValueError("Run load_and_process() first")

        return pd.DataFrame([
            {'reason': k, 'n_excluded': v}
            for k, v in self.exclusion_reasons.items()
        ])

    def to_latex_flow_table(self, caption=None, label=None):
        """
        Generate LaTeX table for sample flow.

        Returns
        -------
        str : LaTeX table code
        """
        df = self.get_flow_table()

        if caption is None:
            caption = "Sample Construction Flow"
        if label is None:
            label = "tab:sample_flow"

        latex = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\begin{tabular}{lrrrl}",
            "\\toprule",
            "Step & N Before & N After & Excluded & Description \\\\",
            "\\midrule"
        ]

        for _, row in df.iterrows():
            latex.append(
                f"{row['step']} & {row['n_before']:,} & {row['n_after']:,} & "
                f"{row['excluded']:,} & {row['description'][:40]}... \\\\"
            )

        latex.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])

        return '\n'.join(latex)

    def to_latex_cohort_table(self, caption=None, label=None):
        """
        Generate LaTeX table for final cohort characteristics.

        Returns
        -------
        str : LaTeX table code
        """
        if self.final_df is None:
            raise ValueError("Run load_and_process() first")

        df = self.final_df

        if caption is None:
            caption = "Final Sample Characteristics by Cohort"
        if label is None:
            label = "tab:cohort_characteristics"

        # Compute statistics by cohort
        stats = []
        for cohort_val, cohort_name in [(0, 'Human'), (1, 'AI')]:
            subset = df[df['is_ai'] == cohort_val]
            stats.append({
                'Cohort': cohort_name,
                'N': len(subset),
                'Pct': 100 * len(subset) / len(df),
                'Year_min': subset['pub_year'].min(),
                'Year_max': subset['pub_year'].max(),
                'GIGO_median': subset['GIGO_Years'].median(),
                'GIGO_mean': subset['GIGO_Years'].mean(),
                'GIGO_std': subset['GIGO_Years'].std()
            })

        # Total row
        stats.append({
            'Cohort': 'Total',
            'N': len(df),
            'Pct': 100.0,
            'Year_min': df['pub_year'].min(),
            'Year_max': df['pub_year'].max(),
            'GIGO_median': df['GIGO_Years'].median(),
            'GIGO_mean': df['GIGO_Years'].mean(),
            'GIGO_std': df['GIGO_Years'].std()
        })

        latex = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\begin{tabular}{lrrcccc}",
            "\\toprule",
            "Cohort & N & \\% & Year Range & \\multicolumn{3}{c}{GIGO (Years)} \\\\",
            "\\cmidrule(lr){5-7}",
            " & & & & Median & Mean & SD \\\\",
            "\\midrule"
        ]

        for s in stats:
            if s['Cohort'] == 'Total':
                latex.append("\\midrule")
            latex.append(
                f"{s['Cohort']} & {s['N']:,} & {s['Pct']:.1f} & "
                f"{s['Year_min']}-{s['Year_max']} & "
                f"{s['GIGO_median']:.2f} & {s['GIGO_mean']:.2f} & {s['GIGO_std']:.2f} \\\\"
            )

        latex.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])

        return '\n'.join(latex)

    def generate_full_report(self, output_dir, verbose=True):
        """
        Generate complete sample construction report.

        Parameters
        ----------
        output_dir : str
            Directory to save output files
        verbose : bool
            Print progress

        Returns
        -------
        dict with paths to generated files
        """
        os.makedirs(output_dir, exist_ok=True)

        if self.final_df is None:
            self.load_and_process(verbose=verbose)

        outputs = {}

        # 1. Flow table (CSV)
        flow_path = os.path.join(output_dir, 'sample_flow.csv')
        self.get_flow_table().to_csv(flow_path, index=False)
        outputs['flow_csv'] = flow_path

        # 2. Flow table (LaTeX)
        latex_flow_path = os.path.join(output_dir, 'sample_flow.tex')
        with open(latex_flow_path, 'w') as f:
            f.write(self.to_latex_flow_table())
        outputs['flow_latex'] = latex_flow_path

        # 3. Cohort table (LaTeX)
        latex_cohort_path = os.path.join(output_dir, 'cohort_characteristics.tex')
        with open(latex_cohort_path, 'w') as f:
            f.write(self.to_latex_cohort_table())
        outputs['cohort_latex'] = latex_cohort_path

        # 4. Exclusion reasons (CSV)
        excl_path = os.path.join(output_dir, 'exclusion_reasons.csv')
        self.get_exclusion_summary().to_csv(excl_path, index=False)
        outputs['exclusions_csv'] = excl_path

        # 5. Full markdown report
        report_path = os.path.join(output_dir, 'sample_construction_report.md')
        with open(report_path, 'w') as f:
            f.write(self._generate_markdown_report())
        outputs['report_md'] = report_path

        # 6. Parameters (JSON)
        import json
        params_path = os.path.join(output_dir, 'sample_parameters.json')
        with open(params_path, 'w') as f:
            json.dump(self.parameters, f, indent=2)
        outputs['params_json'] = params_path

        if verbose:
            print(f"\n✓ Generated {len(outputs)} output files in {output_dir}")

        return outputs

    def _generate_markdown_report(self):
        """Generate comprehensive markdown report."""
        df = self.final_df
        n_ai = df['is_ai'].sum()
        n_human = len(df) - n_ai

        report = f"""# Sample Construction Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. Data Sources

| Source | Records |
|--------|---------|
| Retraction Watch Database | {len(self.raw_rw):,} |
| Problematic Paper Screener | {len(self.raw_prob):,} |

## 2. Sample Construction Parameters

| Parameter | Value |
|-----------|-------|
| Publication year range | {self.start_year}-{self.end_year} |
| Minimum GIGO window | {self.min_gigo_days} days |
| AI classification | {self.ai_definition} definition |
| Extraction date | {self.parameters['extraction_date']} |

## 3. Sample Flow

"""
        # Add flow table
        flow_df = self.get_flow_table()
        report += "| Step | N Before | N After | Excluded | Description |\n"
        report += "|------|----------|---------|----------|-------------|\n"
        for _, row in flow_df.iterrows():
            report += f"| {row['step']} | {row['n_before']:,} | {row['n_after']:,} | {row['excluded']:,} | {row['description']} |\n"

        report += f"""

## 4. Exclusion Details

"""
        excl_df = self.get_exclusion_summary()
        report += "| Reason | N Excluded |\n"
        report += "|--------|------------|\n"
        for _, row in excl_df.iterrows():
            report += f"| {row['reason']} | {row['n_excluded']:,} |\n"

        report += f"""

## 5. AI Classification Breakdown

| Classification Source | N Papers |
|----------------------|----------|
| Detector flags only | {self.ai_classification_breakdown.get('detector_only', 'N/A'):,} |
| Keyword matches only | {self.ai_classification_breakdown.get('keyword_only', 'N/A'):,} |
| Both detector + keyword | {self.ai_classification_breakdown.get('both', 'N/A'):,} |

**AI Detectors used:** tortured phrases, SCIgen, Seek&Blastn

**Keywords used:** generated, ChatGPT, LLM, AI, hallucination, fake, paper mill, tortured phrases, fabricat

## 6. Final Sample Summary

| Cohort | N | Percentage |
|--------|---|------------|
| **AI (Treatment)** | {n_ai:,} | {100*n_ai/len(df):.1f}% |
| **Human (Control)** | {n_human:,} | {100*n_human/len(df):.1f}% |
| **Total** | {len(df):,} | 100.0% |

### Cohort Characteristics

| Statistic | AI Cohort | Human Cohort | Total |
|-----------|-----------|--------------|-------|
| N | {n_ai:,} | {n_human:,} | {len(df):,} |
| Year range | {df[df['is_ai']==1]['pub_year'].min()}-{df[df['is_ai']==1]['pub_year'].max()} | {df[df['is_ai']==0]['pub_year'].min()}-{df[df['is_ai']==0]['pub_year'].max()} | {df['pub_year'].min()}-{df['pub_year'].max()} |
| Median GIGO (years) | {df[df['is_ai']==1]['GIGO_Years'].median():.2f} | {df[df['is_ai']==0]['GIGO_Years'].median():.2f} | {df['GIGO_Years'].median():.2f} |
| Mean GIGO (years) | {df[df['is_ai']==1]['GIGO_Years'].mean():.2f} | {df[df['is_ai']==0]['GIGO_Years'].mean():.2f} | {df['GIGO_Years'].mean():.2f} |

---

*This report was automatically generated to document the sample construction process and address concerns about sample size consistency.*
"""
        return report

    def verify_consistency(self, expected_total=None, expected_ai=None, expected_human=None):
        """
        Verify sample sizes match expected values.

        Parameters
        ----------
        expected_total : int, optional
        expected_ai : int, optional
        expected_human : int, optional

        Returns
        -------
        dict with verification results
        """
        if self.final_df is None:
            raise ValueError("Run load_and_process() first")

        actual_total = len(self.final_df)
        actual_ai = self.final_df['is_ai'].sum()
        actual_human = actual_total - actual_ai

        results = {
            'actual_total': actual_total,
            'actual_ai': actual_ai,
            'actual_human': actual_human,
            'checks_passed': True,
            'discrepancies': []
        }

        if expected_total is not None and actual_total != expected_total:
            results['checks_passed'] = False
            results['discrepancies'].append(
                f"Total: expected {expected_total:,}, got {actual_total:,} (diff: {actual_total - expected_total:+,})"
            )

        if expected_ai is not None and actual_ai != expected_ai:
            results['checks_passed'] = False
            results['discrepancies'].append(
                f"AI: expected {expected_ai:,}, got {actual_ai:,} (diff: {actual_ai - expected_ai:+,})"
            )

        if expected_human is not None and actual_human != expected_human:
            results['checks_passed'] = False
            results['discrepancies'].append(
                f"Human: expected {expected_human:,}, got {actual_human:,} (diff: {actual_human - expected_human:+,})"
            )

        return results


def run_sample_audit(retraction_path, problematic_path, output_dir,
                     expected_total=None, expected_ai=None, expected_human=None,
                     verbose=True):
    """
    Convenience function to run complete sample audit.

    Parameters
    ----------
    retraction_path : str
    problematic_path : str
    output_dir : str
    expected_total, expected_ai, expected_human : int, optional
        Expected values for consistency check
    verbose : bool

    Returns
    -------
    dict with audit results and output paths
    """
    audit = SampleConstructionAudit(retraction_path, problematic_path)
    df = audit.load_and_process(verbose=verbose)
    outputs = audit.generate_full_report(output_dir, verbose=verbose)

    results = {
        'final_df': df,
        'outputs': outputs,
        'n_total': len(df),
        'n_ai': df['is_ai'].sum(),
        'n_human': len(df) - df['is_ai'].sum()
    }

    if any(x is not None for x in [expected_total, expected_ai, expected_human]):
        verification = audit.verify_consistency(expected_total, expected_ai, expected_human)
        results['verification'] = verification

        if verbose:
            print("\n" + "="*70)
            print("CONSISTENCY VERIFICATION")
            print("="*70)
            if verification['checks_passed']:
                print("✓ All consistency checks PASSED")
            else:
                print("✗ Consistency checks FAILED:")
                for disc in verification['discrepancies']:
                    print(f"  - {disc}")

    return results
