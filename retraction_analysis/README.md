# Retraction Analysis Package

Survival analysis of AI-contaminated papers in the Retraction Watch database.

## Installation

```bash
pip install pandas numpy matplotlib seaborn lifelines scipy
```

## Quick Start

```python
from run_all import run_all_analyses

results = run_all_analyses(
    'retraction_watch.csv',
    'problematic_papers.csv',
    output_dir='output'
)
```

## Or step by step:

```python
from src import load_data, define_ai_cohorts
from analysis import kaplan_meier_analysis, cox_regression

# Load data
rw_df, prob_df = load_data('retraction_watch.csv', 'problematic_papers.csv')

# Define AI vs Human cohorts
df = define_ai_cohorts(rw_df, prob_df)

# Kaplan-Meier survival analysis
km_results = kaplan_meier_analysis(df, save_path='figures/km.png')

# Cox regression
cox_results = cox_regression(df)
```

## Output Structure

```
output/
├── figures/
│   ├── fig_kaplan_meier.png
│   ├── fig_hazard_by_era.png
│   ├── fig_reasons.png
│   ├── fig_subjects.png
│   └── fig_temporal.png
└── tables/
    ├── table_cohort_summary.tex
    ├── table_survival.tex
    ├── table_era_hazard.tex
    └── table_reasons.tex
```

## Key Analyses

1. **Kaplan-Meier**: Time-to-retraction survival curves
2. **Cox Regression**: Hazard ratio (AI vs Human)
3. **Era Analysis**: How detection difficulty changed over time
4. **Reason Analysis**: Retraction reasons by cohort
5. **Subject Analysis**: Subject area distribution
6. **Temporal Trends**: AI contamination over time

## Interpretation

- **Hazard Ratio < 1**: AI papers take LONGER to be retracted (harder to detect)
- **Hazard Ratio > 1**: AI papers are retracted FASTER (easier to detect)
- **GIGO Window**: "Garbage In, Garbage Out" - time paper circulated before retraction
