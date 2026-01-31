# Generative-AI-and-the-Quality-of-Science-replication
Replication package for Generative AI and the Quality of Science



his replication package covers the two empirical analyses: the **ICLR peer-review analysis** and the **Retraction Watch survival analysis**. Both build on a master file of ICLR submissions enriched with AI content scores and author metadata.

---

## Data Sources

| Data | Source | Access |
|------|--------|--------|
| ICLR 2026 submissions & reviews | [OpenReview](https://openreview.net/group?id=ICLR.cc/2026/Conference) | Public |
| AI content scores | Pangram Labs | Proprietary* |
| Retraction Watch database | [retractionwatch.com](https://retractionwatch.com) | Public |
| Problematic Paper Screener | [dbrech.irit.fr](https://dbrech.irit.fr/pls/apex/f?p=9999:3) | Public |

\* Contact Pangram Labs for access to AI classification scores. The `ai_percentage` column in the master file is derived from their classifier.

---

## Repository Structure

```
HAI-Frontier/
├── data/
│   └── iclr_submissions_enriched.csv   # Master file (submissions + AI scores + author metadata)
│
├── iclr_analysis/                       # ICLR peer-review analysis
│   ├── src/
│   │   ├── constants.py                 # Thresholds, categories, defaults
│   │   ├── data_loading.py              # Data loading and preparation
│   │   ├── stats_utils.py               # Statistical tests and effect sizes
│   │   ├── plotting.py                  # Visualization utilities
│   │   ├── plotting_enhanced.py         # Enhanced figure generation
│   │   └── openreview_api.py            # Fetch decisions from OpenReview API
│   ├── analysis/
│   │   ├── echo_chamber.py              # 2x2 interaction: AI reviewer x AI paper
│   │   ├── within_paper.py              # Within-paper paired comparisons
│   │   ├── effort_proxies.py            # Substitution signature (presentation vs soundness)
│   │   ├── collaboration_hypothesis.py  # Inverted-U test (moderate AI optimal?)
│   │   ├── heterogeneity.py             # Effects by paper quality tercile
│   │   ├── acceptance_analysis.py       # AI content and acceptance decisions
│   │   ├── confidence.py                # Confidence-weighted analysis
│   │   ├── referee_tests.py             # Referee-requested validation tests
│   │   └── run_all.py                   # Master runner
│   └── generate_tables.py              # LaTeX table generation
│
└── retraction_analysis/                 # Retraction Watch survival analysis
    ├── retraction_src/
    │   └── data_loading.py              # Load and merge retraction data, define cohorts
    ├── retraction_analysis_modules/
    │   ├── survival.py                  # Kaplan-Meier, Cox regression, matched cohorts
    │   ├── descriptive.py               # Retraction reasons, subjects, temporal trends
    │   └── plotting_enhanced.py         # Enhanced survival plots
    └── run_all.py                       # Master runner + LaTeX table generation
```

---

## Replication Instructions

### Part A: ICLR Peer-Review Analysis

This analysis tests whether AI-generated reviews exhibit systematic biases (echo chamber effects, substitution signatures, leniency patterns).

**Input files:**
- `submissions.csv` -- submission-level data with columns: `submission_number`, `ai_percentage`, `avg_rating`, etc. (derived from the master file)
- `reviews.csv` -- review-level data with columns: `submission_number`, `ai_classification`, `rating`, `confidence`, `soundness`, `presentation`, `contribution`

**Run the full pipeline:**

```python
from iclr_analysis.analysis.run_all import run_all

results = run_all(
    submissions_path='data/iclr_submissions_enriched.csv',
    reviews_path='path/to/reviews.csv',
    output_dir='output/iclr'
)
```

**Or from the command line:**

```bash
cd iclr_analysis
python -m analysis.run_all ../data/iclr_submissions_enriched.csv path/to/reviews.csv output/iclr
```

**Individual analyses can also be run separately:**

```python
from iclr_analysis.src.data_loading import load_data
from iclr_analysis.analysis.echo_chamber import run_echo_chamber_analysis
from iclr_analysis.analysis.within_paper import run_within_paper_analysis

submissions_df, reviews_df = load_data('submissions.csv', 'reviews.csv')

echo_results = run_echo_chamber_analysis(submissions_df, reviews_df)
within_results = run_within_paper_analysis(submissions_df, reviews_df)
```

### Part B: Retraction Watch Survival Analysis

This analysis compares time-to-retraction (the "GIGO window") for AI-contaminated vs. human-written papers.

**Input files:**
- `retraction_watch.csv` -- Retraction Watch database export
- `problematic_papers.csv` -- Problematic Paper Screener database export

**Run the full pipeline:**

```python
from retraction_analysis.run_all import run_all_analyses

results = run_all_analyses(
    retraction_path='path/to/retraction_watch.csv',
    problematic_path='path/to/problematic_papers.csv',
    output_dir='output/retraction'
)
```

**Or from the command line:**

```bash
cd retraction_analysis
python run_all.py path/to/retraction_watch.csv path/to/problematic_papers.csv output/retraction
```

**Output:**

```
output/retraction/
├── figures/
│   ├── fig_kaplan_meier.png
│   ├── fig_hazard_by_era.png
│   ├── fig_high_freq_escalation.png
│   ├── fig_matched_cohort.png
│   ├── fig_reasons.png
│   ├── fig_subjects.png
│   ├── fig_temporal.png
│   └── fig_citations.png
└── tables/
    ├── table_cohort_summary.tex
    ├── table_survival.tex
    ├── table_era_hazard.tex
    ├── table_escalation.tex
    ├── table_matched_cohort.tex
    ├── table_reasons.tex
    └── table_citations.tex
```

---

## Requirements

```
Python >= 3.9
```

**Core dependencies:**

```
numpy
pandas
scipy
matplotlib
seaborn
statsmodels
lifelines
```

Install with:

```bash
pip install numpy pandas scipy matplotlib seaborn statsmodels lifelines
```

---

## Statistical Methods

The analyses rely on the following methods (implemented in `iclr_analysis/src/stats_utils.py` and `retraction_analysis/retraction_analysis_modules/survival.py`):

- **Inference:** OLS with clustered standard errors, HC3 robust standard errors, mixed-effects models, ordinal regression
- **Non-parametric tests:** Mann-Whitney, Kruskal-Wallis, Wilcoxon signed-rank, permutation tests (10,000 iterations)
- **Effect sizes:** Cohen's d, Hedges' g, Cliff's delta, rank-biserial correlation, epsilon-squared
- **Bootstrap:** BCa confidence intervals, bootstrap difference CIs (10,000 resamples)
- **Multiple comparisons:** Bonferroni, Holm-Bonferroni, Benjamini-Hochberg FDR
- **Survival analysis:** Kaplan-Meier estimation, Cox proportional hazards, log-rank tests, matched-cohort designs

---

## Citation

```bibtex
@unpublished{leippold2026authoritative,
  title={Generative AI and the Quality of Science},
  author={Leippold, Markus},
  year={2026},
  note={Working paper}
}
```
