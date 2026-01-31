# ICLR AI Contamination Analysis Package v2.0

Consolidated, statistically rigorous analysis of AI-generated content in ICLR peer reviews.

## Installation

```bash
pip install pandas numpy scipy matplotlib seaborn statsmodels --break-system-packages
```

## Quick Start

```python
from src import load_data
from analysis import run_echo_chamber_analysis, run_within_paper_analysis

# Load your data
submissions_df, reviews_df = load_data('submissions.csv', 'reviews.csv')

# Run analyses
echo_results = run_echo_chamber_analysis(submissions_df, reviews_df)
within_results = run_within_paper_analysis(submissions_df, reviews_df)
```

## Package Structure

```
iclr_analysis/
├── src/                          # Core utilities
│   ├── constants.py              # All constants and thresholds
│   ├── data_loading.py           # Data loading and preparation
│   ├── stats_utils.py            # Statistical tests (STATE OF THE ART)
│   └── plotting.py               # Visualization utilities
│
├── analysis/                     # Analysis modules
│   ├── echo_chamber.py           # 2×2 interaction analysis
│   ├── within_paper.py           # Papers with both AI + human reviews
│   ├── effort_proxies.py         # Substitution signature
│   ├── collaboration_hypothesis.py # Inverted-U test
│   ├── heterogeneity.py          # Effects by paper quality
│   └── run_all.py                # Run everything
│
└── README.md
```

## Analyses Included

### 1. Echo Chamber Analysis (`echo_chamber.py`)
Tests whether AI reviewers rate AI papers differently than human reviewers.

**Statistical Methods:**
- OLS with clustered standard errors (by submission)
- Permutation test (10,000 iterations)
- Bootstrap confidence intervals
- Multiple threshold robustness checks

### 2. Within-Paper Comparison (`within_paper.py`)
For papers that received BOTH human and AI reviews, compares ratings.
This is the **most powerful design** as each paper serves as its own control.

**Statistical Methods:**
- Paired t-test / Wilcoxon signed-rank
- Difference-in-differences across paper types
- Mixed effects model with paper random effects

### 3. Effort Proxies (`effort_proxies.py`)
Tests for "substitution signature": AI papers have high presentation but low soundness.

**Statistical Methods:**
- Component score decomposition
- Regression with robust (HC3) standard errors
- FDR correction for multiple comparisons

### 4. Collaboration Hypothesis (`collaboration_hypothesis.py`)
Tests whether moderate AI assistance is optimal (inverted-U relationship).

**Statistical Methods:**
- Quadratic regression
- F-test for quadratic vs linear
- Optimal point estimation

### 5. Heterogeneity Analysis (`heterogeneity.py`)
Tests whether echo chamber effect differs by paper quality.

**Statistical Methods:**
- Three-way interaction (Quality × Paper Type × Reviewer Type)
- Stratified analysis by quality terciles

## Statistical Rigor Features

### Robust Inference
- **Clustered Standard Errors**: Reviews are clustered within submissions
- **HC3 Robust SEs**: Heteroskedasticity-robust for small samples
- **Bootstrap CIs**: Non-parametric confidence intervals
- **Permutation Tests**: Distribution-free hypothesis testing

### Multiple Comparison Corrections
- Bonferroni correction
- FDR (Benjamini-Hochberg) correction

### Effect Sizes
- Cohen's d / Hedges' g (standardized mean difference)
- Rank-biserial correlation (for non-parametric tests)
- Cliff's Delta
- Epsilon-squared (for Kruskal-Wallis)

### Model Selection
- Ordinal regression available for rating outcomes
- Mixed effects models for nested data
- ICC computation to assess clustering

## Example Output

```
================================================================================
AI ECHO CHAMBER ANALYSIS
================================================================================

Sample: 15,234 reviews

Cell counts:
reviewer_type     AI Review  Human Review
paper_type                               
AI Paper               234          1,456
Human Paper          1,023          9,876

Interaction Effect: +0.2341 ***

Tests:
  OLS (clustered SE): p = 0.0023
  Permutation test:   p = 0.0018

✓ SIGNIFICANT INTERACTION DETECTED
  AI reviewers give RELATIVELY HIGHER ratings to AI papers
```

## Citation

If you use this package, please cite:
```
ICLR AI Contamination Analysis Package v2.0
```
