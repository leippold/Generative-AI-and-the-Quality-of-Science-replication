"""
Constants and configuration for ICLR AI Contamination Analysis.
===============================================================
Centralized definitions for categories, colors, thresholds, and defaults.

IMPORTANT - AI VARIABLE NOTATION:
---------------------------------
Throughout this codebase, AI content is measured on a 0-100 PERCENTAGE scale:
  - ai_percentage ∈ [0, 100] (percentage points)
  - NOT ai_percentage ∈ [0, 1] (proportion)

For example:
  - ai_percentage = 75 means 75% AI content
  - ai_percentage = 0 means fully human-written

When converting for regression coefficients:
  - A coefficient of -0.004 means -0.4 points per 100% AI increase
  - To report "per 100% AI", multiply coefficient by 100

The data_loading.py clean_ai_percentage() function handles string formats
like "75%" by converting to numeric 75.0.
"""

# =============================================================================
# REVIEW CLASSIFICATION
# =============================================================================

REVIEW_CATEGORIES = [
    'Fully human-written',
    'Lightly AI-edited',
    'Moderately AI-edited',
    'Heavily AI-edited',
    'Fully AI-generated'
]

REVIEW_COLORS = {
    'Fully human-written': '#2ca02c',
    'Lightly AI-edited': '#98df8a',
    'Moderately AI-edited': '#ffbb78',
    'Heavily AI-edited': '#ff7f0e',
    'Fully AI-generated': '#d62728'
}

REVIEW_BINARY = {
    'Fully human-written': 'Human',
    'Fully AI-generated': 'AI'
}

# =============================================================================
# AI CONTENT BINS
# =============================================================================

AI_CONTENT_BINS = [-0.001, 10, 25, 50, 75, 100.001]
AI_CONTENT_LABELS = ['0-10%', '10-25%', '25-50%', '50-75%', '75-100%']

AI_FINE_BINS = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
AI_FINE_LABELS = ['0-5%', '5-10%', '10-15%', '15-20%', '20-25%', '25-30%',
                  '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']

# =============================================================================
# ECHO CHAMBER THRESHOLDS
# =============================================================================

DEFAULT_AI_PAPER_THRESHOLD = 75
DEFAULT_HUMAN_PAPER_THRESHOLD = 25

ROBUSTNESS_THRESHOLDS = [
    (75, 25, "Baseline"),
    (90, 10, "Strict"),
    (100, 0, "Extreme"),
    (50, 50, "Lenient"),
    (80, 20, "Alternative")
]

ECHO_CHAMBER_COLORS = {
    'Human Paper + Human Review': '#2ca02c',
    'Human Paper + AI Review': '#98df8a',
    'AI Paper + Human Review': '#ff9896',
    'AI Paper + AI Review': '#d62728'
}

PAPER_TYPE_COLORS = {
    'Human Paper': '#2ca02c',
    'AI Paper': '#d62728',
    'Mixed': '#ff7f0e'
}

# =============================================================================
# STATISTICAL DEFAULTS
# =============================================================================

ALPHA = 0.05
N_BOOTSTRAP = 10000
N_PERMUTATIONS = 10000
RANDOM_SEED = 42

MIN_CELL_SIZE = 20
MIN_GROUP_SIZE = 30
MIN_WITHIN_PAPER = 10

# =============================================================================
# REVIEW METRICS
# =============================================================================

REVIEW_METRICS = ['rating', 'confidence', 'soundness', 'presentation', 'contribution']
COMPONENT_METRICS = ['soundness', 'presentation', 'contribution']

RATING_MIN = 1
RATING_MAX = 10

# =============================================================================
# PLOTTING
# =============================================================================

FIGURE_DPI = 300
FIGURE_STYLE = 'seaborn-v0_8-whitegrid'
QUALITY_COLORS = ['#d62728', '#ff7f0e', '#2ca02c']
AI_GRADIENT = ['#2ca02c', '#7cb342', '#fdd835', '#ff9800', '#d32f2f']
