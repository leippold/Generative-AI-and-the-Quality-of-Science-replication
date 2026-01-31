"""
ICLR AI Contamination Analysis - Source Utilities
=================================================

Consolidated, statistically rigorous analysis package.

Usage:
------
from src import load_data, mann_whitney_test, plot_heatmap
from src.constants import REVIEW_CATEGORIES, ALPHA
"""

from .constants import *
from .data_loading import (
    load_data,
    create_ai_categories,
    merge_paper_info,
    classify_papers,
    classify_reviewers,
    prepare_echo_chamber_data,
    get_cell_data,
    compute_sample_summary
)
from .stats_utils import (
    # Effect sizes
    cohens_d,
    hedges_g,
    rank_biserial_correlation,
    epsilon_squared,
    cliffs_delta,
    
    # Basic tests
    mann_whitney_test,
    kruskal_wallis_test,
    chi_square_test,
    levene_test,
    
    # Bootstrap
    bootstrap_ci,
    bootstrap_diff_ci,
    
    # Permutation tests
    permutation_test_two_groups,
    permutation_test_interaction,
    
    # Multiple comparisons
    bonferroni_correction,
    fdr_correction,
    pairwise_comparisons,
    
    # Regression
    ols_with_clustered_se,
    ols_robust,
    ordinal_regression,
    mixed_effects_model,
    compute_icc,
    
    # Confidence-weighted
    weighted_mean,
    weighted_std,
    paper_weighted_rating,
    confident_leniency_index,
    
    # Comprehensive
    comprehensive_comparison,
    format_results_table
)
from .plotting import (
    setup_style,
    save_figure,
    plot_interaction_2x2,
    plot_heatmap,
    plot_grouped_bars,
    plot_distribution_comparison,
    plot_effect_sizes,
    plot_permutation_distribution,
    plot_dose_response,
    create_summary_figure
)
from .openreview_api import (
    fetch_iclr_decisions,
    merge_acceptance_data,
    load_or_fetch_decisions,
    check_decision_availability,
    extract_forum_id
)

__version__ = '2.1.0'
__author__ = 'ICLR Analysis Team'
