"""Analysis modules for arxiv integration."""

from .selection_robustness import (
    run_selection_robustness_analysis,
    merge_author_data,
    create_analysis_variables,
    create_reputation_tiers,
    ols_with_clustered_se,
    cohens_d,
    bootstrap_diff_ci,
    fdr_correction,
    create_interaction_plots,
    create_interaction_plots_regression,
    create_interaction_plots_terciles,
    test_hindex_interactions_robust,
    create_hindex_interaction_figure
)

from .reviewer_robustness_tests import (
    run_reviewer_robustness_tests,
    compute_oster_bounds,
    test_ai_fingerprint,
    test_detector_bias,
    generate_robustness_latex_table,
    OsterBoundsResult,
    DetectorBiasResult,
    ComponentDifferentialResult
)

__all__ = [
    # Selection robustness
    'run_selection_robustness_analysis',
    'merge_author_data',
    'create_analysis_variables',
    'create_reputation_tiers',
    'ols_with_clustered_se',
    'cohens_d',
    'bootstrap_diff_ci',
    'fdr_correction',
    'create_interaction_plots',
    'create_interaction_plots_regression',
    'create_interaction_plots_terciles',
    'test_hindex_interactions_robust',
    'create_hindex_interaction_figure',

    # Reviewer robustness tests
    'run_reviewer_robustness_tests',
    'compute_oster_bounds',
    'test_ai_fingerprint',
    'test_detector_bias',
    'generate_robustness_latex_table',
    'OsterBoundsResult',
    'DetectorBiasResult',
    'ComponentDifferentialResult',
]
