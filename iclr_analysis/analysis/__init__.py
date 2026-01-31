"""ICLR Analysis Modules"""
from .echo_chamber import run_echo_chamber_analysis
from .within_paper import run_within_paper_analysis
from .effort_proxies import run_effort_proxy_analysis, test_substitution_stacked_regression
from .collaboration_hypothesis import run_collaboration_analysis
from .heterogeneity import run_heterogeneity_analysis
from .acceptance_analysis import (
    run_acceptance_analysis,
    ai_acceptance_probit,
    ai_acceptance_by_category,
    threshold_discontinuity,
    optimal_bandwidth_selection,
    presentation_tier_analysis,
    reputation_acceptance_interaction,
    selection_bounds_analysis,
    score_acceptance_residual,
    prepare_acceptance_data
)
