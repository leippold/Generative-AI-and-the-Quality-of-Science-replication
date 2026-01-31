"""Source modules for arxiv integration."""

from .openalex_api import OpenAlexClient
from .semantic_scholar_api import SemanticScholarClient
from .elsevier_api import ElsevierClient
from .affiliation_enrichment import AffiliationEnricher
from .country_utils import (
    standardize_country,
    get_region,
    is_top_university,
    classify_h_index,
    add_derived_columns,
    compute_affiliation_summary
)

__all__ = [
    'OpenAlexClient',
    'SemanticScholarClient',
    'ElsevierClient',
    'AffiliationEnricher',
    'standardize_country',
    'get_region',
    'is_top_university',
    'classify_h_index',
    'add_derived_columns',
    'compute_affiliation_summary'
]
