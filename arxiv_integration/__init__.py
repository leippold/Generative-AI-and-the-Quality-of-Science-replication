"""
ArXiv Integration Module for ICLR Analysis.
============================================
Enriches ICLR submission data with author affiliations and reputation metrics.

Data sources (in priority order for h-index):
1. OpenAlex: Paper metadata, author affiliations, institutions, h-index (FREE, FAST)
2. Scopus/Elsevier: Reliable h-index, citations (requires API key)
3. Semantic Scholar: Author h-index, citations (slower, fallback)

Usage:
    from arxiv_integration import AffiliationEnricher

    # Initialize with your API keys for best coverage
    enricher = AffiliationEnricher(
        email="your@email.com",               # OpenAlex polite pool (10x rate limit)
        elsevier_api_key="your-scopus-key",   # Optional: better h-index coverage
        semantic_scholar_api_key="your-key"   # Optional: fallback
    )

    # Or use environment variables (ELSEVIER_API_KEY, SEMANTIC_SCHOLAR)
    enricher = AffiliationEnricher(email="your@email.com")

    # Enrich ICLR submissions DataFrame
    enriched_df = enricher.enrich_submissions(
        submissions_df,
        title_col='title',
        include_reputation=True,
        h_index_source='auto'  # 'auto', 'openalex', 'scopus', 'semantic_scholar'
    )

    # Add derived columns (region, reputation category, etc.)
    from arxiv_integration import add_derived_columns
    enriched_df = add_derived_columns(enriched_df)

    # Or look up a single paper
    result = enricher.lookup_paper("Attention Is All You Need")
    print(result.first_author.country)  # e.g., "United States"
    print(result.first_author.h_index)  # e.g., 45
"""

from .src.openalex_api import OpenAlexClient
from .src.semantic_scholar_api import SemanticScholarClient
from .src.elsevier_api import ElsevierClient
from .src.affiliation_enrichment import AffiliationEnricher, enrich_iclr_data
from .src.country_utils import (
    standardize_country,
    get_region,
    is_top_university,
    classify_h_index,
    add_derived_columns,
    compute_affiliation_summary,
    COUNTRY_CODE_MAP,
    REGION_MAP,
    TOP_CS_UNIVERSITIES
)
from .src.visualizations import (
    create_country_analysis,
    generate_statistical_summary,
    set_publication_style,
    fix_kaleido_colab,
    save_plotly_figure
)

__all__ = [
    # Main classes
    'OpenAlexClient',
    'SemanticScholarClient',
    'ElsevierClient',
    'AffiliationEnricher',

    # Convenience function
    'enrich_iclr_data',

    # Country utilities
    'standardize_country',
    'get_region',
    'is_top_university',
    'classify_h_index',
    'add_derived_columns',
    'compute_affiliation_summary',

    # Visualizations
    'create_country_analysis',
    'generate_statistical_summary',
    'set_publication_style',
    'fix_kaleido_colab',
    'save_plotly_figure',

    # Reference data
    'COUNTRY_CODE_MAP',
    'REGION_MAP',
    'TOP_CS_UNIVERSITIES',
]
