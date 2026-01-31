"""
Affiliation Enrichment Pipeline for ICLR Data.
===============================================
Enriches ICLR submission data with author affiliations and reputation metrics.

Main pipeline:
1. Search for each paper title in OpenAlex (affiliations + optional h-index)
2. Extract ALL author affiliations (country, university)
3. Fetch author reputation metrics using priority: OpenAlex > Scopus > Semantic Scholar
4. Compute aggregated metrics (mean, median, max h-index across all authors)
5. Return enriched DataFrame with both individual and aggregated metrics

This version supports:
- Processing ALL authors (not just first/last)
- Aggregated h-index metrics (mean, median, max, min)
- Better country detection (tries all authors)
- Multiple API sources for robustness

Usage:
    enricher = AffiliationEnricher(
        email="your@email.com",
        elsevier_api_key="your-key",  # optional
        semantic_scholar_api_key="your-key"  # optional
    )
    enriched_df = enricher.enrich_submissions(
        submissions_df,
        process_all_authors=True  # NEW: get data for all authors
    )
"""

import logging
import os
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple, Any, Literal
from dataclasses import dataclass, asdict, field
from tqdm import tqdm
from enum import Enum
import json

from .openalex_api import OpenAlexClient, PaperMatch, Author
from .semantic_scholar_api import SemanticScholarClient, AuthorMetrics
from .elsevier_api import ElsevierClient, ScopusAuthorMetrics

logger = logging.getLogger(__name__)


class HIndexSource(Enum):
    """Source of h-index data."""
    OPENALEX = "openalex"
    SCOPUS = "scopus"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    NONE = "none"


@dataclass
class AuthorAffiliationResult:
    """Result of affiliation lookup for an author."""
    name: str
    position: int  # 0-indexed position in author list
    is_first: bool = False
    is_last: bool = False
    country: Optional[str] = None
    country_code: Optional[str] = None
    university: Optional[str] = None
    institution_type: Optional[str] = None
    raw_affiliation: Optional[str] = None
    openalex_id: Optional[str] = None

    # Reputation metrics
    h_index: Optional[int] = None
    citation_count: Optional[int] = None
    paper_count: Optional[int] = None
    h_index_source: str = "none"  # Track where h-index came from


@dataclass
class PaperAffiliationResult:
    """Complete affiliation result for a paper."""
    title: str
    matched: bool
    match_score: float = 0.0
    openalex_id: Optional[str] = None
    doi: Optional[str] = None
    cited_by_count: int = 0

    # All authors
    all_authors: List[AuthorAffiliationResult] = field(default_factory=list)

    # Legacy support - first and last author references
    first_author: Optional[AuthorAffiliationResult] = None
    last_author: Optional[AuthorAffiliationResult] = None

    # Aggregated metrics
    mean_h_index: Optional[float] = None
    median_h_index: Optional[float] = None
    max_h_index: Optional[int] = None
    min_h_index: Optional[int] = None
    n_authors: int = 0
    n_authors_with_h_index: int = 0

    # Primary country (most common or first author's)
    primary_country: Optional[str] = None
    primary_country_code: Optional[str] = None
    all_countries: List[str] = field(default_factory=list)

    # Errors/warnings
    error: Optional[str] = None


class AffiliationEnricher:
    """
    Enriches ICLR submission data with author affiliations and reputation metrics.

    Supports:
    - Processing ALL authors (not just first/last)
    - Multiple API sources with automatic fallback
    - Aggregated metrics (mean, median, max h-index)
    - Better country detection from any author

    Usage:
        # Basic usage (OpenAlex only)
        enricher = AffiliationEnricher(email="your@email.com")

        # With all APIs for best h-index coverage
        enricher = AffiliationEnricher(
            email="your@email.com",
            elsevier_api_key="your-elsevier-key",
            semantic_scholar_api_key="your-s2-key"
        )

        # Enrich a DataFrame with ALL author metrics
        enriched_df = enricher.enrich_submissions(
            submissions_df,
            title_col='title',
            process_all_authors=True,  # Get h-index for all authors
            include_reputation=True,
            h_index_source='auto'
        )
    """

    # Standard column names for output
    OUTPUT_COLUMNS = {
        'first_author': [
            'first_author_name',
            'first_author_country',
            'first_author_country_code',
            'first_author_university',
            'first_author_institution_type',
            'first_author_h_index',
            'first_author_h_index_source',
            'first_author_citation_count',
            'first_author_paper_count',
        ],
        'last_author': [
            'last_author_name',
            'last_author_country',
            'last_author_country_code',
            'last_author_university',
            'last_author_institution_type',
            'last_author_h_index',
            'last_author_h_index_source',
            'last_author_citation_count',
            'last_author_paper_count',
        ],
        'aggregated': [
            'n_authors',
            'n_authors_with_h_index',
            'mean_author_h_index',
            'median_author_h_index',
            'max_author_h_index',
            'min_author_h_index',
            'primary_country',
            'primary_country_code',
            'all_author_countries',
        ],
        'paper': [
            'openalex_match_score',
            'openalex_id',
            'doi',
            'paper_cited_by_count',
        ]
    }

    def __init__(
        self,
        email: Optional[str] = None,
        elsevier_api_key: Optional[str] = None,
        semantic_scholar_api_key: Optional[str] = None,
        cache_results: bool = True
    ):
        """
        Initialize affiliation enricher.

        Parameters
        ----------
        email : str, optional
            Email for OpenAlex polite pool (recommended for 10x rate limits)
        elsevier_api_key : str, optional
            Elsevier/Scopus API key (or set ELSEVIER_API_KEY env var)
        semantic_scholar_api_key : str, optional
            Semantic Scholar API key (or set SEMANTIC_SCHOLAR env var)
        cache_results : bool
            Whether to cache results to avoid duplicate lookups
        """
        # Initialize API clients
        self.openalex = OpenAlexClient(email=email)

        # Elsevier/Scopus - check env var if not provided
        elsevier_key = elsevier_api_key or os.environ.get('ELSEVIER_API_KEY')
        self.elsevier = ElsevierClient(api_key=elsevier_key)

        # Semantic Scholar - check env var if not provided
        s2_key = semantic_scholar_api_key or os.environ.get('SEMANTIC_SCHOLAR')
        self.semantic_scholar = SemanticScholarClient(api_key=s2_key)

        self.cache_results = cache_results
        self._cache: Dict[str, PaperAffiliationResult] = {}

        # Log available APIs
        apis = ['OpenAlex']
        if self.elsevier.is_available:
            apis.append('Scopus/Elsevier')
        if self.semantic_scholar.api_key:
            apis.append('Semantic Scholar')
        logger.info(f"Initialized enricher with APIs: {', '.join(apis)}")

    def lookup_paper(
        self,
        title: str,
        year: Optional[int] = None,
        include_reputation: bool = True,
        process_all_authors: bool = True,
        max_authors_for_h_index: int = 10,
        min_match_score: float = 0.7,
        h_index_source: Literal['auto', 'openalex', 'scopus', 'semantic_scholar'] = 'auto'
    ) -> PaperAffiliationResult:
        """
        Look up affiliations for a single paper.

        Parameters
        ----------
        title : str
            Paper title
        year : int, optional
            Publication year (helps with matching)
        include_reputation : bool
            Whether to fetch h-index/citations
        process_all_authors : bool
            If True, get h-index for ALL authors. If False, only first/last.
        max_authors_for_h_index : int
            Maximum number of authors to fetch h-index for (to limit API calls)
        min_match_score : float
            Minimum title match score to accept
        h_index_source : str
            Where to get h-index: 'auto' (try all), 'openalex', 'scopus', 'semantic_scholar'

        Returns
        -------
        PaperAffiliationResult
        """
        # Check cache
        cache_key = f"{title}:{year}:{h_index_source}:{process_all_authors}"
        if self.cache_results and cache_key in self._cache:
            return self._cache[cache_key]

        result = PaperAffiliationResult(title=title, matched=False)

        try:
            # Search OpenAlex for paper and affiliations
            match = self.openalex.search_by_title(
                title, year=year, min_score=min_match_score
            )

            if not match:
                result.error = "No match found in OpenAlex"
                return self._cache_and_return(cache_key, result)

            result.matched = True
            result.match_score = match.match_score
            result.openalex_id = match.openalex_id
            result.doi = match.doi
            result.cited_by_count = match.cited_by_count
            result.n_authors = len(match.authors)

            # Extract ALL authors
            for author in match.authors:
                author_result = self._extract_author_affiliation(author)
                result.all_authors.append(author_result)

                if author.is_first:
                    result.first_author = author_result
                if author.is_last:
                    result.last_author = author_result

            # Handle single-author papers
            if result.n_authors == 1 and result.first_author:
                result.last_author = result.first_author

            # Fetch reputation metrics (h-index) for authors
            if include_reputation:
                if process_all_authors:
                    self._add_all_author_metrics(
                        result, title, h_index_source, max_authors_for_h_index
                    )
                else:
                    # Legacy: only first/last
                    self._add_reputation_metrics(result, title, h_index_source)

            # Compute aggregated metrics
            self._compute_aggregated_metrics(result)

            # Determine primary country
            self._determine_primary_country(result)

        except Exception as e:
            logger.error(f"Error processing '{title[:50]}...': {e}")
            result.error = str(e)

        return self._cache_and_return(cache_key, result)

    def _cache_and_return(
        self,
        key: str,
        result: PaperAffiliationResult
    ) -> PaperAffiliationResult:
        """Cache and return result."""
        if self.cache_results:
            self._cache[key] = result
        return result

    def _extract_author_affiliation(self, author: Author) -> AuthorAffiliationResult:
        """Extract affiliation info from OpenAlex author."""
        inst = author.primary_institution

        return AuthorAffiliationResult(
            name=author.name,
            position=author.position,
            is_first=author.is_first,
            is_last=author.is_last,
            country=inst.country if inst else None,
            country_code=inst.country_code if inst else None,
            university=inst.name if inst else None,
            institution_type=inst.type if inst else None,
            raw_affiliation=author.raw_affiliation,
            openalex_id=author.openalex_id
        )

    def _add_all_author_metrics(
        self,
        result: PaperAffiliationResult,
        title: str,
        h_index_source: str = 'auto',
        max_authors: int = 10
    ):
        """
        Add h-index and citation metrics for ALL authors (up to max_authors).
        """
        authors_to_process = result.all_authors[:max_authors]

        for i, author in enumerate(authors_to_process):
            self._fetch_author_h_index(
                author, title, f'position_{author.position}', h_index_source
            )

    def _add_reputation_metrics(
        self,
        result: PaperAffiliationResult,
        title: str,
        h_index_source: str = 'auto'
    ):
        """
        Add h-index and citation metrics (legacy: only first/last authors).
        """
        # Process first author
        if result.first_author:
            self._fetch_author_h_index(
                result.first_author, title, 'first', h_index_source
            )

        # Process last author
        if result.last_author and result.last_author != result.first_author:
            self._fetch_author_h_index(
                result.last_author, title, 'last', h_index_source
            )

    def _fetch_author_h_index(
        self,
        author: AuthorAffiliationResult,
        title: str,
        position: str,
        source: str
    ):
        """Fetch h-index for a single author using specified source(s)."""

        # Determine sources to try
        if source == 'auto':
            sources = ['openalex', 'scopus']
            if self.semantic_scholar.api_key and self.semantic_scholar.is_available:
                sources.append('semantic_scholar')
        else:
            sources = [source]

        for src in sources:
            if author.h_index is not None:
                break  # Already got h-index

            try:
                if src == 'openalex':
                    h_index = self._get_h_index_openalex(author.openalex_id)
                    if h_index is not None:
                        author.h_index = h_index
                        author.h_index_source = 'openalex'
                        # Also get citation/paper counts AND fill in missing affiliation
                        details = self.openalex.get_author_details(author.openalex_id)
                        if details:
                            author.citation_count = details.get('cited_by_count')
                            author.paper_count = details.get('works_count')
                            # Fill in missing country/university from author profile
                            institutions = details.get('last_known_institutions', [])
                            if institutions and len(institutions) > 0:
                                inst = institutions[0]
                                if author.country is None:
                                    author.country = inst.get('country')
                                if author.country_code is None:
                                    author.country_code = inst.get('country_code')
                                if author.university is None:
                                    author.university = inst.get('display_name') or inst.get('name')
                                if author.institution_type is None:
                                    author.institution_type = inst.get('type')
                        return

                elif src == 'scopus' and self.elsevier.is_available:
                    # Scopus lookup is more expensive, only do for first/last
                    if position in ['first', 'last', 'position_0']:
                        metrics = self.elsevier.get_author_metrics_from_paper(
                            title, author_position='first' if position == 'position_0' else position
                        )
                        if metrics and metrics.h_index is not None:
                            author.h_index = metrics.h_index
                            author.h_index_source = 'scopus'
                            author.citation_count = metrics.citation_count
                            author.paper_count = metrics.document_count
                            return

                elif src == 'semantic_scholar' and self.semantic_scholar.is_available:
                    if author.h_index is not None:
                        return
                    # Semantic Scholar lookup - only for first/last authors
                    if position in ['first', 'last', 'position_0']:
                        metrics = self.semantic_scholar.get_author_metrics_from_paper(
                            title, author_position='first' if position == 'position_0' else position
                        )
                        if metrics and metrics.h_index is not None:
                            author.h_index = metrics.h_index
                            author.h_index_source = 'semantic_scholar'
                            author.citation_count = metrics.citation_count
                            author.paper_count = metrics.paper_count
                            return

            except Exception as e:
                logger.debug(f"Failed to get h-index from {src}: {e}")
                continue

    def _get_h_index_openalex(self, openalex_id: Optional[str]) -> Optional[int]:
        """Get h-index from OpenAlex author endpoint."""
        if not openalex_id:
            return None
        return self.openalex.get_author_h_index(openalex_id)

    def _compute_aggregated_metrics(self, result: PaperAffiliationResult):
        """Compute aggregated h-index metrics across all authors."""
        h_indices = [
            a.h_index for a in result.all_authors
            if a.h_index is not None
        ]

        result.n_authors_with_h_index = len(h_indices)

        if h_indices:
            result.mean_h_index = np.mean(h_indices)
            result.median_h_index = np.median(h_indices)
            result.max_h_index = int(max(h_indices))
            result.min_h_index = int(min(h_indices))

    def _determine_primary_country(self, result: PaperAffiliationResult):
        """
        Determine the primary country for the paper.

        Strategy:
        1. Use first author's country if available
        2. Otherwise, use the most common country among all authors
        3. Collect all unique countries for reference
        """
        # Collect all countries
        countries = []
        country_codes = []

        for author in result.all_authors:
            if author.country:
                countries.append(author.country)
            if author.country_code:
                country_codes.append(author.country_code)

        result.all_countries = list(set(countries))

        # Primary country: prefer first author
        if result.first_author and result.first_author.country:
            result.primary_country = result.first_author.country
            result.primary_country_code = result.first_author.country_code
        elif countries:
            # Fall back to most common
            from collections import Counter
            country_counts = Counter(countries)
            result.primary_country = country_counts.most_common(1)[0][0]
            # Find corresponding code
            for author in result.all_authors:
                if author.country == result.primary_country:
                    result.primary_country_code = author.country_code
                    break

    def enrich_submissions(
        self,
        df: pd.DataFrame,
        title_col: str = 'title',
        year_col: Optional[str] = None,
        include_reputation: bool = True,
        process_all_authors: bool = True,
        max_authors_for_h_index: int = 10,
        min_match_score: float = 0.7,
        h_index_source: Literal['auto', 'openalex', 'scopus', 'semantic_scholar'] = 'auto',
        progress_bar: bool = True,
        batch_size: int = 100
    ) -> pd.DataFrame:
        """
        Enrich a submissions DataFrame with author affiliations.

        Parameters
        ----------
        df : DataFrame
            Submissions data with paper titles
        title_col : str
            Name of column containing paper titles
        year_col : str, optional
            Name of column containing publication years
        include_reputation : bool
            Whether to include h-index/citations
        process_all_authors : bool
            If True, get h-index for ALL authors. If False, only first/last.
        max_authors_for_h_index : int
            Maximum number of authors to fetch h-index for per paper
        min_match_score : float
            Minimum title match score
        h_index_source : str
            Where to get h-index: 'auto', 'openalex', 'scopus', 'semantic_scholar'
        progress_bar : bool
            Whether to show progress bar
        batch_size : int
            Number of papers per progress update

        Returns
        -------
        DataFrame with additional affiliation columns
        """
        if title_col not in df.columns:
            raise ValueError(f"Column '{title_col}' not found in DataFrame")

        # Log configuration
        logger.info(f"Starting enrichment of {len(df)} papers")
        logger.info(f"H-index source: {h_index_source}")
        logger.info(f"Process all authors: {process_all_authors}")
        if h_index_source == 'auto':
            sources = ['OpenAlex']
            if self.elsevier.is_available:
                sources.append('Scopus')
            if self.semantic_scholar.api_key:
                sources.append('Semantic Scholar')
            logger.info(f"Available h-index sources: {', '.join(sources)}")

        # Prepare output
        results = []
        iterator = df.iterrows()

        if progress_bar:
            iterator = tqdm(iterator, total=len(df), desc="Enriching papers")

        for idx, row in iterator:
            title = row[title_col]
            year = row.get(year_col) if year_col else None

            # Skip if no title
            if pd.isna(title) or not str(title).strip():
                results.append(self._empty_result_dict(process_all_authors))
                continue

            # Lookup affiliations
            result = self.lookup_paper(
                str(title),
                year=int(year) if year and not pd.isna(year) else None,
                include_reputation=include_reputation,
                process_all_authors=process_all_authors,
                max_authors_for_h_index=max_authors_for_h_index,
                min_match_score=min_match_score,
                h_index_source=h_index_source
            )

            results.append(self._result_to_dict(result, process_all_authors))

        # Create result DataFrame
        result_df = pd.DataFrame(results)

        # Combine with original
        enriched = pd.concat([df.reset_index(drop=True), result_df], axis=1)

        # Log summary
        n_matched = (enriched['openalex_match_score'] > 0).sum()
        n_with_h_index = enriched['first_author_h_index'].notna().sum()
        n_with_country = enriched['primary_country'].notna().sum() if 'primary_country' in enriched.columns else 0

        logger.info(f"Enriched {n_matched}/{len(df)} papers ({100*n_matched/len(df):.1f}%)")
        logger.info(f"H-index coverage: {n_with_h_index}/{len(df)} ({100*n_with_h_index/len(df):.1f}%)")
        logger.info(f"Country coverage: {n_with_country}/{len(df)} ({100*n_with_country/len(df):.1f}%)")

        # Log h-index source distribution
        if 'first_author_h_index_source' in enriched.columns:
            sources = enriched['first_author_h_index_source'].value_counts()
            logger.info(f"H-index sources: {sources.to_dict()}")

        return enriched

    def _result_to_dict(
        self,
        result: PaperAffiliationResult,
        include_aggregated: bool = True
    ) -> Dict[str, Any]:
        """Convert PaperAffiliationResult to flat dictionary."""
        d = {
            'openalex_match_score': result.match_score if result.matched else np.nan,
            'openalex_id': result.openalex_id,
            'doi': result.doi,
            'paper_cited_by_count': result.cited_by_count if result.matched else np.nan,
            'affiliation_lookup_error': result.error,
        }

        # First author
        if result.first_author:
            fa = result.first_author
            d.update({
                'first_author_name': fa.name,
                'first_author_country': fa.country,
                'first_author_country_code': fa.country_code,
                'first_author_university': fa.university,
                'first_author_institution_type': fa.institution_type,
                'first_author_h_index': fa.h_index,
                'first_author_h_index_source': fa.h_index_source,
                'first_author_citation_count': fa.citation_count,
                'first_author_paper_count': fa.paper_count,
            })
        else:
            d.update({k: None for k in self.OUTPUT_COLUMNS['first_author']})

        # Last author
        if result.last_author:
            la = result.last_author
            d.update({
                'last_author_name': la.name,
                'last_author_country': la.country,
                'last_author_country_code': la.country_code,
                'last_author_university': la.university,
                'last_author_institution_type': la.institution_type,
                'last_author_h_index': la.h_index,
                'last_author_h_index_source': la.h_index_source,
                'last_author_citation_count': la.citation_count,
                'last_author_paper_count': la.paper_count,
            })
        else:
            d.update({k: None for k in self.OUTPUT_COLUMNS['last_author']})

        # Aggregated metrics (always include for convenience)
        d.update({
            'n_authors': result.n_authors,
            'n_authors_with_h_index': result.n_authors_with_h_index,
            'mean_author_h_index': result.mean_h_index,
            'median_author_h_index': result.median_h_index,
            'max_author_h_index': result.max_h_index,
            'min_author_h_index': result.min_h_index,
            'primary_country': result.primary_country,
            'primary_country_code': result.primary_country_code,
            'all_author_countries': json.dumps(result.all_countries) if result.all_countries else None,
        })

        # Store all author h-indices as JSON for later analysis
        if include_aggregated and result.all_authors:
            all_h_indices = [a.h_index for a in result.all_authors]
            d['all_author_h_indices'] = json.dumps(all_h_indices)

        return d

    def _empty_result_dict(self, include_aggregated: bool = True) -> Dict[str, Any]:
        """Return dict with all NaN values."""
        d = {
            'openalex_match_score': np.nan,
            'openalex_id': None,
            'doi': None,
            'paper_cited_by_count': np.nan,
            'affiliation_lookup_error': 'No title provided',
        }
        d.update({k: None for k in self.OUTPUT_COLUMNS['first_author']})
        d.update({k: None for k in self.OUTPUT_COLUMNS['last_author']})
        d.update({k: None for k in self.OUTPUT_COLUMNS['aggregated']})
        d['all_author_h_indices'] = None
        return d

    def get_summary_stats(self, enriched_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute summary statistics for enriched data.
        """
        stats = {}

        # Match rate
        matched = enriched_df['openalex_match_score'].notna()
        stats['n_total'] = len(enriched_df)
        stats['n_matched'] = matched.sum()
        stats['match_rate'] = matched.mean()
        stats['avg_match_score'] = enriched_df.loc[matched, 'openalex_match_score'].mean()

        # H-index coverage
        stats['n_with_h_index'] = enriched_df['first_author_h_index'].notna().sum()
        stats['h_index_coverage'] = stats['n_with_h_index'] / stats['n_total']

        # Country coverage
        if 'primary_country' in enriched_df.columns:
            stats['n_with_country'] = enriched_df['primary_country'].notna().sum()
            stats['country_coverage'] = stats['n_with_country'] / stats['n_total']

        # H-index source distribution
        if 'first_author_h_index_source' in enriched_df.columns:
            stats['h_index_sources'] = enriched_df['first_author_h_index_source'].value_counts().to_dict()

        # Aggregated h-index stats
        for col in ['mean_author_h_index', 'median_author_h_index', 'max_author_h_index']:
            if col in enriched_df.columns:
                values = enriched_df[col].dropna()
                if len(values) > 0:
                    stats[f'{col}_mean'] = values.mean()
                    stats[f'{col}_std'] = values.std()

        # Country distribution
        for prefix in ['primary', 'first_author', 'last_author']:
            country_col = f'{prefix}_country'
            if country_col in enriched_df.columns:
                countries = enriched_df[country_col].value_counts()
                stats[f'{prefix}_top_countries'] = countries.head(10).to_dict()
                stats[f'{prefix}_n_countries'] = countries.count()

        return stats

    def clear_cache(self):
        """Clear the results cache."""
        self._cache.clear()


def enrich_iclr_data(
    submissions_path: str,
    output_path: str,
    email: Optional[str] = None,
    elsevier_api_key: Optional[str] = None,
    semantic_scholar_key: Optional[str] = None,
    title_col: str = 'title',
    include_reputation: bool = True,
    process_all_authors: bool = True,
    max_authors_for_h_index: int = 10,
    h_index_source: str = 'auto'
) -> pd.DataFrame:
    """
    Convenience function to enrich ICLR data from file.

    Parameters
    ----------
    submissions_path : str
        Path to submissions CSV/Excel file
    output_path : str
        Path to save enriched data
    email : str, optional
        Email for OpenAlex API
    elsevier_api_key : str, optional
        Elsevier/Scopus API key
    semantic_scholar_key : str, optional
        API key for Semantic Scholar
    title_col : str
        Column containing paper titles
    include_reputation : bool
        Whether to include author metrics
    process_all_authors : bool
        If True, get h-index for ALL authors
    max_authors_for_h_index : int
        Maximum authors to get h-index for
    h_index_source : str
        'auto', 'openalex', 'scopus', or 'semantic_scholar'

    Returns
    -------
    Enriched DataFrame (also saved to output_path)
    """
    # Load data
    if submissions_path.endswith('.csv'):
        df = pd.read_csv(submissions_path)
    else:
        df = pd.read_excel(submissions_path)

    print(f"Loaded {len(df)} submissions")

    # Enrich
    enricher = AffiliationEnricher(
        email=email,
        elsevier_api_key=elsevier_api_key,
        semantic_scholar_api_key=semantic_scholar_key
    )

    enriched = enricher.enrich_submissions(
        df,
        title_col=title_col,
        include_reputation=include_reputation,
        process_all_authors=process_all_authors,
        max_authors_for_h_index=max_authors_for_h_index,
        h_index_source=h_index_source
    )

    # Save
    if output_path.endswith('.csv'):
        enriched.to_csv(output_path, index=False)
    else:
        enriched.to_excel(output_path, index=False)

    print(f"Saved enriched data to {output_path}")

    # Print summary
    stats = enricher.get_summary_stats(enriched)
    print(f"\nSummary:")
    print(f"  Matched: {stats['n_matched']}/{stats['n_total']} ({100*stats['match_rate']:.1f}%)")
    print(f"  H-index coverage: {stats['n_with_h_index']}/{stats['n_total']} ({100*stats['h_index_coverage']:.1f}%)")

    if 'country_coverage' in stats:
        print(f"  Country coverage: {stats['n_with_country']}/{stats['n_total']} ({100*stats['country_coverage']:.1f}%)")

    if 'h_index_sources' in stats:
        print(f"\n  H-index sources: {stats['h_index_sources']}")

    if 'primary_top_countries' in stats:
        print(f"\n  Top countries:")
        for country, count in list(stats['primary_top_countries'].items())[:5]:
            print(f"    {country}: {count}")

    if 'mean_author_h_index_mean' in stats:
        print(f"\n  Mean author h-index (avg): {stats['mean_author_h_index_mean']:.1f}")
        print(f"  Max author h-index (avg): {stats.get('max_author_h_index_mean', 0):.1f}")

    return enriched
