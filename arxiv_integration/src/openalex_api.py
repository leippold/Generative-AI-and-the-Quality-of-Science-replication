"""
OpenAlex API Client for Author Affiliations.
=============================================
Searches papers by title and extracts author institution information.

OpenAlex is free, requires no authentication, and has comprehensive
coverage of academic papers with institutional affiliations.

API Documentation: https://docs.openalex.org/
Rate Limit: 100,000 requests/day (polite pool with email)
"""

import time
import logging
import requests
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from urllib.parse import quote

logger = logging.getLogger(__name__)


@dataclass
class Institution:
    """Represents an academic institution."""
    openalex_id: str
    name: str
    country: Optional[str] = None
    country_code: Optional[str] = None
    type: Optional[str] = None  # education, company, government, etc.
    ror: Optional[str] = None  # Research Organization Registry ID


@dataclass
class Author:
    """Represents an author with affiliations."""
    openalex_id: str
    name: str
    position: int  # 0-indexed position in author list
    is_first: bool = False
    is_last: bool = False
    institutions: List[Institution] = field(default_factory=list)
    raw_affiliation: Optional[str] = None

    @property
    def primary_institution(self) -> Optional[Institution]:
        """Return the first (primary) institution."""
        return self.institutions[0] if self.institutions else None

    @property
    def country(self) -> Optional[str]:
        """Return the country of the primary institution."""
        inst = self.primary_institution
        return inst.country if inst else None

    @property
    def country_code(self) -> Optional[str]:
        """Return the country code of the primary institution."""
        inst = self.primary_institution
        return inst.country_code if inst else None

    @property
    def university(self) -> Optional[str]:
        """Return the name of the primary institution."""
        inst = self.primary_institution
        return inst.name if inst else None


@dataclass
class PaperMatch:
    """Represents a matched paper from OpenAlex."""
    openalex_id: str
    title: str
    doi: Optional[str]
    publication_year: Optional[int]
    authors: List[Author]
    match_score: float  # Title similarity score
    cited_by_count: int = 0

    @property
    def first_author(self) -> Optional[Author]:
        """Return the first author."""
        for author in self.authors:
            if author.is_first:
                return author
        return self.authors[0] if self.authors else None

    @property
    def last_author(self) -> Optional[Author]:
        """Return the last author (often senior author)."""
        for author in self.authors:
            if author.is_last:
                return author
        return self.authors[-1] if self.authors else None


class OpenAlexClient:
    """
    Client for the OpenAlex API.

    Usage:
        client = OpenAlexClient(email="your@email.com")
        result = client.search_by_title("Attention Is All You Need")
        if result:
            print(f"First author: {result.first_author.name}")
            print(f"University: {result.first_author.university}")
            print(f"Country: {result.first_author.country}")
    """

    BASE_URL = "https://api.openalex.org"

    def __init__(
        self,
        email: Optional[str] = None,
        requests_per_second: float = 10.0,
        timeout: int = 30
    ):
        """
        Initialize OpenAlex client.

        Parameters
        ----------
        email : str, optional
            Email for polite pool (higher rate limits, recommended)
        requests_per_second : float
            Rate limit (default: 10 for polite pool)
        timeout : int
            Request timeout in seconds
        """
        self.email = email
        self.timeout = timeout
        self.min_request_interval = 1.0 / requests_per_second
        self._last_request_time = 0.0

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'HAI-Frontier-Research/1.0 (Academic Research)',
            'Accept': 'application/json'
        })

        if email:
            logger.info(f"Using polite pool with email: {email}")

    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict]:
        """Make a rate-limited API request."""
        self._rate_limit()

        url = f"{self.BASE_URL}/{endpoint}"

        if params is None:
            params = {}

        # Add email for polite pool
        if self.email:
            params['mailto'] = self.email

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenAlex API request failed: {e}")
            return None

    def search_by_title(
        self,
        title: str,
        year: Optional[int] = None,
        min_score: float = 0.7
    ) -> Optional[PaperMatch]:
        """
        Search for a paper by title.

        Parameters
        ----------
        title : str
            Paper title to search for
        year : int, optional
            Publication year to filter results
        min_score : float
            Minimum relevance score (0-1) to accept match

        Returns
        -------
        PaperMatch or None
            Best matching paper with author/affiliation data
        """
        # Clean title for search
        clean_title = title.strip().replace('\n', ' ')

        params = {
            'search': clean_title,
            'select': 'id,title,doi,publication_year,authorships,cited_by_count',
            'per_page': 5
        }

        if year:
            params['filter'] = f'publication_year:{year}'

        data = self._make_request('works', params)

        if not data or 'results' not in data or not data['results']:
            logger.debug(f"No results for title: {title[:50]}...")
            return None

        # Find best match
        best_match = None
        best_score = 0.0

        for result in data['results']:
            result_title = result.get('title') or ''  # Handle None values
            score = self._compute_title_similarity(
                clean_title.lower(),
                result_title.lower()
            )

            if score > best_score and score >= min_score:
                best_score = score
                best_match = result

        if not best_match:
            logger.debug(f"No good match for title: {title[:50]}... (best score: {best_score:.2f})")
            return None

        return self._parse_paper_result(best_match, best_score)

    def _compute_title_similarity(self, title1: str, title2: str) -> float:
        """
        Compute similarity between two titles.

        Uses Jaccard similarity on word tokens for robustness
        to minor differences.
        """
        # Tokenize and normalize
        def tokenize(s):
            # Remove punctuation and split
            import re
            words = re.findall(r'\b\w+\b', s.lower())
            return set(words)

        tokens1 = tokenize(title1)
        tokens2 = tokenize(title2)

        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union if union > 0 else 0.0

    def _parse_paper_result(self, result: Dict, score: float) -> PaperMatch:
        """Parse OpenAlex result into PaperMatch object."""
        authors = []
        authorships = result.get('authorships', [])
        n_authors = len(authorships)

        for idx, authorship in enumerate(authorships):
            author_data = authorship.get('author', {})

            # Parse institutions
            institutions = []
            for inst_data in authorship.get('institutions', []):
                institution = Institution(
                    openalex_id=inst_data.get('id', ''),
                    name=inst_data.get('display_name', ''),
                    country=inst_data.get('country', None),
                    country_code=inst_data.get('country_code', None),
                    type=inst_data.get('type', None),
                    ror=inst_data.get('ror', None)
                )
                institutions.append(institution)

            # Get raw affiliation string if available
            raw_affil = None
            raw_strings = authorship.get('raw_affiliation_strings', [])
            if raw_strings:
                raw_affil = '; '.join(raw_strings)

            author = Author(
                openalex_id=author_data.get('id', ''),
                name=author_data.get('display_name', 'Unknown'),
                position=idx,
                is_first=(idx == 0),
                is_last=(idx == n_authors - 1),
                institutions=institutions,
                raw_affiliation=raw_affil
            )
            authors.append(author)

        return PaperMatch(
            openalex_id=result.get('id', ''),
            title=result.get('title', ''),
            doi=result.get('doi'),
            publication_year=result.get('publication_year'),
            authors=authors,
            match_score=score,
            cited_by_count=result.get('cited_by_count', 0)
        )

    def get_author_details(self, openalex_id: str) -> Optional[Dict]:
        """
        Get detailed information about an author.

        Parameters
        ----------
        openalex_id : str
            OpenAlex author ID (e.g., 'A2164292938')

        Returns
        -------
        dict with author details including works_count, cited_by_count,
        last_known_institutions, etc.
        """
        # Extract ID from URL if full URL provided
        if openalex_id.startswith('https://'):
            openalex_id = openalex_id.split('/')[-1]

        params = {
            'select': 'id,display_name,works_count,cited_by_count,'
                      'last_known_institutions,summary_stats,affiliations'
        }

        data = self._make_request(f'authors/{openalex_id}', params)

        if not data:
            return None

        return {
            'openalex_id': data.get('id'),
            'name': data.get('display_name'),
            'works_count': data.get('works_count', 0),
            'cited_by_count': data.get('cited_by_count', 0),
            'h_index': data.get('summary_stats', {}).get('h_index'),
            'i10_index': data.get('summary_stats', {}).get('i10_index'),
            '2yr_mean_citedness': data.get('summary_stats', {}).get('2yr_mean_citedness'),
            'last_known_institutions': [
                {
                    'name': inst.get('display_name'),
                    'country': inst.get('country'),
                    'country_code': inst.get('country_code'),
                    'type': inst.get('type')
                }
                for inst in data.get('last_known_institutions', [])
            ]
        }

    def get_author_h_index(self, openalex_id: str) -> Optional[int]:
        """
        Get h-index for an author (fast, single-field query).

        Parameters
        ----------
        openalex_id : str
            OpenAlex author ID

        Returns
        -------
        int or None
        """
        if not openalex_id:
            return None

        # Extract ID from URL if full URL provided
        if openalex_id.startswith('https://'):
            openalex_id = openalex_id.split('/')[-1]

        params = {'select': 'summary_stats'}
        data = self._make_request(f'authors/{openalex_id}', params)

        if not data:
            return None

        return data.get('summary_stats', {}).get('h_index')

    def batch_get_author_h_indices(
        self,
        author_ids: List[str],
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Optional[int]]:
        """
        Get h-indices for multiple authors.

        Parameters
        ----------
        author_ids : list of str
            OpenAlex author IDs
        progress_callback : callable, optional
            Function called with (current, total)

        Returns
        -------
        dict mapping author_id to h_index (or None)
        """
        results = {}
        n_authors = len(author_ids)

        for i, author_id in enumerate(author_ids):
            if progress_callback:
                progress_callback(i + 1, n_authors)

            results[author_id] = self.get_author_h_index(author_id)

        return results

    def batch_search_titles(
        self,
        titles: List[str],
        years: Optional[List[int]] = None,
        min_score: float = 0.7,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Optional[PaperMatch]]:
        """
        Search for multiple papers by title.

        Parameters
        ----------
        titles : list of str
            Paper titles to search
        years : list of int, optional
            Corresponding publication years
        min_score : float
            Minimum match score
        progress_callback : callable, optional
            Function called with (current, total) for progress tracking

        Returns
        -------
        dict mapping title to PaperMatch (or None if not found)
        """
        results = {}
        n_titles = len(titles)

        if years is None:
            years = [None] * n_titles

        for i, (title, year) in enumerate(zip(titles, years)):
            if progress_callback:
                progress_callback(i + 1, n_titles)

            results[title] = self.search_by_title(title, year=year, min_score=min_score)

        return results
