"""
Elsevier/Scopus API Client for Author Metrics.
===============================================
Provides h-index, citation counts, and affiliation data from Scopus.

Scopus has excellent coverage for academic papers and reliable author metrics.
Requires an API key from https://dev.elsevier.com/

API Documentation: https://dev.elsevier.com/documentation/
Rate Limits: Varies by subscription (typically 2-10 RPS)
"""

import time
import logging
import os
import requests
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ScopusAuthorMetrics:
    """Author metrics from Scopus."""
    scopus_id: str
    name: str
    h_index: Optional[int] = None
    citation_count: int = 0
    document_count: int = 0
    cited_by_count: int = 0
    affiliation_name: Optional[str] = None
    affiliation_country: Optional[str] = None
    orcid: Optional[str] = None
    subject_areas: List[str] = field(default_factory=list)


@dataclass
class ScopusPaperInfo:
    """Paper information from Scopus."""
    scopus_id: str
    eid: str
    title: str
    doi: Optional[str] = None
    publication_year: Optional[int] = None
    citation_count: int = 0
    authors: List[Dict[str, Any]] = field(default_factory=list)
    source_title: Optional[str] = None  # Journal/conference name


class ElsevierClient:
    """
    Client for the Elsevier/Scopus API.

    Provides author metrics and paper lookup functionality.

    Usage:
        client = ElsevierClient(api_key="your-key")

        # Search for a paper
        paper = client.search_paper("Attention Is All You Need")

        # Get author metrics
        if paper and paper.authors:
            author_id = paper.authors[0].get('authid')
            metrics = client.get_author_metrics(author_id)
            print(f"h-index: {metrics.h_index}")
    """

    SCOPUS_BASE_URL = "https://api.elsevier.com/content"
    SEARCH_BASE_URL = "https://api.elsevier.com/content/search/scopus"
    AUTHOR_BASE_URL = "https://api.elsevier.com/content/author"

    def __init__(
        self,
        api_key: Optional[str] = None,
        inst_token: Optional[str] = None,
        requests_per_second: float = 2.0,
        timeout: int = 30
    ):
        """
        Initialize Elsevier client.

        Parameters
        ----------
        api_key : str, optional
            Elsevier API key (can also be set via ELSEVIER_API_KEY env var)
        inst_token : str, optional
            Institution token for higher rate limits
        requests_per_second : float
            Rate limit (default: 2.0, adjust based on your subscription)
        timeout : int
            Request timeout in seconds
        """
        # Strip whitespace/newlines from API key (common issue with copy-paste)
        raw_key = api_key or os.environ.get('ELSEVIER_API_KEY')
        self.api_key = raw_key.strip() if raw_key else None
        self.inst_token = inst_token
        self.timeout = timeout
        self.min_request_interval = 1.0 / requests_per_second
        self._last_request_time = 0.0

        if not self.api_key:
            logger.warning("No Elsevier API key provided - Scopus features will be disabled")

        self.session = requests.Session()
        headers = {
            'Accept': 'application/json',
            'User-Agent': 'HAI-Frontier-Research/1.0 (Academic Research)'
        }
        if self.api_key:
            headers['X-ELS-APIKey'] = self.api_key
        if self.inst_token:
            headers['X-ELS-Insttoken'] = self.inst_token

        self.session.headers.update(headers)

    @property
    def is_available(self) -> bool:
        """Check if the API is available (has API key)."""
        return self.api_key is not None

    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _make_request(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        retries: int = 3
    ) -> Optional[Dict]:
        """Make a rate-limited API request with retries."""
        if not self.api_key:
            return None

        self._rate_limit()

        for attempt in range(retries):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)

                # Handle rate limiting
                if response.status_code == 429:
                    wait_time = min(2 ** (attempt + 1), 10)
                    logger.warning(f"Elsevier rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                # Handle quota exceeded
                if response.status_code == 403:
                    error_data = response.json() if response.text else {}
                    error_msg = error_data.get('error-response', {}).get('error-message', 'Quota exceeded')
                    logger.warning(f"Elsevier API quota issue: {error_msg}")
                    return None

                response.raise_for_status()
                return response.json()

            except requests.exceptions.Timeout:
                logger.warning(f"Elsevier request timeout (attempt {attempt + 1}/{retries})")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None
            except requests.exceptions.RequestException as e:
                logger.error(f"Elsevier API request failed: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None

        return None

    def search_paper(
        self,
        title: str,
        year: Optional[int] = None,
        min_score: float = 0.7
    ) -> Optional[ScopusPaperInfo]:
        """
        Search for a paper by title.

        Parameters
        ----------
        title : str
            Paper title to search
        year : int, optional
            Publication year to filter
        min_score : float
            Minimum title similarity score

        Returns
        -------
        ScopusPaperInfo or None
        """
        if not self.api_key:
            return None

        # Build query
        clean_title = title.strip().replace('"', '')
        query = f'TITLE("{clean_title}")'
        if year:
            query += f' AND PUBYEAR = {year}'

        params = {
            'query': query,
            'count': 5,
            'field': 'dc:title,dc:identifier,prism:doi,prism:coverDate,'
                     'citedby-count,author,prism:publicationName,eid'
        }

        data = self._make_request(self.SEARCH_BASE_URL, params)

        if not data:
            return None

        results = data.get('search-results', {}).get('entry', [])
        if not results or (len(results) == 1 and results[0].get('error')):
            logger.debug(f"No Scopus results for: {title[:50]}...")
            return None

        # Find best match
        best_match = None
        best_score = 0.0

        for result in results:
            result_title = result.get('dc:title', '')
            score = self._title_similarity(title.lower(), result_title.lower())
            if score > best_score and score >= min_score:
                best_score = score
                best_match = result

        if not best_match:
            return None

        # Parse authors
        authors = []
        author_data = best_match.get('author', [])
        if isinstance(author_data, dict):
            author_data = [author_data]
        for auth in author_data:
            authors.append({
                'authid': auth.get('authid'),
                'authname': auth.get('authname'),
                'afid': auth.get('afid')
            })

        # Parse year from date
        pub_year = None
        cover_date = best_match.get('prism:coverDate', '')
        if cover_date and len(cover_date) >= 4:
            try:
                pub_year = int(cover_date[:4])
            except ValueError:
                pass

        return ScopusPaperInfo(
            scopus_id=best_match.get('dc:identifier', '').replace('SCOPUS_ID:', ''),
            eid=best_match.get('eid', ''),
            title=best_match.get('dc:title', ''),
            doi=best_match.get('prism:doi'),
            publication_year=pub_year,
            citation_count=int(best_match.get('citedby-count', 0)),
            authors=authors,
            source_title=best_match.get('prism:publicationName')
        )

    def _title_similarity(self, title1: str, title2: str) -> float:
        """Compute Jaccard similarity between titles."""
        import re

        def tokenize(s):
            words = re.findall(r'\b\w+\b', s.lower())
            return set(words)

        tokens1 = tokenize(title1)
        tokens2 = tokenize(title2)

        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union if union > 0 else 0.0

    def get_author_metrics(self, author_id: str) -> Optional[ScopusAuthorMetrics]:
        """
        Get author metrics by Scopus author ID.

        Parameters
        ----------
        author_id : str
            Scopus author ID

        Returns
        -------
        ScopusAuthorMetrics or None
        """
        if not self.api_key or not author_id:
            return None

        url = f"{self.AUTHOR_BASE_URL}/author_id/{author_id}"
        params = {
            'view': 'ENHANCED',
            'field': 'h-index,citation-count,document-count,cited-by-count,'
                     'affiliation-current,orcid,subject-area,given-name,surname'
        }

        data = self._make_request(url, params)

        if not data:
            return None

        author_data = data.get('author-retrieval-response', [{}])
        if isinstance(author_data, list):
            author_data = author_data[0] if author_data else {}

        # Extract core info
        core_data = author_data.get('coredata', {})
        h_index_data = author_data.get('h-index')

        # Extract affiliation
        affil_name = None
        affil_country = None
        affil_current = author_data.get('affiliation-current', {})
        if isinstance(affil_current, list):
            affil_current = affil_current[0] if affil_current else {}
        if affil_current:
            affil_name = affil_current.get('affiliation-name')
            affil_country = affil_current.get('affiliation-country')

        # Extract name
        pref_name = author_data.get('author-profile', {}).get('preferred-name', {})
        given_name = pref_name.get('given-name', '')
        surname = pref_name.get('surname', '')
        name = f"{given_name} {surname}".strip() or f"Author {author_id}"

        # Extract subject areas
        subject_areas = []
        areas = author_data.get('subject-areas', {}).get('subject-area', [])
        if isinstance(areas, list):
            subject_areas = [area.get('$', '') for area in areas[:5]]

        return ScopusAuthorMetrics(
            scopus_id=author_id,
            name=name,
            h_index=int(h_index_data) if h_index_data else None,
            citation_count=int(core_data.get('citation-count', 0)),
            document_count=int(core_data.get('document-count', 0)),
            cited_by_count=int(core_data.get('cited-by-count', 0)),
            affiliation_name=affil_name,
            affiliation_country=affil_country,
            orcid=author_data.get('coredata', {}).get('orcid'),
            subject_areas=subject_areas
        )

    def get_author_metrics_from_paper(
        self,
        title: str,
        author_position: str = 'first',
        year: Optional[int] = None
    ) -> Optional[ScopusAuthorMetrics]:
        """
        Get author metrics by first searching for a paper.

        Parameters
        ----------
        title : str
            Paper title
        author_position : str
            'first' or 'last'
        year : int, optional
            Publication year

        Returns
        -------
        ScopusAuthorMetrics or None
        """
        paper = self.search_paper(title, year=year)
        if not paper or not paper.authors:
            return None

        # Get target author
        if author_position == 'last':
            target_author = paper.authors[-1]
        else:
            target_author = paper.authors[0]

        author_id = target_author.get('authid')
        if not author_id:
            return None

        return self.get_author_metrics(author_id)

    def search_author(
        self,
        name: str,
        affiliation: Optional[str] = None
    ) -> Optional[ScopusAuthorMetrics]:
        """
        Search for an author by name.

        Parameters
        ----------
        name : str
            Author name
        affiliation : str, optional
            Affiliation to help disambiguate

        Returns
        -------
        ScopusAuthorMetrics or None (first match)
        """
        if not self.api_key:
            return None

        # Build query
        query = f'AUTHNAME({name})'
        if affiliation:
            query += f' AND AFFIL({affiliation})'

        params = {
            'query': query,
            'count': 5
        }

        url = f"{self.AUTHOR_BASE_URL}"
        data = self._make_request(f"{url}/search/author", params)

        if not data:
            return None

        results = data.get('search-results', {}).get('entry', [])
        if not results:
            return None

        # Get first result's ID and fetch full details
        first_author = results[0]
        author_id = first_author.get('dc:identifier', '').replace('AUTHOR_ID:', '')

        if author_id:
            return self.get_author_metrics(author_id)

        return None
