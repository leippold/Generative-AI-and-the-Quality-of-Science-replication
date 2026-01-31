"""
Semantic Scholar API Client for Author Reputation Metrics.
==========================================================
Provides h-index, citation counts, and other author metrics.

API Documentation: https://api.semanticscholar.org/api-docs/
Rate Limits:
  - With API key: 1-10 RPS depending on key tier
  - Without API key: Shared pool (may be slow/unreliable)

NOTE: For faster h-index lookups, prefer OpenAlex or Scopus.
Semantic Scholar is best used as a fallback or for additional metrics.
"""

import os
import time
import logging
import requests
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AuthorMetrics:
    """Author reputation metrics from Semantic Scholar."""
    semantic_scholar_id: str
    name: str
    h_index: Optional[int] = None
    citation_count: int = 0
    paper_count: int = 0
    highly_influential_citation_count: int = 0
    affiliations: List[str] = field(default_factory=list)
    homepage: Optional[str] = None

    @property
    def reputation_score(self) -> float:
        """
        Compute a simple reputation score.

        Combines h-index (quality/consistency) with citation count (impact).
        Score = h_index + log10(citation_count + 1)
        """
        import math
        h = self.h_index or 0
        return h + math.log10(self.citation_count + 1)


@dataclass
class PaperInfo:
    """Paper information from Semantic Scholar."""
    semantic_scholar_id: str
    title: str
    year: Optional[int]
    citation_count: int
    influential_citation_count: int
    authors: List[Dict[str, Any]]
    venue: Optional[str] = None
    arxiv_id: Optional[str] = None


class SemanticScholarClient:
    """
    Client for the Semantic Scholar Academic Graph API.

    Provides author-level metrics for reputation analysis.

    Usage:
        client = SemanticScholarClient(api_key="your-key")  # Optional

        # Search by author name
        metrics = client.search_author("Yoshua Bengio")
        print(f"h-index: {metrics.h_index}")

        # Search by paper title first, then get author metrics
        paper = client.search_paper("Attention Is All You Need")
        if paper:
            first_author = paper.authors[0]
            metrics = client.get_author_by_id(first_author['authorId'])

    Note: For batch processing, consider using OpenAlex or Scopus
    which have better rate limits. Use Semantic Scholar as a fallback.
    """

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        requests_per_second: float = 1.0,
        timeout: int = 15,
        max_retries: int = 3,
        fail_fast: bool = True
    ):
        """
        Initialize Semantic Scholar client.

        Parameters
        ----------
        api_key : str, optional
            API key (can also be set via SEMANTIC_SCHOLAR env var)
        requests_per_second : float
            Rate limit (default: 1.0 for API key users)
        timeout : int
            Request timeout in seconds (reduced from 30 for faster failures)
        max_retries : int
            Maximum retry attempts
        fail_fast : bool
            If True, return None quickly on errors instead of retrying
        """
        # Strip whitespace/newlines from API key (common issue with copy-paste)
        raw_key = api_key or os.environ.get('SEMANTIC_SCHOLAR')
        self.api_key = raw_key.strip() if raw_key else None
        self.timeout = timeout
        self.max_retries = max_retries
        self.fail_fast = fail_fast
        self.min_request_interval = 1.0 / requests_per_second
        self._last_request_time = 0.0
        self._consecutive_failures = 0
        self._disabled_until = 0.0

        self.session = requests.Session()
        headers = {
            'User-Agent': 'HAI-Frontier-Research/1.0 (Academic Research)',
            'Accept': 'application/json'
        }
        if self.api_key:
            headers['x-api-key'] = self.api_key
            logger.info("Using Semantic Scholar API with authentication")
        else:
            logger.warning("No Semantic Scholar API key - using shared pool (may be slow)")

        self.session.headers.update(headers)

    @property
    def is_available(self) -> bool:
        """Check if API is temporarily disabled due to failures."""
        if self._disabled_until > time.time():
            return False
        return True

    def _disable_temporarily(self, duration: float = 60.0):
        """Temporarily disable API after consecutive failures."""
        self._disabled_until = time.time() + duration
        logger.warning(f"Semantic Scholar disabled for {duration}s due to failures")

    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        retries: Optional[int] = None
    ) -> Optional[Dict]:
        """Make a rate-limited API request with retries and circuit breaker."""
        # Check if API is temporarily disabled
        if not self.is_available:
            logger.debug("Semantic Scholar API temporarily disabled")
            return None

        self._rate_limit()

        url = f"{self.BASE_URL}/{endpoint}"
        retries = retries if retries is not None else self.max_retries

        # Use fewer retries if fail_fast is enabled
        if self.fail_fast:
            retries = min(retries, 2)

        for attempt in range(retries):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)

                # Handle rate limiting
                if response.status_code == 429:
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= 5:
                        self._disable_temporarily(60.0)
                        return None
                    wait_time = min(2 ** attempt, 8)  # Cap at 8 seconds
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                # Handle server errors
                if response.status_code >= 500:
                    self._consecutive_failures += 1
                    if self.fail_fast:
                        return None
                    if attempt < retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return None

                response.raise_for_status()

                # Success - reset failure counter
                self._consecutive_failures = 0
                return response.json()

            except requests.exceptions.Timeout:
                logger.warning(f"Semantic Scholar timeout (attempt {attempt + 1}/{retries})")
                self._consecutive_failures += 1
                if self.fail_fast or attempt >= retries - 1:
                    if self._consecutive_failures >= 3:
                        self._disable_temporarily(30.0)
                    return None
                time.sleep(1)
                continue

            except requests.exceptions.RequestException as e:
                logger.error(f"Semantic Scholar API request failed: {e}")
                self._consecutive_failures += 1
                if self.fail_fast or attempt >= retries - 1:
                    return None
                time.sleep(2 ** attempt)
                continue

        return None

    def search_paper(
        self,
        title: str,
        fields: Optional[List[str]] = None
    ) -> Optional[PaperInfo]:
        """
        Search for a paper by title.

        Parameters
        ----------
        title : str
            Paper title to search
        fields : list, optional
            Fields to retrieve

        Returns
        -------
        PaperInfo or None
        """
        if fields is None:
            fields = [
                'paperId', 'title', 'year', 'citationCount',
                'influentialCitationCount', 'authors', 'venue',
                'externalIds'
            ]

        params = {
            'query': title,
            'fields': ','.join(fields),
            'limit': 5
        }

        data = self._make_request('paper/search', params)

        if not data or 'data' not in data or not data['data']:
            logger.debug(f"No paper found for: {title[:50]}...")
            return None

        # Find best match by title similarity
        best_match = None
        best_score = 0.0

        for paper in data['data']:
            score = self._title_similarity(title, paper.get('title', ''))
            if score > best_score:
                best_score = score
                best_match = paper

        if not best_match or best_score < 0.7:
            return None

        external_ids = best_match.get('externalIds', {}) or {}

        return PaperInfo(
            semantic_scholar_id=best_match.get('paperId', ''),
            title=best_match.get('title', ''),
            year=best_match.get('year'),
            citation_count=best_match.get('citationCount', 0),
            influential_citation_count=best_match.get('influentialCitationCount', 0),
            authors=best_match.get('authors', []),
            venue=best_match.get('venue'),
            arxiv_id=external_ids.get('ArXiv')
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

    def get_author_by_id(
        self,
        author_id: str,
        fields: Optional[List[str]] = None
    ) -> Optional[AuthorMetrics]:
        """
        Get author metrics by Semantic Scholar author ID.

        Parameters
        ----------
        author_id : str
            Semantic Scholar author ID
        fields : list, optional
            Fields to retrieve

        Returns
        -------
        AuthorMetrics or None
        """
        if fields is None:
            fields = [
                'authorId', 'name', 'affiliations', 'homepage',
                'paperCount', 'citationCount', 'hIndex'
            ]

        params = {'fields': ','.join(fields)}
        data = self._make_request(f'author/{author_id}', params)

        if not data:
            return None

        return AuthorMetrics(
            semantic_scholar_id=data.get('authorId', ''),
            name=data.get('name', 'Unknown'),
            h_index=data.get('hIndex'),
            citation_count=data.get('citationCount', 0),
            paper_count=data.get('paperCount', 0),
            affiliations=data.get('affiliations', []) or [],
            homepage=data.get('homepage')
        )

    def search_author(
        self,
        name: str,
        fields: Optional[List[str]] = None
    ) -> Optional[AuthorMetrics]:
        """
        Search for an author by name.

        Note: Name matching can be ambiguous. Prefer using
        get_author_by_id when you have the ID from a paper search.

        Parameters
        ----------
        name : str
            Author name to search
        fields : list, optional
            Fields to retrieve

        Returns
        -------
        AuthorMetrics or None (returns first match)
        """
        if fields is None:
            fields = [
                'authorId', 'name', 'affiliations', 'homepage',
                'paperCount', 'citationCount', 'hIndex'
            ]

        params = {
            'query': name,
            'fields': ','.join(fields),
            'limit': 5
        }

        data = self._make_request('author/search', params)

        if not data or 'data' not in data or not data['data']:
            logger.debug(f"No author found for: {name}")
            return None

        # Return first match (usually most relevant)
        author = data['data'][0]

        return AuthorMetrics(
            semantic_scholar_id=author.get('authorId', ''),
            name=author.get('name', 'Unknown'),
            h_index=author.get('hIndex'),
            citation_count=author.get('citationCount', 0),
            paper_count=author.get('paperCount', 0),
            affiliations=author.get('affiliations', []) or [],
            homepage=author.get('homepage')
        )

    def get_author_metrics_from_paper(
        self,
        title: str,
        author_position: str = 'first'
    ) -> Optional[AuthorMetrics]:
        """
        Get author metrics by first searching for a paper.

        Parameters
        ----------
        title : str
            Paper title
        author_position : str
            'first' for first author, 'last' for last author

        Returns
        -------
        AuthorMetrics or None
        """
        paper = self.search_paper(title)
        if not paper or not paper.authors:
            return None

        # Get the target author
        if author_position == 'last':
            target_author = paper.authors[-1]
        else:
            target_author = paper.authors[0]

        author_id = target_author.get('authorId')
        if not author_id:
            return None

        return self.get_author_by_id(author_id)

    def batch_get_author_metrics(
        self,
        author_ids: List[str],
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Optional[AuthorMetrics]]:
        """
        Get metrics for multiple authors.

        Parameters
        ----------
        author_ids : list of str
            Semantic Scholar author IDs
        progress_callback : callable, optional
            Function called with (current, total)

        Returns
        -------
        dict mapping author_id to AuthorMetrics
        """
        results = {}
        n_authors = len(author_ids)

        for i, author_id in enumerate(author_ids):
            if progress_callback:
                progress_callback(i + 1, n_authors)

            results[author_id] = self.get_author_by_id(author_id)

        return results
