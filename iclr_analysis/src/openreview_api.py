"""
OpenReview API Client for ICLR Acceptance Data.
================================================

Downloads decision and presentation tier data from OpenReview.
Supports ICLR 2024, 2025, and future conferences.

Usage:
    from src.openreview_api import fetch_iclr_decisions, merge_acceptance_data

    # Fetch decisions for ICLR 2025
    decisions_df = fetch_iclr_decisions(year=2025)

    # Merge with existing submissions data
    enriched_df = merge_acceptance_data(submissions_df, decisions_df)
"""

import requests
import pandas as pd
import numpy as np
import time
from typing import Optional, Dict, List, Tuple
from urllib.parse import urlparse, parse_qs
import warnings
import json
import re


# =============================================================================
# OPENREVIEW API CONFIGURATION
# =============================================================================

OPENREVIEW_API_V2 = "https://api2.openreview.net"

# Venue IDs for ICLR conferences
ICLR_VENUE_IDS = {
    2024: "ICLR.cc/2024/Conference",
    2025: "ICLR.cc/2025/Conference",
    2026: "ICLR.cc/2026/Conference",
}

# Decision mappings
DECISION_MAPPING = {
    'Accept (Oral)': {'accepted': True, 'tier': 3, 'tier_name': 'Oral'},
    'Accept (Spotlight)': {'accepted': True, 'tier': 2, 'tier_name': 'Spotlight'},
    'Accept (Poster)': {'accepted': True, 'tier': 1, 'tier_name': 'Poster'},
    'Accept': {'accepted': True, 'tier': 1, 'tier_name': 'Poster'},  # Generic accept
    'Reject': {'accepted': False, 'tier': 0, 'tier_name': 'Rejected'},
    'Withdrawn': {'accepted': False, 'tier': -1, 'tier_name': 'Withdrawn'},
    'Desk Reject': {'accepted': False, 'tier': -2, 'tier_name': 'Desk Rejected'},
}

# Rate limiting
REQUEST_DELAY = 0.5  # seconds between requests
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


# =============================================================================
# API HELPER FUNCTIONS
# =============================================================================

def _make_request(url: str, params: Optional[Dict] = None,
                  headers: Optional[Dict] = None) -> Optional[Dict]:
    """
    Make a rate-limited request to OpenReview API with retry logic.
    """
    if headers is None:
        headers = {
            'Accept': 'application/json',
            'User-Agent': 'HAI-Frontier-Research/1.0'
        }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)

            if response.status_code == 200:
                time.sleep(REQUEST_DELAY)
                return response.json()
            elif response.status_code == 429:
                # Rate limited
                wait_time = RETRY_DELAY * (2 ** attempt)
                print(f"Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
            elif response.status_code == 404:
                return None
            else:
                print(f"Request failed with status {response.status_code}: {response.text[:200]}")
                time.sleep(RETRY_DELAY)

        except requests.exceptions.RequestException as e:
            print(f"Request error (attempt {attempt + 1}): {e}")
            time.sleep(RETRY_DELAY)

    return None


def extract_forum_id(openreview_url: str) -> Optional[str]:
    """
    Extract forum ID from OpenReview URL.

    Examples:
        https://openreview.net/forum?id=1AFenZBIcW -> 1AFenZBIcW
    """
    if pd.isna(openreview_url) or not openreview_url:
        return None

    try:
        parsed = urlparse(openreview_url)
        query_params = parse_qs(parsed.query)

        if 'id' in query_params:
            return query_params['id'][0]

        # Try to extract from path
        match = re.search(r'forum\?id=([a-zA-Z0-9_-]+)', openreview_url)
        if match:
            return match.group(1)

    except Exception:
        pass

    return None


# =============================================================================
# DECISION FETCHING FUNCTIONS
# =============================================================================

def fetch_paper_decision(forum_id: str, venue_id: str) -> Optional[Dict]:
    """
    Fetch decision for a single paper from OpenReview.

    Parameters
    ----------
    forum_id : str
        The OpenReview forum ID
    venue_id : str
        The venue ID (e.g., "ICLR.cc/2025/Conference")

    Returns
    -------
    dict with decision info or None if not found
    """
    # Try to get the note/submission directly
    url = f"{OPENREVIEW_API_V2}/notes"
    params = {
        'id': forum_id,
        'details': 'directReplies'
    }

    result = _make_request(url, params)

    if not result or 'notes' not in result or len(result['notes']) == 0:
        return None

    note = result['notes'][0]

    # Look for decision in direct replies
    decision_info = None

    if 'details' in note and 'directReplies' in note['details']:
        for reply in note['details']['directReplies']:
            # Check if this is a decision note
            invitations = reply.get('invitations', [])
            for inv in invitations:
                if 'Decision' in inv or 'Meta_Review' in inv:
                    content = reply.get('content', {})

                    # Try different field names for decision
                    decision_val = None
                    for field in ['decision', 'recommendation', 'final_decision']:
                        if field in content:
                            val = content[field]
                            decision_val = val.get('value', val) if isinstance(val, dict) else val
                            break

                    if decision_val:
                        decision_info = {
                            'forum_id': forum_id,
                            'decision_raw': decision_val,
                        }

                        # Parse decision
                        parsed = DECISION_MAPPING.get(decision_val, {})
                        decision_info.update(parsed)

                        # Try to get venue (oral/spotlight/poster)
                        if 'venue' in content:
                            venue_val = content['venue']
                            venue_val = venue_val.get('value', venue_val) if isinstance(venue_val, dict) else venue_val
                            decision_info['venue'] = venue_val

                        return decision_info

    return decision_info


def fetch_venue_decisions_bulk(venue_id: str, limit: int = 5000) -> pd.DataFrame:
    """
    Fetch all decisions for a venue in bulk using the notes API.

    This is more efficient than fetching individual papers.

    Parameters
    ----------
    venue_id : str
        The venue ID (e.g., "ICLR.cc/2025/Conference")
    limit : int
        Maximum number of submissions to fetch

    Returns
    -------
    DataFrame with decisions
    """
    print(f"Fetching decisions for {venue_id}...")

    all_decisions = []
    offset = 0
    batch_size = 1000

    while offset < limit:
        # Fetch submissions with decisions
        url = f"{OPENREVIEW_API_V2}/notes"
        params = {
            'invitation': f"{venue_id}/-/Submission",
            'limit': min(batch_size, limit - offset),
            'offset': offset,
            'details': 'directReplies'
        }

        result = _make_request(url, params)

        if not result or 'notes' not in result:
            print(f"No results at offset {offset}")
            break

        notes = result['notes']
        if len(notes) == 0:
            break

        print(f"Processing batch at offset {offset}: {len(notes)} submissions")

        for note in notes:
            forum_id = note.get('id') or note.get('forum')
            title = note.get('content', {}).get('title', {})
            title = title.get('value', title) if isinstance(title, dict) else title

            decision_info = {
                'forum_id': forum_id,
                'title': title,
                'decision_raw': None,
                'accepted': None,
                'tier': None,
                'tier_name': None,
            }

            # Look for decision in replies
            if 'details' in note and 'directReplies' in note['details']:
                for reply in note['details']['directReplies']:
                    invitations = reply.get('invitations', [])
                    content = reply.get('content', {})

                    for inv in invitations:
                        if 'Decision' in inv or 'Meta_Review' in inv or 'decision' in inv.lower():
                            # Try different field names
                            for field in ['decision', 'recommendation', 'final_decision']:
                                if field in content:
                                    val = content[field]
                                    decision_val = val.get('value', val) if isinstance(val, dict) else val

                                    decision_info['decision_raw'] = decision_val

                                    # Parse decision
                                    parsed = DECISION_MAPPING.get(decision_val, {})
                                    if parsed:
                                        decision_info.update(parsed)
                                    else:
                                        # Try to infer
                                        decision_lower = decision_val.lower()
                                        if 'accept' in decision_lower:
                                            decision_info['accepted'] = True
                                            if 'oral' in decision_lower:
                                                decision_info['tier'] = 3
                                                decision_info['tier_name'] = 'Oral'
                                            elif 'spotlight' in decision_lower:
                                                decision_info['tier'] = 2
                                                decision_info['tier_name'] = 'Spotlight'
                                            else:
                                                decision_info['tier'] = 1
                                                decision_info['tier_name'] = 'Poster'
                                        elif 'reject' in decision_lower:
                                            decision_info['accepted'] = False
                                            decision_info['tier'] = 0
                                            decision_info['tier_name'] = 'Rejected'
                                    break
                            break

            all_decisions.append(decision_info)

        offset += len(notes)

        if len(notes) < batch_size:
            break

    df = pd.DataFrame(all_decisions)
    print(f"Fetched {len(df)} total submissions")

    if 'accepted' in df.columns:
        n_accepted = df['accepted'].sum() if df['accepted'].notna().any() else 0
        n_with_decision = df['decision_raw'].notna().sum()
        print(f"Decisions available: {n_with_decision}")
        print(f"Accepted: {n_accepted}")

    return df


def fetch_iclr_decisions(year: int = 2025,
                         use_bulk: bool = True,
                         submissions_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Fetch ICLR decisions for a given year.

    Parameters
    ----------
    year : int
        ICLR year (2024, 2025, 2026)
    use_bulk : bool
        If True, fetch all decisions in bulk (recommended)
        If False, fetch only for papers in submissions_df
    submissions_df : DataFrame, optional
        Existing submissions data (used if use_bulk=False)

    Returns
    -------
    DataFrame with columns: forum_id, decision_raw, accepted, tier, tier_name
    """
    if year not in ICLR_VENUE_IDS:
        raise ValueError(f"Unknown ICLR year: {year}. Known years: {list(ICLR_VENUE_IDS.keys())}")

    venue_id = ICLR_VENUE_IDS[year]

    if use_bulk:
        return fetch_venue_decisions_bulk(venue_id)

    # Fetch for individual papers
    if submissions_df is None:
        raise ValueError("submissions_df required when use_bulk=False")

    print(f"Fetching decisions for {len(submissions_df)} papers...")

    decisions = []
    for i, row in submissions_df.iterrows():
        forum_id = extract_forum_id(row.get('openreview_url', ''))

        if not forum_id:
            decisions.append({
                'forum_id': None,
                'decision_raw': None,
                'accepted': None,
                'tier': None,
                'tier_name': None
            })
            continue

        decision = fetch_paper_decision(forum_id, venue_id)

        if decision:
            decisions.append(decision)
        else:
            decisions.append({
                'forum_id': forum_id,
                'decision_raw': None,
                'accepted': None,
                'tier': None,
                'tier_name': None
            })

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(submissions_df)} papers")

    return pd.DataFrame(decisions)


# =============================================================================
# DATA MERGING FUNCTIONS
# =============================================================================

def merge_acceptance_data(submissions_df: pd.DataFrame,
                          decisions_df: pd.DataFrame,
                          on_column: str = 'forum_id') -> pd.DataFrame:
    """
    Merge acceptance data with submissions DataFrame.

    Parameters
    ----------
    submissions_df : DataFrame
        Original submissions data (must have 'openreview_url' column)
    decisions_df : DataFrame
        Decisions data from fetch_iclr_decisions()
    on_column : str
        Column to merge on ('forum_id' or 'title')

    Returns
    -------
    DataFrame with acceptance columns added
    """
    df = submissions_df.copy()

    # Extract forum IDs if not present
    if 'forum_id' not in df.columns and 'openreview_url' in df.columns:
        df['forum_id'] = df['openreview_url'].apply(extract_forum_id)

    # Merge
    if on_column == 'forum_id':
        merged = df.merge(
            decisions_df[['forum_id', 'decision_raw', 'accepted', 'tier', 'tier_name']],
            on='forum_id',
            how='left'
        )
    elif on_column == 'title':
        # Normalize titles for matching
        df['_title_norm'] = df['title'].str.lower().str.strip()
        decisions_df = decisions_df.copy()
        decisions_df['_title_norm'] = decisions_df['title'].str.lower().str.strip()

        merged = df.merge(
            decisions_df[['_title_norm', 'decision_raw', 'accepted', 'tier', 'tier_name']],
            on='_title_norm',
            how='left'
        )
        merged = merged.drop(columns=['_title_norm'])
    else:
        raise ValueError(f"Unknown merge column: {on_column}")

    # Report merge statistics
    n_matched = merged['accepted'].notna().sum()
    n_accepted = merged['accepted'].sum() if merged['accepted'].notna().any() else 0

    print(f"\nMerge Statistics:")
    print(f"  Total submissions: {len(merged)}")
    print(f"  Matched with decisions: {n_matched} ({100*n_matched/len(merged):.1f}%)")
    print(f"  Accepted: {n_accepted}")
    print(f"  Rejected/Withdrawn: {n_matched - n_accepted}")

    if 'tier_name' in merged.columns:
        tier_counts = merged['tier_name'].value_counts()
        print(f"\nTier Distribution:")
        for tier, count in tier_counts.items():
            print(f"  {tier}: {count}")

    return merged


def load_or_fetch_decisions(submissions_df: pd.DataFrame,
                            cache_path: str = 'data/iclr_decisions.csv',
                            year: int = 2025,
                            force_refresh: bool = False) -> pd.DataFrame:
    """
    Load cached decisions or fetch from OpenReview.

    Parameters
    ----------
    submissions_df : DataFrame
        Submissions data to merge with
    cache_path : str
        Path to cache file
    year : int
        ICLR year
    force_refresh : bool
        If True, fetch fresh data even if cache exists

    Returns
    -------
    DataFrame with acceptance data merged
    """
    import os

    if not force_refresh and os.path.exists(cache_path):
        print(f"Loading cached decisions from {cache_path}")
        decisions_df = pd.read_csv(cache_path)
        return merge_acceptance_data(submissions_df, decisions_df)

    # Fetch fresh data
    print(f"Fetching decisions from OpenReview for ICLR {year}...")
    decisions_df = fetch_iclr_decisions(year=year, use_bulk=True)

    # Cache the results
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    decisions_df.to_csv(cache_path, index=False)
    print(f"Cached decisions to {cache_path}")

    return merge_acceptance_data(submissions_df, decisions_df)


# =============================================================================
# DIAGNOSTIC FUNCTIONS
# =============================================================================

def check_decision_availability(year: int = 2025) -> Dict:
    """
    Check if ICLR decisions are available for a given year.

    Returns
    -------
    dict with availability info
    """
    venue_id = ICLR_VENUE_IDS.get(year)
    if not venue_id:
        return {'available': False, 'reason': f'Unknown year: {year}'}

    # Try to fetch a small sample
    url = f"{OPENREVIEW_API_V2}/notes"
    params = {
        'invitation': f"{venue_id}/-/Submission",
        'limit': 10,
        'details': 'directReplies'
    }

    result = _make_request(url, params)

    if not result:
        return {'available': False, 'reason': 'API request failed'}

    notes = result.get('notes', [])
    if len(notes) == 0:
        return {'available': False, 'reason': 'No submissions found'}

    # Check for decisions - look for multiple possible invitation patterns
    n_with_decision = 0
    sample_decisions = []

    for note in notes:
        if 'details' in note and 'directReplies' in note['details']:
            for reply in note['details']['directReplies']:
                invitations = reply.get('invitations', [])
                content = reply.get('content', {})

                for inv in invitations:
                    # Check multiple possible invitation patterns (same as bulk fetch)
                    if 'Decision' in inv or 'Meta_Review' in inv or 'decision' in inv.lower():
                        # Try to extract decision value
                        for field in ['decision', 'recommendation', 'final_decision']:
                            if field in content:
                                val = content[field]
                                val = val.get('value', val) if isinstance(val, dict) else val
                                if val:
                                    n_with_decision += 1
                                    sample_decisions.append(val)
                                break
                        break

    if n_with_decision > 0:
        return {
            'available': True,
            'n_submissions_sampled': len(notes),
            'n_with_decisions': n_with_decision,
            'sample_decisions': sample_decisions[:5],
            'venue_id': venue_id
        }
    else:
        return {
            'available': False,
            'reason': 'No decision data found in sample',
            'n_submissions_sampled': len(notes),
            'venue_id': venue_id
        }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        year = int(sys.argv[1])
    else:
        year = 2025

    print(f"Checking ICLR {year} decision availability...")
    status = check_decision_availability(year)
    print(json.dumps(status, indent=2))

    if status.get('available'):
        print(f"\nFetching all decisions...")
        decisions_df = fetch_iclr_decisions(year=year)
        print(f"\nSample:")
        print(decisions_df.head(10))
