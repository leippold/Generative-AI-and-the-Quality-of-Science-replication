"""
Country and Institution Utilities.
==================================
Standardization and classification utilities for affiliations.
"""

from typing import Optional, Dict, List, Tuple
import pandas as pd
import numpy as np

# ISO 3166-1 alpha-2 country codes to full names
COUNTRY_CODE_MAP = {
    'US': 'United States',
    'GB': 'United Kingdom',
    'CN': 'China',
    'DE': 'Germany',
    'FR': 'France',
    'JP': 'Japan',
    'CA': 'Canada',
    'AU': 'Australia',
    'KR': 'South Korea',
    'CH': 'Switzerland',
    'NL': 'Netherlands',
    'IL': 'Israel',
    'SG': 'Singapore',
    'HK': 'Hong Kong',
    'IN': 'India',
    'IT': 'Italy',
    'ES': 'Spain',
    'SE': 'Sweden',
    'BE': 'Belgium',
    'AT': 'Austria',
    'DK': 'Denmark',
    'FI': 'Finland',
    'NO': 'Norway',
    'PL': 'Poland',
    'PT': 'Portugal',
    'TW': 'Taiwan',
    'BR': 'Brazil',
    'RU': 'Russia',
    'IE': 'Ireland',
    'NZ': 'New Zealand',
}

# Regions for geographic analysis
REGION_MAP = {
    # North America
    'United States': 'North America',
    'Canada': 'North America',

    # Europe
    'United Kingdom': 'Europe',
    'Germany': 'Europe',
    'France': 'Europe',
    'Switzerland': 'Europe',
    'Netherlands': 'Europe',
    'Italy': 'Europe',
    'Spain': 'Europe',
    'Sweden': 'Europe',
    'Belgium': 'Europe',
    'Austria': 'Europe',
    'Denmark': 'Europe',
    'Finland': 'Europe',
    'Norway': 'Europe',
    'Poland': 'Europe',
    'Portugal': 'Europe',
    'Ireland': 'Europe',
    'Russia': 'Europe',

    # Asia
    'China': 'Asia',
    'Japan': 'Asia',
    'South Korea': 'Asia',
    'Singapore': 'Asia',
    'Hong Kong': 'Asia',
    'India': 'Asia',
    'Taiwan': 'Asia',

    # Middle East
    'Israel': 'Middle East',

    # Oceania
    'Australia': 'Oceania',
    'New Zealand': 'Oceania',

    # South America
    'Brazil': 'South America',
}

# Common abbreviations for matching
UNIVERSITY_ABBREVIATIONS = {
    'MIT': 'Massachusetts Institute of Technology',
    'CMU': 'Carnegie Mellon University',
    'UC Berkeley': 'University of California, Berkeley',
    'UCB': 'University of California, Berkeley',
    'UCLA': 'University of California, Los Angeles',
    'UIUC': 'University of Illinois Urbana-Champaign',
    'UW': 'University of Washington',
    'GT': 'Georgia Institute of Technology',
    'Georgia Tech': 'Georgia Institute of Technology',
    'Caltech': 'California Institute of Technology',
    'USC': 'University of Southern California',
    'NYU': 'New York University',
    'UPenn': 'University of Pennsylvania',
    'NUS': 'National University of Singapore',
    'NTU': 'Nanyang Technological University',
    'ETH': 'ETH Zurich',
    'EPFL': 'EPFL',
    'UofT': 'University of Toronto',
    'ANU': 'Australian National University',
    'THU': 'Tsinghua University',
    'PKU': 'Peking University',
    'SJTU': 'Shanghai Jiao Tong University',
    'SNU': 'Seoul National University',
}

# Top CS/AI universities for reputation classification
TOP_CS_UNIVERSITIES = {
    # United States
    'Massachusetts Institute of Technology',
    'MIT',
    'Stanford University',
    'Carnegie Mellon University',
    'University of California, Berkeley',
    'University of Washington',
    'Cornell University',
    'Georgia Institute of Technology',
    'University of Illinois Urbana-Champaign',
    'University of Michigan',
    'Princeton University',
    'Harvard University',
    'California Institute of Technology',
    'University of California, Los Angeles',
    'University of Texas at Austin',
    'Columbia University',
    'New York University',
    'University of Pennsylvania',
    'University of Southern California',
    'Yale University',
    'University of Maryland',

    # Industry Labs
    'Google',
    'Google DeepMind',
    'DeepMind',
    'OpenAI',
    'Meta',
    'Microsoft',
    'Microsoft Research',
    'Apple',
    'Amazon',
    'NVIDIA',
    'IBM Research',

    # International
    'University of Cambridge',
    'University of Oxford',
    'ETH Zurich',
    'EPFL',
    'University of Toronto',
    'McGill University',
    'University of Montreal',
    'Mila',
    'Max Planck Institute',
    'Tsinghua University',
    'Peking University',
    'Shanghai Jiao Tong University',
    'University of Tokyo',
    'National University of Singapore',
    'Nanyang Technological University',
    'KAIST',
    'Seoul National University',
    'Hebrew University of Jerusalem',
    'Technion',
    'Tel Aviv University',
    'Australian National University',
    'University of Melbourne',
}


def standardize_country(
    country: Optional[str],
    country_code: Optional[str] = None
) -> Optional[str]:
    """
    Standardize country name.

    Parameters
    ----------
    country : str or None
        Country name from API
    country_code : str or None
        ISO country code

    Returns
    -------
    Standardized country name or None
    """
    if country and isinstance(country, str):
        return country

    if country_code and isinstance(country_code, str) and country_code in COUNTRY_CODE_MAP:
        return COUNTRY_CODE_MAP[country_code]

    return None


def get_region(country: Optional[str]) -> Optional[str]:
    """
    Get geographic region for a country.

    Parameters
    ----------
    country : str or None
        Country name

    Returns
    -------
    Region name or 'Other'
    """
    if not country or not isinstance(country, str):
        return None

    return REGION_MAP.get(country, 'Other')


def is_top_university(university: Optional[str]) -> bool:
    """
    Check if university is in the top CS/AI institutions list.

    Parameters
    ----------
    university : str or None
        Institution name

    Returns
    -------
    bool
    """
    if not university or not isinstance(university, str):
        return False

    # Check exact match
    if university in TOP_CS_UNIVERSITIES:
        return True

    # Check partial match (for variations like "MIT" vs "Massachusetts Institute of Technology")
    university_lower = university.lower()
    for top_uni in TOP_CS_UNIVERSITIES:
        if top_uni.lower() in university_lower or university_lower in top_uni.lower():
            return True

    return False


def classify_h_index(h_index: Optional[float]) -> str:
    """
    Classify h-index into reputation categories.

    Categories based on typical CS researcher h-indices:
    - Emerging: < 10 (early career, students, new researchers)
    - Established: 10-30 (mid-career, assistant/associate professors)
    - Senior: 30-60 (full professors, lab leaders)
    - Highly Cited: > 60 (field leaders, famous researchers)

    Parameters
    ----------
    h_index : float or None
        Author's h-index

    Returns
    -------
    str category
    """
    if h_index is None or np.isnan(h_index):
        return 'Unknown'

    if h_index < 10:
        return 'Emerging'
    elif h_index < 30:
        return 'Established'
    elif h_index < 60:
        return 'Senior'
    else:
        return 'Highly Cited'


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived columns to enriched DataFrame.

    Adds:
    - Fills missing country names from country codes (first_author, last_author, primary)
    - first_author_region, last_author_region
    - first_author_top_university, last_author_top_university
    - first_author_reputation, last_author_reputation
    - same_country (first and last from same country)
    - international_collaboration (different countries)

    Parameters
    ----------
    df : DataFrame
        Enriched DataFrame with affiliation columns

    Returns
    -------
    DataFrame with additional columns
    """
    df = df.copy()

    # First: Map country codes to country names where country name is missing
    # This handles the case where OpenAlex returns country_code but not country
    country_prefixes = ['first_author', 'last_author', 'primary']
    for prefix in country_prefixes:
        country_col = f'{prefix}_country'
        code_col = f'{prefix}_country_code'

        if code_col in df.columns:
            # Ensure country column exists
            if country_col not in df.columns:
                df[country_col] = None

            # Fill missing country names from codes
            mask = df[country_col].isna() & df[code_col].notna()
            if mask.any():
                df.loc[mask, country_col] = df.loc[mask, code_col].map(COUNTRY_CODE_MAP)

    # Regions (including primary_country from all-author enrichment)
    for prefix in ['first_author', 'last_author', 'primary']:
        country_col = f'{prefix}_country'
        if country_col in df.columns:
            df[f'{prefix}_region'] = df[country_col].apply(get_region)

    # Top university flags
    for prefix in ['first_author', 'last_author']:
        uni_col = f'{prefix}_university'
        if uni_col in df.columns:
            df[f'{prefix}_top_university'] = df[uni_col].apply(is_top_university)

    # Reputation categories
    for prefix in ['first_author', 'last_author']:
        h_col = f'{prefix}_h_index'
        if h_col in df.columns:
            df[f'{prefix}_reputation'] = df[h_col].apply(classify_h_index)

    # Collaboration patterns
    if 'first_author_country' in df.columns and 'last_author_country' in df.columns:
        df['same_country'] = df['first_author_country'] == df['last_author_country']
        df['international_collaboration'] = (
            df['first_author_country'].notna() &
            df['last_author_country'].notna() &
            (df['first_author_country'] != df['last_author_country'])
        )

    return df


def compute_affiliation_summary(df: pd.DataFrame) -> Dict:
    """
    Compute summary statistics for affiliations.

    Parameters
    ----------
    df : DataFrame
        Enriched DataFrame

    Returns
    -------
    dict with summary statistics
    """
    summary = {}

    # Country distribution
    for prefix in ['first_author', 'last_author']:
        country_col = f'{prefix}_country'
        if country_col in df.columns:
            summary[f'{prefix}_country_counts'] = df[country_col].value_counts().to_dict()

        region_col = f'{prefix}_region'
        if region_col in df.columns:
            summary[f'{prefix}_region_counts'] = df[region_col].value_counts().to_dict()

        # Top university stats
        top_uni_col = f'{prefix}_top_university'
        if top_uni_col in df.columns:
            summary[f'{prefix}_top_university_pct'] = df[top_uni_col].mean()

        # Reputation distribution
        rep_col = f'{prefix}_reputation'
        if rep_col in df.columns:
            summary[f'{prefix}_reputation_counts'] = df[rep_col].value_counts().to_dict()

    # Collaboration patterns
    if 'same_country' in df.columns:
        summary['same_country_pct'] = df['same_country'].mean()

    if 'international_collaboration' in df.columns:
        summary['international_collab_pct'] = df['international_collaboration'].mean()

    return summary
