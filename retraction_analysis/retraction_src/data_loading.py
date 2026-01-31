"""
Data Loading for Retraction Analysis
=====================================
Loads Retraction Watch + Problematic Paper Screener data.
"""

import pandas as pd
import numpy as np
import re


def normalize_title(title):
    """Normalize title for matching across datasets."""
    if not isinstance(title, str):
        return ""
    return re.sub(r'[^a-z0-9]', '', title.lower())


def load_data(retraction_path, problematic_path, start_year=2005):
    """
    Load and clean retraction + problematic paper data.
    
    Parameters
    ----------
    retraction_path : str
        Path to Retraction Watch CSV
    problematic_path : str
        Path to Problematic Paper Screener CSV
    start_year : int
        Filter papers published after this year (default: 2005)
    
    Returns
    -------
    rw_df, prob_df : tuple of DataFrames
    """
    print("Loading data...")
    rw_df = pd.read_csv(retraction_path, encoding='latin-1', low_memory=False)
    prob_df = pd.read_csv(problematic_path, encoding='latin-1', low_memory=False)
    
    print("Cleaning dates...")
    rw_df['RetractionDate'] = pd.to_datetime(rw_df['RetractionDate'], errors='coerce')
    rw_df['OriginalPaperDate'] = pd.to_datetime(rw_df['OriginalPaperDate'], errors='coerce')
    
    # Drop invalid dates
    before = len(rw_df)
    rw_df = rw_df.dropna(subset=['RetractionDate', 'OriginalPaperDate'])
    print(f"  Dropped {before - len(rw_df)} records with missing dates")
    
    # Filter time horizon
    before = len(rw_df)
    rw_df = rw_df[rw_df['OriginalPaperDate'].dt.year >= start_year]
    print(f"  Filtered {before - len(rw_df)} papers before {start_year}")
    
    # Calculate GIGO Window (time to retraction)
    rw_df['GIGO_Years'] = (rw_df['RetractionDate'] - rw_df['OriginalPaperDate']).dt.days / 365.25
    
    # Filter noise (negative time, immediate retractions)
    before = len(rw_df)
    rw_df = rw_df[rw_df['GIGO_Years'] > 0.1]  # > ~36 days
    print(f"  Filtered {before - len(rw_df)} immediate/invalid retractions")
    
    # Add publication year
    rw_df['pub_year'] = rw_df['OriginalPaperDate'].dt.year
    rw_df['retraction_year'] = rw_df['RetractionDate'].dt.year
    
    print(f"\nFinal cohort: {len(rw_df):,} retracted papers")
    print(f"Problematic papers database: {len(prob_df):,} records")
    
    return rw_df, prob_df


def define_ai_cohorts(rw_df, prob_df, 
                      target_detectors=None,
                      ai_keywords=None,
                      merge_citations=True):
    """
    Define AI vs Human cohorts based on detector flags and retraction reasons.
    
    Parameters
    ----------
    rw_df : DataFrame
        Retraction Watch data
    prob_df : DataFrame
        Problematic Paper Screener data
    target_detectors : list, optional
        Detector types to flag as AI (default: tortured, scigen, Seek&Blastn)
    ai_keywords : list, optional
        Keywords in retraction reason to flag as AI
    merge_citations : bool
        Whether to merge citation counts from problematic papers (default: True)
    
    Returns
    -------
    DataFrame with 'is_ai' column (1 = AI, 0 = Human) and optionally 'Citations'
    """
    if target_detectors is None:
        target_detectors = ['tortured', 'scigen', 'Seek&Blastn']
    
    if ai_keywords is None:
        ai_keywords = [
            'generated', 'ChatGPT', 'LLM', 'AI', 'hallucination', 
            'fake', 'paper mill', 'tortured phrases', 'fabricat'
        ]
    
    print("Defining AI cohorts...")
    
    # Normalize titles for matching
    rw_df = rw_df.copy()
    rw_df['title_norm'] = rw_df['Title'].apply(normalize_title)
    prob_df = prob_df.copy()
    prob_df['title_norm'] = prob_df['Title'].apply(normalize_title)
    
    # Columns to merge from problematic papers
    merge_cols = ['title_norm', 'Detectors']
    if merge_citations and 'Citations' in prob_df.columns:
        merge_cols.append('Citations')
        print("  Merging citation counts from problematic papers...")
    
    # Merge detector info (and citations) from problematic papers
    merged = rw_df.merge(
        prob_df[merge_cols].drop_duplicates(subset=['title_norm']),
        on='title_norm',
        how='left'
    )
    
    # Flag 1: Has AI-related detector
    merged['Detectors'] = merged['Detectors'].fillna('')
    has_ai_detector = merged['Detectors'].apply(
        lambda x: any(d.lower() in str(x).lower() for d in target_detectors)
    )
    
    # Flag 2: Has AI-related retraction reason
    merged['Reason'] = merged['Reason'].fillna('')
    has_ai_reason = merged['Reason'].str.contains(
        '|'.join(ai_keywords), case=False, na=False
    )
    
    # Combined flag
    merged['is_ai'] = (has_ai_detector | has_ai_reason).astype(int)
    
    # Clean up
    merged = merged.drop(columns=['title_norm', 'Detectors'], errors='ignore')
    
    # Convert citations to numeric
    if 'Citations' in merged.columns:
        merged['Citations'] = pd.to_numeric(merged['Citations'], errors='coerce')
        n_with_cites = merged['Citations'].notna().sum()
        print(f"  Papers with citation data: {n_with_cites:,}")
    
    # Summary
    n_ai = merged['is_ai'].sum()
    n_human = len(merged) - n_ai
    print(f"\nCohorts defined:")
    print(f"  AI (Treatment):    {n_ai:,} ({100*n_ai/len(merged):.1f}%)")
    print(f"  Human (Control):   {n_human:,} ({100*n_human/len(merged):.1f}%)")
    
    return merged


def get_cohort_summary(df):
    """
    Summary statistics for each cohort.
    """
    summary = df.groupby('is_ai').agg({
        'GIGO_Years': ['count', 'mean', 'median', 'std'],
        'pub_year': ['min', 'max', 'mean']
    }).round(2)
    
    summary.index = ['Human', 'AI']
    return summary
