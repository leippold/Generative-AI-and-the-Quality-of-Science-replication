"""
Data Loading Utilities for ICLR Analysis.
==========================================
Handles loading, cleaning, merging, and categorizing data.
"""

import pandas as pd
import numpy as np
from .constants import (
    AI_CONTENT_BINS, AI_CONTENT_LABELS,
    DEFAULT_AI_PAPER_THRESHOLD, DEFAULT_HUMAN_PAPER_THRESHOLD,
    REVIEW_BINARY
)


def clean_ai_percentage(df, col='ai_percentage'):
    """
    Convert ai_percentage column to numeric, handling string formats like '100%'.

    Parameters
    ----------
    df : DataFrame
        DataFrame with ai_percentage column
    col : str
        Name of the column to clean

    Returns
    -------
    DataFrame with numeric ai_percentage
    """
    df = df.copy()
    if col in df.columns:
        # Handle string format like "100%", "50%"
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace('%', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def load_data(submissions_path, reviews_path):
    """
    Load and clean submissions and reviews data.
    
    Parameters
    ----------
    submissions_path : str
        Path to submissions CSV/Excel file
    reviews_path : str
        Path to reviews CSV/Excel file
        
    Returns
    -------
    submissions_df, reviews_df : tuple of DataFrames
    """
    # Load files
    if submissions_path.endswith('.csv'):
        submissions_df = pd.read_csv(submissions_path)
    else:
        submissions_df = pd.read_excel(submissions_path)
    
    if reviews_path.endswith('.csv'):
        reviews_df = pd.read_csv(reviews_path)
    else:
        reviews_df = pd.read_excel(reviews_path)
    
    # Clean ai_percentage in both dataframes
    for df in [submissions_df, reviews_df]:
        if 'ai_percentage' in df.columns:
            if df['ai_percentage'].dtype == 'object':
                df['ai_percentage'] = (
                    df['ai_percentage'].astype(str)
                    .str.replace('%', '', regex=False)
                    .str.strip()
                    .replace('', np.nan)
                    .astype(float)
                )
            df['ai_percentage'] = pd.to_numeric(df['ai_percentage'], errors='coerce')
    
    # Ensure numeric columns in submissions
    numeric_cols_subs = ['avg_rating', 'submission_number']
    for col in numeric_cols_subs:
        if col in submissions_df.columns:
            submissions_df[col] = pd.to_numeric(submissions_df[col], errors='coerce')
    
    # Ensure numeric columns in reviews
    numeric_cols_revs = ['soundness', 'presentation', 'contribution', 
                         'rating', 'confidence', 'submission_number']
    for col in numeric_cols_revs:
        if col in reviews_df.columns:
            reviews_df[col] = pd.to_numeric(reviews_df[col], errors='coerce')
    
    print(f"Loaded {len(submissions_df):,} submissions, {len(reviews_df):,} reviews")
    
    return submissions_df, reviews_df


def create_ai_categories(df, col='ai_percentage', bins=None, labels=None):
    """
    Create AI content categories from percentage column.
    
    Parameters
    ----------
    df : DataFrame
    col : str
        Column name with AI percentage
    bins : list, optional
        Bin edges (default: AI_CONTENT_BINS)
    labels : list, optional
        Bin labels (default: AI_CONTENT_LABELS)
        
    Returns
    -------
    Series with categorical AI levels
    """
    if bins is None:
        bins = AI_CONTENT_BINS
    if labels is None:
        labels = AI_CONTENT_LABELS
    
    return pd.cut(df[col], bins=bins, labels=labels)


def merge_paper_info(reviews_df, submissions_df,
                     paper_cols=['submission_number', 'ai_percentage', 'avg_rating']):
    """
    Merge paper-level information into reviews dataframe.

    Parameters
    ----------
    reviews_df : DataFrame
    submissions_df : DataFrame
    paper_cols : list
        Columns to merge from submissions

    Returns
    -------
    DataFrame with merged data
    """
    # Clean ai_percentage in submissions if needed
    submissions_clean = clean_ai_percentage(submissions_df.copy())

    available_cols = [c for c in paper_cols if c in submissions_clean.columns]

    merged = reviews_df.merge(
        submissions_clean[available_cols],
        on='submission_number',
        how='left',
        suffixes=('_review', '_paper')
    )

    # Also clean ai_percentage in the merged result
    merged = clean_ai_percentage(merged)

    return merged


def classify_papers(df, ai_col='ai_percentage',
                   ai_threshold=DEFAULT_AI_PAPER_THRESHOLD,
                   human_threshold=DEFAULT_HUMAN_PAPER_THRESHOLD):
    """
    Classify papers as Human Paper, AI Paper, or Mixed.

    Parameters
    ----------
    df : DataFrame
    ai_col : str
        Column with AI percentage
    ai_threshold : float
        Threshold for AI Paper classification (default: 75)
    human_threshold : float
        Threshold for Human Paper classification (default: 25)

    Returns
    -------
    Series with paper type classification
    """
    # Clean ai_percentage column if needed (handles string formats like "100%")
    if ai_col in df.columns and df[ai_col].dtype == 'object':
        df = clean_ai_percentage(df, ai_col)

    return np.where(
        df[ai_col] >= ai_threshold, 'AI Paper',
        np.where(df[ai_col] <= human_threshold, 'Human Paper', 'Mixed')
    )


def classify_reviewers(df, classification_col='ai_classification'):
    """
    Classify reviewers as Human Review or AI Review.
    Only includes Fully human-written and Fully AI-generated.
    
    Parameters
    ----------
    df : DataFrame
    classification_col : str
        Column with AI classification
        
    Returns
    -------
    Series with reviewer type (NaN for edited categories)
    """
    return df[classification_col].map({
        'Fully human-written': 'Human Review',
        'Fully AI-generated': 'AI Review'
    })


def prepare_echo_chamber_data(reviews_df, submissions_df,
                              ai_paper_threshold=DEFAULT_AI_PAPER_THRESHOLD,
                              human_paper_threshold=DEFAULT_HUMAN_PAPER_THRESHOLD):
    """
    Prepare clean 2×2 data for echo chamber analysis.
    
    Returns only clear cases: (Human Paper OR AI Paper) × (Human Review OR AI Review)
    
    Parameters
    ----------
    reviews_df : DataFrame
    submissions_df : DataFrame
    ai_paper_threshold : float
    human_paper_threshold : float
    
    Returns
    -------
    DataFrame with paper_type, reviewer_type, and binary indicators
    """
    # Merge paper info
    merged = merge_paper_info(reviews_df, submissions_df)
    
    # Classify
    merged['paper_type'] = classify_papers(
        merged, 
        ai_threshold=ai_paper_threshold,
        human_threshold=human_paper_threshold
    )
    merged['reviewer_type'] = classify_reviewers(merged)
    
    # Filter to clear 2×2 cases
    clean = merged[
        (merged['paper_type'].isin(['AI Paper', 'Human Paper'])) &
        (merged['reviewer_type'].isin(['AI Review', 'Human Review'])) &
        (merged['rating'].notna())
    ].copy()
    
    # Binary indicators for regression
    clean['paper_AI'] = (clean['paper_type'] == 'AI Paper').astype(int)
    clean['reviewer_AI'] = (clean['reviewer_type'] == 'AI Review').astype(int)
    
    # Interaction term
    clean['paper_AI_x_reviewer_AI'] = clean['paper_AI'] * clean['reviewer_AI']
    
    # Match type label
    clean['match_type'] = clean.apply(
        lambda row: f"{row['paper_type']} + {row['reviewer_type']}", axis=1
    )
    
    return clean


def get_cell_data(df, paper_type, reviewer_type, outcome='rating'):
    """
    Extract data for a specific cell of the 2×2 design.
    
    Parameters
    ----------
    df : DataFrame (from prepare_echo_chamber_data)
    paper_type : str ('Human Paper' or 'AI Paper')
    reviewer_type : str ('Human Review' or 'AI Review')
    outcome : str
        Column to extract
        
    Returns
    -------
    numpy array of values
    """
    mask = (df['paper_type'] == paper_type) & (df['reviewer_type'] == reviewer_type)
    return df.loc[mask, outcome].dropna().values


def compute_sample_summary(df):
    """
    Compute summary statistics for a dataset.
    
    Parameters
    ----------
    df : DataFrame
        
    Returns
    -------
    dict with summary statistics
    """
    summary = {
        'n_submissions': df['submission_number'].nunique() if 'submission_number' in df.columns else None,
        'n_reviews': len(df),
    }
    
    if 'ai_percentage' in df.columns:
        ai_pct = df['ai_percentage'].dropna()
        summary['ai_pct_mean'] = ai_pct.mean()
        summary['ai_pct_median'] = ai_pct.median()
        summary['ai_pct_std'] = ai_pct.std()
        summary['n_zero_ai'] = (df['ai_percentage'] == 0).sum()
    
    if 'rating' in df.columns:
        summary['rating_mean'] = df['rating'].mean()
        summary['rating_std'] = df['rating'].std()
    
    return summary
