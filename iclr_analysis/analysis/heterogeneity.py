"""
Heterogeneity Analysis: Effects by Paper Quality
=================================================
Tests whether the echo chamber effect differs by paper quality.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional

import sys
sys.path.insert(0, '..')

from src.data_loading import load_data, prepare_echo_chamber_data
from src.stats_utils import ols_with_clustered_se
from src.plotting import setup_style, save_figure
from src.constants import QUALITY_COLORS


def create_quality_terciles(df, rating_col='avg_rating'):
    return pd.qcut(df[rating_col], q=3, labels=['Low', 'Medium', 'High'])


def analyze_heterogeneity_by_quality(reviews_df, submissions_df, verbose=True):
    if verbose:
        print("\n" + "="*70)
        print("HETEROGENEITY ANALYSIS: Echo Chamber by Paper Quality")
        print("="*70)
    
    clean = prepare_echo_chamber_data(reviews_df, submissions_df)
    
    paper_quality = submissions_df[['submission_number', 'avg_rating']].drop_duplicates()
    paper_quality['quality_tercile'] = create_quality_terciles(paper_quality)
    
    clean = clean.merge(paper_quality[['submission_number', 'quality_tercile']], 
                        on='submission_number', how='left')
    clean = clean.dropna(subset=['quality_tercile'])
    
    if verbose:
        print(f"\nSample: {len(clean):,} reviews")
        print(clean['quality_tercile'].value_counts().sort_index())
    
    results = {'by_tercile': {}}
    
    for tercile in ['Low', 'Medium', 'High']:
        subset = clean[clean['quality_tercile'] == tercile]
        if len(subset) < 50:
            continue
        
        mean_table = subset.pivot_table(values='rating', index='paper_type',
                                        columns='reviewer_type', aggfunc='mean')
        try:
            HH = mean_table.loc['Human Paper', 'Human Review']
            HA = mean_table.loc['Human Paper', 'AI Review']
            AH = mean_table.loc['AI Paper', 'Human Review']
            AA = mean_table.loc['AI Paper', 'AI Review']
            interaction = (AA - AH) - (HA - HH)
            
            reg = ols_with_clustered_se(subset, 'rating ~ paper_AI * reviewer_AI', 'submission_number')
            p_int = reg['p_values'].get('paper_AI:reviewer_AI', np.nan)
            
            results['by_tercile'][tercile] = {
                'n': len(subset), 'interaction': interaction, 'p_value': p_int
            }
            
            if verbose:
                print(f"\n{tercile}: interaction={interaction:+.4f}, p={p_int:.4f}")
        except:
            pass
    
    # Three-way test
    clean_binary = clean[clean['quality_tercile'].isin(['Low', 'High'])].copy()
    clean_binary['quality_high'] = (clean_binary['quality_tercile'] == 'High').astype(int)
    
    if len(clean_binary) >= 100:
        try:
            reg_3way = ols_with_clustered_se(clean_binary, 
                'rating ~ paper_AI * reviewer_AI * quality_high', 'submission_number')
            results['three_way'] = {
                'coef': reg_3way['params'].get('paper_AI:reviewer_AI:quality_high', np.nan),
                'p_value': reg_3way['p_values'].get('paper_AI:reviewer_AI:quality_high', np.nan)
            }
        except:
            pass
    
    return results


def run_heterogeneity_analysis(submissions_path_or_df, reviews_path_or_df=None,
                               save_figures=True, output_dir='.'):
    if isinstance(submissions_path_or_df, str):
        submissions_df, reviews_df = load_data(submissions_path_or_df, reviews_path_or_df)
    else:
        submissions_df, reviews_df = submissions_path_or_df, reviews_path_or_df
    
    return analyze_heterogeneity_by_quality(reviews_df, submissions_df)


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        run_heterogeneity_analysis(sys.argv[1], sys.argv[2])
