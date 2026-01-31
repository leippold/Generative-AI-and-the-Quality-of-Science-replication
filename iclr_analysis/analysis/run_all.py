"""Master runner for all analyses."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loading import load_data
from analysis.echo_chamber import run_echo_chamber_analysis
from analysis.within_paper import run_within_paper_analysis
from analysis.effort_proxies import run_effort_proxy_analysis
from analysis.collaboration_hypothesis import run_collaboration_analysis
from analysis.heterogeneity import run_heterogeneity_analysis
from analysis.referee_tests import run_referee_tests

def run_all(submissions_path, reviews_path, output_dir='output'):
    os.makedirs(output_dir, exist_ok=True)
    submissions_df, reviews_df = load_data(submissions_path, reviews_path)
    
    results = {}
    results['echo_chamber'] = run_echo_chamber_analysis(submissions_df, reviews_df, output_dir=output_dir)
    results['within_paper'] = run_within_paper_analysis(submissions_df, reviews_df, output_dir=output_dir)
    results['effort_proxies'] = run_effort_proxy_analysis(submissions_df, reviews_df, output_dir=output_dir)
    results['collaboration'] = run_collaboration_analysis(submissions_df, reviews_df, output_dir=output_dir)
    results['heterogeneity'] = run_heterogeneity_analysis(submissions_df, reviews_df, output_dir=output_dir)

    # Referee-requested statistical tests (Lemma 2 variance compression + Table 2 interaction)
    results['referee_tests'] = run_referee_tests(submissions_df, reviews_df, output_dir=output_dir)

    return results

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        run_all(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else 'output')
