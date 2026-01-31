#!/usr/bin/env python3
"""
Example Usage of the ArXiv Integration Pipeline.
=================================================
Demonstrates how to enrich ICLR data with author affiliations and reputation metrics.

This script shows:
1. Looking up a single paper
2. Batch enrichment of a DataFrame
3. Adding derived columns (regions, reputation categories)
4. Computing summary statistics

Usage:
    # As a script
    python example_usage.py

    # Or import and use in your analysis
    from arxiv_integration import AffiliationEnricher
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from arxiv_integration.src.openalex_api import OpenAlexClient
from arxiv_integration.src.semantic_scholar_api import SemanticScholarClient
from arxiv_integration.src.affiliation_enrichment import AffiliationEnricher, enrich_iclr_data
from arxiv_integration.src.country_utils import add_derived_columns, compute_affiliation_summary


def demo_single_lookup():
    """Demonstrate looking up a single paper."""
    print("=" * 60)
    print("Demo 1: Single Paper Lookup")
    print("=" * 60)

    # Initialize the enricher
    # Provide your email for higher rate limits (optional but recommended)
    enricher = AffiliationEnricher(
        email=None,  # Replace with your email for polite pool
        semantic_scholar_api_key=None  # Optional: for higher rate limits
    )

    # Look up a famous paper
    title = "Attention Is All You Need"
    print(f"\nLooking up: '{title}'")

    result = enricher.lookup_paper(title, include_reputation=True)

    if result.matched:
        print(f"\n  Match score: {result.match_score:.2f}")
        print(f"  OpenAlex ID: {result.openalex_id}")
        print(f"  DOI: {result.doi}")
        print(f"  Citations: {result.cited_by_count:,}")

        if result.first_author:
            fa = result.first_author
            print(f"\n  First Author:")
            print(f"    Name: {fa.name}")
            print(f"    University: {fa.university}")
            print(f"    Country: {fa.country}")
            print(f"    H-index: {fa.h_index}")
            print(f"    Citation count: {fa.citation_count:,}" if fa.citation_count else "")

        if result.last_author and result.last_author != result.first_author:
            la = result.last_author
            print(f"\n  Last Author:")
            print(f"    Name: {la.name}")
            print(f"    University: {la.university}")
            print(f"    Country: {la.country}")
            print(f"    H-index: {la.h_index}")
            print(f"    Citation count: {la.citation_count:,}" if la.citation_count else "")
    else:
        print(f"  No match found: {result.error}")


def demo_openalex_direct():
    """Demonstrate direct OpenAlex API usage."""
    print("\n" + "=" * 60)
    print("Demo 2: Direct OpenAlex API Usage")
    print("=" * 60)

    client = OpenAlexClient()

    # Search for a paper
    title = "BERT: Pre-training of Deep Bidirectional Transformers"
    print(f"\nSearching OpenAlex for: '{title}'")

    match = client.search_by_title(title)

    if match:
        print(f"\n  Found: {match.title}")
        print(f"  Year: {match.publication_year}")
        print(f"  Authors: {len(match.authors)}")

        # Show all authors
        for i, author in enumerate(match.authors[:5]):  # First 5
            print(f"\n  Author {i+1}: {author.name}")
            if author.university:
                print(f"    Institution: {author.university}")
                print(f"    Country: {author.country}")

        if len(match.authors) > 5:
            print(f"\n  ... and {len(match.authors) - 5} more authors")


def demo_batch_enrichment():
    """Demonstrate batch enrichment of a DataFrame."""
    print("\n" + "=" * 60)
    print("Demo 3: Batch DataFrame Enrichment")
    print("=" * 60)

    # Create a sample DataFrame (simulating ICLR submissions)
    sample_data = pd.DataFrame({
        'submission_number': [1, 2, 3, 4, 5],
        'title': [
            'Attention Is All You Need',
            'BERT: Pre-training of Deep Bidirectional Transformers',
            'Language Models are Unsupervised Multitask Learners',
            'Deep Residual Learning for Image Recognition',
            'This Paper Does Not Exist XYZ123'  # Should not match
        ],
        'avg_rating': [8.5, 8.0, 7.5, 9.0, 5.0],
        'ai_percentage': [10, 5, 15, 0, 100]
    })

    print(f"\nSample data:\n{sample_data[['title', 'avg_rating']]}")

    # Initialize enricher
    enricher = AffiliationEnricher()

    # Enrich the data (this will make API calls)
    print("\nEnriching data (this may take a moment)...")
    enriched = enricher.enrich_submissions(
        sample_data,
        title_col='title',
        include_reputation=False,  # Set to True for h-index (slower)
        progress_bar=True
    )

    # Show results
    cols_to_show = [
        'title', 'first_author_name', 'first_author_country',
        'first_author_university', 'openalex_match_score'
    ]
    available_cols = [c for c in cols_to_show if c in enriched.columns]
    print(f"\nEnriched results:")
    print(enriched[available_cols].to_string())

    # Add derived columns
    enriched = add_derived_columns(enriched)

    # Show summary
    stats = enricher.get_summary_stats(enriched)
    print(f"\nSummary statistics:")
    print(f"  Match rate: {stats['match_rate']:.1%}")
    if 'first_author_top_countries' in stats:
        print(f"  Top countries: {list(stats['first_author_top_countries'].keys())[:3]}")

    return enriched


def demo_with_real_iclr_data(submissions_path: str):
    """
    Demonstrate with real ICLR data.

    Parameters
    ----------
    submissions_path : str
        Path to your ICLR submissions CSV/Excel file
    """
    print("\n" + "=" * 60)
    print("Demo 4: Real ICLR Data Enrichment")
    print("=" * 60)

    # Load data
    if submissions_path.endswith('.csv'):
        df = pd.read_csv(submissions_path)
    else:
        df = pd.read_excel(submissions_path)

    print(f"Loaded {len(df)} submissions")

    # Check for title column
    if 'title' not in df.columns:
        print("Error: No 'title' column found. Available columns:")
        print(df.columns.tolist())
        return None

    # Enrich (with reputation metrics)
    enricher = AffiliationEnricher(
        email="your-email@university.edu"  # Replace with your email
    )

    # For large datasets, you might want to process in batches
    # or disable reputation metrics for speed
    enriched = enricher.enrich_submissions(
        df,
        title_col='title',
        include_reputation=True,  # Set to False for faster processing
        progress_bar=True
    )

    # Add derived columns
    enriched = add_derived_columns(enriched)

    # Compute summary
    summary = compute_affiliation_summary(enriched)

    print(f"\n{'=' * 40}")
    print("Summary Statistics")
    print(f"{'=' * 40}")

    print(f"\nMatch rate: {enricher.get_summary_stats(enriched)['match_rate']:.1%}")

    if 'first_author_country_counts' in summary:
        print("\nFirst Author Countries (top 10):")
        for country, count in list(summary['first_author_country_counts'].items())[:10]:
            print(f"  {country}: {count}")

    if 'first_author_region_counts' in summary:
        print("\nFirst Author Regions:")
        for region, count in summary['first_author_region_counts'].items():
            print(f"  {region}: {count}")

    if 'first_author_reputation_counts' in summary:
        print("\nFirst Author Reputation Distribution:")
        for rep, count in summary['first_author_reputation_counts'].items():
            print(f"  {rep}: {count}")

    if 'international_collab_pct' in summary:
        print(f"\nInternational collaboration rate: {summary['international_collab_pct']:.1%}")

    return enriched


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("ArXiv Integration Pipeline - Example Usage")
    print("=" * 60)
    print("\nThis pipeline enriches ICLR submission data with:")
    print("  - Author affiliations (country, university)")
    print("  - Author reputation metrics (h-index, citations)")
    print("  - Derived features (region, top university flag)")

    # Run demos
    demo_single_lookup()
    demo_openalex_direct()

    # Uncomment to run batch demo (makes multiple API calls)
    # demo_batch_enrichment()

    # To use with your actual ICLR data:
    # enriched_df = demo_with_real_iclr_data("path/to/iclr_submissions.csv")

    print("\n" + "=" * 60)
    print("Demos complete!")
    print("=" * 60)
    print("\nTo use with your ICLR data:")
    print("  from arxiv_integration import AffiliationEnricher")
    print("  enricher = AffiliationEnricher(email='your@email.com')")
    print("  enriched_df = enricher.enrich_submissions(submissions_df)")


if __name__ == "__main__":
    main()
