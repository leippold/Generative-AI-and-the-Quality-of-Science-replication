"""
Professional Visualizations for Author Enrichment Analysis.
============================================================
Publication-ready figures for geographic and demographic analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List, Tuple
import warnings

# Try to import plotly for interactive maps
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    warnings.warn("Plotly not installed - map visualizations will be disabled")

# Check for kaleido (needed for plotly static image export)
HAS_KALEIDO = False
if HAS_PLOTLY:
    try:
        import kaleido
        HAS_KALEIDO = True
    except ImportError:
        pass


def _safe_write_image(fig, path: str, **kwargs) -> bool:
    """
    Safely write plotly figure to image with multiple fallback methods.

    Handles the common Colab kaleido issues by trying:
    1. kaleido (preferred)
    2. orca (legacy)
    3. Skip with warning (fallback)

    Returns True if successful, False otherwise.
    """
    if not HAS_PLOTLY:
        return False

    # Method 1: Try kaleido (default)
    if HAS_KALEIDO:
        try:
            fig.write_image(path, **kwargs)
            return True
        except Exception as e:
            warnings.warn(f"Kaleido export failed: {e}. Trying fallback methods...")

    # Method 2: Try with explicit engine specification
    try:
        fig.write_image(path, engine="kaleido", **kwargs)
        return True
    except Exception:
        pass

    # Method 3: Try orca (legacy engine)
    try:
        fig.write_image(path, engine="orca", **kwargs)
        return True
    except Exception:
        pass

    # Method 4: For PNG, try converting from SVG via matplotlib
    if path.endswith('.png'):
        try:
            import io
            from PIL import Image
            svg_bytes = fig.to_image(format="svg")
            # Use cairosvg if available
            try:
                import cairosvg
                png_bytes = cairosvg.svg2png(bytestring=svg_bytes)
                with open(path, 'wb') as f:
                    f.write(png_bytes)
                return True
            except ImportError:
                pass
        except Exception:
            pass

    # All methods failed - provide helpful message
    html_path = path.rsplit('.', 1)[0] + '.html'
    warnings.warn(
        f"⚠️ Could not export static image to {path}.\n"
        f"   This is a known issue with kaleido in Colab/cloud environments.\n"
        f"   Workarounds:\n"
        f"   1. Use the interactive HTML version: {html_path}\n"
        f"   2. Install kaleido properly: !pip install -U kaleido\n"
        f"   3. Restart runtime after installing kaleido\n"
        f"   4. Use fig.show() and screenshot manually"
    )
    return False


def _create_matplotlib_choropleth(
    country_stats: pd.DataFrame,
    value_col: str,
    title: str,
    output_path: str,
    cmap: str = 'viridis'
) -> Optional[str]:
    """
    Create a matplotlib-based choropleth map as fallback when plotly export fails.

    Uses geopandas if available, otherwise creates a bar chart representation.

    Parameters
    ----------
    country_stats : DataFrame
        Country statistics with 'country' column and value column
    value_col : str
        Column name for the values to plot
    title : str
        Figure title
    output_path : str
        Path to save the figure
    cmap : str
        Colormap name

    Returns
    -------
    str or None : Path to saved figure, or None if failed
    """
    try:
        # Try geopandas for actual map
        try:
            import geopandas as gpd
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

            # Merge with our data
            world = world.merge(
                country_stats[['country', value_col]],
                left_on='name',
                right_on='country',
                how='left'
            )

            fig, ax = plt.subplots(1, 1, figsize=(15, 8))
            world.plot(
                column=value_col,
                ax=ax,
                legend=True,
                cmap=cmap,
                missing_kwds={'color': 'lightgray'},
                legend_kwds={'label': value_col.replace('_', ' ').title()}
            )
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            return output_path

        except (ImportError, Exception):
            pass

        # Fallback: Create a horizontal bar chart as "map" representation
        fig, ax = plt.subplots(figsize=(12, 8))

        # Sort by value and take top 20
        top_countries = country_stats.nlargest(20, value_col)

        colors = plt.cm.get_cmap(cmap)(np.linspace(0.2, 0.8, len(top_countries)))

        bars = ax.barh(
            range(len(top_countries)),
            top_countries[value_col].values,
            color=colors,
            edgecolor='black',
            linewidth=0.5
        )

        ax.set_yticks(range(len(top_countries)))
        ax.set_yticklabels(top_countries['country'].values)
        ax.set_xlabel(value_col.replace('_', ' ').title())
        ax.set_title(f'{title}\n(Top 20 Countries)', fontsize=14, fontweight='bold')
        ax.invert_yaxis()

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_countries[value_col].values)):
            if value_col == 'n_papers':
                label = f'{int(val):,}'
            else:
                label = f'{val:.1f}'
            ax.text(val + max(top_countries[value_col]) * 0.02, i, label,
                   va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"ℹ️ Created matplotlib fallback for {output_path}")
        return output_path

    except Exception as e:
        warnings.warn(f"Could not create matplotlib fallback: {e}")
        return None


# Publication-quality style settings
STYLE_CONFIG = {
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
}


def set_publication_style():
    """Set matplotlib style for publication-quality figures."""
    plt.rcParams.update(STYLE_CONFIG)
    sns.set_palette('colorblind')


def create_country_analysis(
    df: pd.DataFrame,
    output_dir: str = '.',
    min_papers: int = 10,
    top_n: int = 15,
    save_formats: List[str] = ['png', 'pdf']
) -> Dict[str, any]:
    """
    Create comprehensive country analysis with professional visualizations.

    Parameters
    ----------
    df : DataFrame
        Enriched submissions data
    output_dir : str
        Directory to save figures
    min_papers : int
        Minimum papers to include country in analysis
    top_n : int
        Number of top countries to show in bar charts
    save_formats : list
        Output formats for figures

    Returns
    -------
    dict with statistics and figure paths
    """
    set_publication_style()

    results = {}

    # Compute country statistics
    country_stats = df.groupby('first_author_country').agg({
        'first_author_h_index': ['mean', 'median', 'std', 'count'],
        'title': 'count'
    }).reset_index()
    country_stats.columns = ['country', 'h_index_mean', 'h_index_median',
                              'h_index_std', 'h_index_count', 'n_papers']

    # Filter by minimum papers
    country_stats = country_stats[country_stats['n_papers'] >= min_papers].copy()
    country_stats['h_index_sum'] = country_stats['h_index_mean'] * country_stats['h_index_count']

    results['country_stats'] = country_stats

    # === Figure 1: Side-by-side Bar Charts ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Paper Volume
    top_volume = country_stats.nlargest(top_n, 'n_papers')
    colors_volume = sns.color_palette('Blues_r', n_colors=top_n)

    bars1 = axes[0].barh(
        range(len(top_volume)),
        top_volume['n_papers'].values,
        color=colors_volume,
        edgecolor='darkblue',
        linewidth=0.5
    )
    axes[0].set_yticks(range(len(top_volume)))
    axes[0].set_yticklabels(top_volume['country'].values)
    axes[0].set_xlabel('Number of Papers')
    axes[0].set_title('A. Research Volume by Country', fontweight='bold', loc='left')
    axes[0].invert_yaxis()

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, top_volume['n_papers'].values)):
        axes[0].text(val + max(top_volume['n_papers']) * 0.02, i, f'{int(val):,}',
                     va='center', fontsize=9)

    # Panel B: Mean H-index (Quality)
    top_quality = country_stats.nlargest(top_n, 'h_index_mean')
    colors_quality = sns.color_palette('Oranges_r', n_colors=top_n)

    bars2 = axes[1].barh(
        range(len(top_quality)),
        top_quality['h_index_mean'].values,
        color=colors_quality,
        edgecolor='darkorange',
        linewidth=0.5
    )
    axes[1].set_yticks(range(len(top_quality)))
    axes[1].set_yticklabels(top_quality['country'].values)
    axes[1].set_xlabel('Mean Author H-index')
    axes[1].set_title('B. Author Quality by Country', fontweight='bold', loc='left')
    axes[1].invert_yaxis()

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, top_quality['h_index_mean'].values)):
        axes[1].text(val + max(top_quality['h_index_mean']) * 0.02, i, f'{val:.1f}',
                     va='center', fontsize=9)

    plt.tight_layout()

    # Save
    for fmt in save_formats:
        path = f'{output_dir}/country_analysis_bars.{fmt}'
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    results['bar_chart_path'] = f'{output_dir}/country_analysis_bars.png'
    plt.close()

    # === Figure 2: World Map (if plotly available) ===
    if HAS_PLOTLY:
        # Create choropleth map
        fig_map = px.choropleth(
            country_stats,
            locations='country',
            locationmode='country names',
            color='h_index_mean',
            hover_name='country',
            hover_data={
                'country': False,
                'n_papers': ':,',
                'h_index_mean': ':.1f',
                'h_index_median': ':.0f'
            },
            color_continuous_scale='Viridis',
            labels={
                'h_index_mean': 'Mean H-index',
                'n_papers': 'Papers',
                'h_index_median': 'Median H-index'
            },
            title='Average Author H-index by Country'
        )

        # Professional map styling - focus on relevant regions
        fig_map.update_geos(
            showcoastlines=True,
            coastlinecolor='gray',
            showland=True,
            landcolor='lightgray',
            showocean=True,
            oceancolor='aliceblue',
            showlakes=False,
            showframe=False,
            projection_type='natural earth',
            lataxis_range=[-55, 75],  # Exclude Antarctica
            lonaxis_range=[-130, 160],
            bgcolor='white'
        )

        fig_map.update_layout(
            title={
                'text': 'Average Author H-index by Country',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            geo=dict(
                showframe=False,
                showcoastlines=True,
            ),
            margin=dict(l=0, r=0, t=50, b=0),
            coloraxis_colorbar=dict(
                title='Mean H-index',
                tickformat='.0f'
            )
        )

        # Save as HTML (interactive) and PNG
        html_path = f'{output_dir}/country_map_quality.html'
        png_path = f'{output_dir}/country_map_quality.png'
        fig_map.write_html(html_path)

        png_success = _safe_write_image(fig_map, png_path, scale=2)
        if png_success:
            results['map_path'] = png_path
        else:
            # Create matplotlib fallback for PNG
            fallback_path = _create_matplotlib_choropleth(
                country_stats, 'h_index_mean',
                'Average Author H-index by Country',
                f'{output_dir}/country_map_quality.png',
                cmap='viridis'
            )
            if fallback_path:
                results['map_path'] = fallback_path
            else:
                results['map_path'] = html_path  # Fall back to HTML
        results['map_html_path'] = html_path

        # === Figure 3: Volume Map ===
        fig_vol = px.choropleth(
            country_stats,
            locations='country',
            locationmode='country names',
            color='n_papers',
            hover_name='country',
            hover_data={
                'country': False,
                'n_papers': ':,',
                'h_index_mean': ':.1f'
            },
            color_continuous_scale='Blues',
            labels={
                'n_papers': 'Papers',
                'h_index_mean': 'Mean H-index'
            },
            title='Research Volume by Country'
        )

        fig_vol.update_geos(
            showcoastlines=True,
            coastlinecolor='gray',
            showland=True,
            landcolor='lightgray',
            showocean=True,
            oceancolor='aliceblue',
            showlakes=False,
            showframe=False,
            projection_type='natural earth',
            lataxis_range=[-55, 75],
            lonaxis_range=[-130, 160],
            bgcolor='white'
        )

        fig_vol.update_layout(
            title={
                'text': 'Research Volume by Country (Number of Papers)',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            margin=dict(l=0, r=0, t=50, b=0),
            coloraxis_colorbar=dict(
                title='Papers',
                tickformat=','
            )
        )

        html_path_vol = f'{output_dir}/country_map_volume.html'
        png_path_vol = f'{output_dir}/country_map_volume.png'
        fig_vol.write_html(html_path_vol)

        png_success_vol = _safe_write_image(fig_vol, png_path_vol, scale=2)
        if png_success_vol:
            results['volume_map_path'] = png_path_vol
        else:
            # Create matplotlib fallback for PNG
            fallback_path_vol = _create_matplotlib_choropleth(
                country_stats, 'n_papers',
                'Research Volume by Country',
                f'{output_dir}/country_map_volume.png',
                cmap='Blues'
            )
            if fallback_path_vol:
                results['volume_map_path'] = fallback_path_vol
            else:
                results['volume_map_path'] = html_path_vol  # Fall back to HTML
        results['volume_map_html_path'] = html_path_vol

    # === Figure 4: Combined Summary Figure ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel A: Volume vs Quality scatter
    ax = axes[0, 0]
    scatter = ax.scatter(
        country_stats['n_papers'],
        country_stats['h_index_mean'],
        s=country_stats['h_index_count'] / 2,
        alpha=0.6,
        c=country_stats['h_index_mean'],
        cmap='viridis',
        edgecolors='black',
        linewidth=0.5
    )

    # Label top countries
    for _, row in country_stats.nlargest(8, 'n_papers').iterrows():
        ax.annotate(
            row['country'],
            (row['n_papers'], row['h_index_mean']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9
        )

    ax.set_xlabel('Number of Papers')
    ax.set_ylabel('Mean Author H-index')
    ax.set_title('A. Volume vs Quality by Country', fontweight='bold', loc='left')
    ax.set_xscale('log')

    # Panel B: H-index distribution by top countries
    ax = axes[0, 1]
    top_countries = country_stats.nlargest(8, 'n_papers')['country'].tolist()
    plot_data = df[df['first_author_country'].isin(top_countries)].copy()

    order = plot_data.groupby('first_author_country')['first_author_h_index'].median().sort_values(ascending=False).index
    sns.boxplot(
        data=plot_data,
        y='first_author_country',
        x='first_author_h_index',
        order=order,
        ax=ax,
        palette='viridis'
    )
    ax.set_xlabel('Author H-index')
    ax.set_ylabel('')
    ax.set_title('B. H-index Distribution (Top 8 Countries)', fontweight='bold', loc='left')

    # Panel C: Regional breakdown
    ax = axes[1, 0]
    if 'first_author_region' in df.columns:
        region_stats = df.groupby('first_author_region').agg({
            'first_author_h_index': 'mean',
            'title': 'count'
        }).reset_index()
        region_stats.columns = ['region', 'h_index_mean', 'n_papers']
        region_stats = region_stats.sort_values('n_papers', ascending=True)

        colors = plt.cm.Set2(np.linspace(0, 1, len(region_stats)))
        bars = ax.barh(region_stats['region'], region_stats['n_papers'], color=colors)
        ax.set_xlabel('Number of Papers')
        ax.set_title('C. Papers by Region', fontweight='bold', loc='left')

        # Add h-index annotation
        for i, (bar, h_idx) in enumerate(zip(bars, region_stats['h_index_mean'])):
            ax.text(bar.get_width() + max(region_stats['n_papers']) * 0.02,
                   bar.get_y() + bar.get_height()/2,
                   f'h={h_idx:.1f}', va='center', fontsize=9, color='gray')

    # Panel D: Quality ranking changes
    ax = axes[1, 1]
    # Compare volume rank vs quality rank
    country_stats['volume_rank'] = country_stats['n_papers'].rank(ascending=False)
    country_stats['quality_rank'] = country_stats['h_index_mean'].rank(ascending=False)
    country_stats['rank_diff'] = country_stats['volume_rank'] - country_stats['quality_rank']

    top_by_volume = country_stats.nlargest(12, 'n_papers').copy()
    x = range(len(top_by_volume))

    ax.bar(x, top_by_volume['volume_rank'], width=0.35, label='Volume Rank', color='steelblue', alpha=0.7)
    ax.bar([i + 0.35 for i in x], top_by_volume['quality_rank'], width=0.35, label='Quality Rank', color='coral', alpha=0.7)

    ax.set_xticks([i + 0.175 for i in x])
    ax.set_xticklabels(top_by_volume['country'], rotation=45, ha='right')
    ax.set_ylabel('Rank (lower is better)')
    ax.set_title('D. Volume vs Quality Ranking', fontweight='bold', loc='left')
    ax.legend()
    ax.invert_yaxis()

    plt.tight_layout()

    for fmt in save_formats:
        plt.savefig(f'{output_dir}/country_analysis_summary.{fmt}', dpi=300, bbox_inches='tight', facecolor='white')
    results['summary_path'] = f'{output_dir}/country_analysis_summary.png'
    plt.close()

    return results


def generate_statistical_summary(
    df: pd.DataFrame,
    output_dir: str = '.'
) -> str:
    """
    Generate a statistical summary with key findings.

    Returns markdown-formatted text.
    """
    # Basic stats
    total = len(df)
    with_country = df['first_author_country'].notna().sum()
    with_h_index = df['first_author_h_index'].notna().sum()

    # Country stats
    country_counts = df['first_author_country'].value_counts()
    top_countries = country_counts.head(5)

    # Quality stats
    country_quality = df.groupby('first_author_country')['first_author_h_index'].agg(['mean', 'median', 'count'])
    country_quality = country_quality[country_quality['count'] >= 10].sort_values('mean', ascending=False)

    # Volume vs quality comparison
    top_by_volume = country_counts.head(10).index.tolist()
    top_by_quality = country_quality.head(10).index.tolist()

    summary = f"""
# Geographic Analysis of ICLR Submissions

## Data Coverage
- **Total papers**: {total:,}
- **With country data**: {with_country:,} ({100*with_country/total:.1f}%)
- **With h-index data**: {with_h_index:,} ({100*with_h_index/total:.1f}%)

## Key Findings

### 1. Research Volume (Where papers come from)
Top 5 countries by paper volume:
"""

    for i, (country, count) in enumerate(top_countries.items(), 1):
        pct = 100 * count / with_country
        summary += f"   {i}. **{country}**: {count:,} papers ({pct:.1f}%)\n"

    summary += f"""
### 2. Research Quality (Author expertise)
Top 5 countries by average author h-index (min 10 papers):
"""

    for i, (country, row) in enumerate(country_quality.head(5).iterrows(), 1):
        summary += f"   {i}. **{country}**: mean h-index = {row['mean']:.1f} (n={int(row['count'])})\n"

    # Volume vs Quality insight
    china_vol_rank = list(country_counts.index).index('China') + 1 if 'China' in country_counts.index else None
    us_vol_rank = list(country_counts.index).index('United States') + 1 if 'United States' in country_counts.index else None

    china_qual = country_quality.loc['China', 'mean'] if 'China' in country_quality.index else None
    us_qual = country_quality.loc['United States', 'mean'] if 'United States' in country_quality.index else None

    summary += f"""
### 3. Volume vs Quality Comparison

"""

    if china_vol_rank and us_vol_rank and china_qual and us_qual:
        summary += f"""**China vs United States:**
- China: #{china_vol_rank} by volume, mean h-index = {china_qual:.1f}
- United States: #{us_vol_rank} by volume, mean h-index = {us_qual:.1f}

**Interpretation**: """

        if china_vol_rank < us_vol_rank and china_qual < us_qual:
            summary += "China leads in volume but US authors have higher average h-index, suggesting more submissions from emerging researchers in China."
        elif china_vol_rank < us_vol_rank and china_qual >= us_qual:
            summary += "China leads in both volume and quality metrics."
        else:
            summary += "US leads in volume with comparable or higher quality metrics."

    # Regional patterns
    if 'first_author_region' in df.columns:
        region_stats = df.groupby('first_author_region').agg({
            'first_author_h_index': 'mean',
            'title': 'count'
        })
        region_stats.columns = ['h_index_mean', 'n_papers']
        region_stats = region_stats.sort_values('n_papers', ascending=False)

        summary += f"""

### 4. Regional Distribution
"""
        for region, row in region_stats.iterrows():
            pct = 100 * row['n_papers'] / with_country
            summary += f"- **{region}**: {int(row['n_papers']):,} papers ({pct:.1f}%), mean h-index = {row['h_index_mean']:.1f}\n"

    summary += f"""
## Figures Generated
1. `country_analysis_bars.png` - Volume and quality bar charts
2. `country_map_quality.png` - World map colored by mean h-index
3. `country_map_volume.png` - World map colored by paper count
4. `country_analysis_summary.png` - Combined 4-panel summary figure

## Interpretation Notes

The geographic analysis reveals important patterns:

1. **Volume concentration**: A small number of countries (primarily China and US)
   contribute the majority of submissions, reflecting the global concentration of
   ML/AI research capacity.

2. **Quality variation**: Mean author h-index varies significantly by country,
   with established research hubs (US, UK, Canada) showing higher average author
   reputation. However, this may reflect career stage differences rather than
   inherent quality differences.

3. **Emerging vs Established**: High-volume countries with lower average h-index
   may indicate strong growth in research capacity with many early-career researchers
   entering the field.

4. **Implications for selection analysis**: Geographic variation in author h-index
   should be controlled for when analyzing AI content effects to avoid confounding
   with regional research culture differences.
"""

    # Save summary
    with open(f'{output_dir}/geographic_analysis_summary.md', 'w') as f:
        f.write(summary)

    return summary


def fix_kaleido_colab():
    """
    Helper function to fix kaleido issues in Google Colab.

    Run this at the start of your Colab notebook if you encounter
    kaleido export errors.

    Usage:
        from arxiv_integration.src.visualizations import fix_kaleido_colab
        fix_kaleido_colab()
    """
    import subprocess
    import sys

    print("🔧 Attempting to fix kaleido for Colab...")

    # Step 1: Uninstall existing kaleido
    try:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "kaleido"],
                      capture_output=True, check=False)
    except Exception:
        pass

    # Step 2: Install specific version that works in Colab
    try:
        # Version 0.1.0.post1 is known to work better in Colab
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "kaleido==0.1.0.post1"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✅ Installed kaleido==0.1.0.post1")
        else:
            # Try latest version
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-U", "kaleido"],
                capture_output=True
            )
            print("✅ Installed latest kaleido")
    except Exception as e:
        print(f"⚠️ Could not install kaleido: {e}")
        return False

    # Step 3: Reload the module
    try:
        import importlib
        import kaleido
        importlib.reload(kaleido)
        global HAS_KALEIDO
        HAS_KALEIDO = True
        print("✅ Kaleido is ready!")
        print("   ⚠️ You may need to restart the runtime for changes to take effect.")
        return True
    except ImportError:
        print("⚠️ Kaleido import failed. Please restart the runtime.")
        return False


def save_plotly_figure(fig, path: str, **kwargs) -> bool:
    """
    Public wrapper for saving plotly figures with automatic fallback.

    This is the recommended way to save plotly figures in this project.
    It handles kaleido issues gracefully.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The plotly figure to save
    path : str
        Output path (supports .png, .pdf, .svg, .html)
    **kwargs : dict
        Additional arguments passed to write_image (e.g., scale=2)

    Returns
    -------
    bool : True if static image export succeeded, False otherwise
           (HTML is always saved as backup)

    Example
    -------
    >>> fig = px.scatter(df, x='x', y='y')
    >>> save_plotly_figure(fig, 'output/my_figure.png', scale=2)
    """
    if not HAS_PLOTLY:
        warnings.warn("Plotly not available")
        return False

    # Always save HTML backup first
    html_path = path.rsplit('.', 1)[0] + '.html'
    try:
        fig.write_html(html_path)
    except Exception as e:
        warnings.warn(f"Could not save HTML: {e}")

    # For HTML requests, we're done
    if path.endswith('.html'):
        return True

    # Try to save static image
    return _safe_write_image(fig, path, **kwargs)
