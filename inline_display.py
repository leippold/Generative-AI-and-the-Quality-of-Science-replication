"""
inline_display.py - Wrapper for displaying analysis outputs inline in Jupyter/Colab

Add this file to your repo and use these functions instead of plt.savefig()
to both save AND display figures inline.

Usage:
    from inline_display import save_and_show, show_dataframe, show_latex_table
    
    # Instead of: fig.savefig('output.png')
    # Use:
    save_and_show(fig, 'output/figures/my_plot.png', title='My Analysis')
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Check if we're in a notebook environment
try:
    from IPython.display import display, HTML, Image, Markdown
    from IPython import get_ipython
    IN_NOTEBOOK = get_ipython() is not None
except ImportError:
    IN_NOTEBOOK = False
    display = print


def save_and_show(fig, filepath, title=None, dpi=150, formats=None):
    """
    Save a matplotlib figure to disk AND display it inline.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure to save and display
    filepath : str
        Path to save the figure (e.g., 'output/figures/plot.png')
    title : str, optional
        Title to display above the figure
    dpi : int
        Resolution for saved figure
    formats : list, optional
        Additional formats to save (e.g., ['pdf', 'svg'])
    
    Example:
    --------
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y)
    ax.set_title('My Plot')
    save_and_show(fig, 'output/figures/my_plot.png', title='Figure 1: Results')
    """
    # Ensure directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Save main format
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    # Save additional formats if requested
    if formats:
        base_path = str(Path(filepath).with_suffix(''))
        for fmt in formats:
            fig.savefig(f"{base_path}.{fmt}", dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
    
    # Display inline if in notebook
    if IN_NOTEBOOK:
        if title:
            display(HTML(f"<h3>📊 {title}</h3>"))
        
        # Show the figure
        display(fig)
        
        # Show save location
        display(HTML(f"<p style='color:gray;font-size:12px'>💾 Saved: {filepath}</p>"))
    else:
        if title:
            print(f"\n{'='*60}")
            print(f"📊 {title}")
            print(f"{'='*60}")
        print(f"Saved: {filepath}")
    
    plt.close(fig)


def show_dataframe(df, title=None, max_rows=20, max_cols=None):
    """
    Display a DataFrame with nice formatting.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to display
    title : str, optional
        Title to show above the table
    max_rows : int
        Maximum rows to display
    max_cols : int, optional
        Maximum columns to display
    """
    if IN_NOTEBOOK:
        if title:
            display(HTML(f"<h3>📋 {title}</h3>"))
        
        # Apply styling
        styled = df.head(max_rows)
        if max_cols:
            styled = styled.iloc[:, :max_cols]
        
        display(styled.style.set_properties(**{
            'text-align': 'left',
            'font-size': '12px'
        }).set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#f0f0f0'), ('font-weight', 'bold')]}
        ]))
        
        if len(df) > max_rows:
            display(HTML(f"<p style='color:gray'>... showing {max_rows} of {len(df)} rows</p>"))
    else:
        if title:
            print(f"\n{title}")
            print("-" * len(title))
        print(df.head(max_rows).to_string())


def show_latex_table(latex_str, title=None, save_path=None):
    """
    Display LaTeX table code and optionally save to file.
    
    Parameters:
    -----------
    latex_str : str
        LaTeX table code
    title : str, optional
        Title for the table
    save_path : str, optional
        Path to save the .tex file
    """
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(latex_str)
    
    if IN_NOTEBOOK:
        if title:
            display(HTML(f"<h3>📄 {title}</h3>"))
        
        # Show in a code block
        display(HTML(f"<pre style='background:#f5f5f5;padding:10px;overflow-x:auto;'>{latex_str[:3000]}</pre>"))
        
        if save_path:
            display(HTML(f"<p style='color:gray;font-size:12px'>💾 Saved: {save_path}</p>"))
    else:
        if title:
            print(f"\n{title}")
        print(latex_str[:3000])
        if save_path:
            print(f"Saved: {save_path}")


def show_statistics(stats_dict, title=None):
    """
    Display a dictionary of statistics nicely formatted.
    
    Parameters:
    -----------
    stats_dict : dict
        Dictionary of statistic names and values
    title : str, optional
        Title for the statistics block
    """
    if IN_NOTEBOOK:
        if title:
            display(HTML(f"<h3>📈 {title}</h3>"))
        
        html = "<table style='border-collapse:collapse;'>"
        for key, value in stats_dict.items():
            if isinstance(value, float):
                value = f"{value:.4f}" if abs(value) < 0.01 else f"{value:.2f}"
            html += f"<tr><td style='padding:5px;border-bottom:1px solid #ddd;'><b>{key}</b></td>"
            html += f"<td style='padding:5px;border-bottom:1px solid #ddd;'>{value}</td></tr>"
        html += "</table>"
        display(HTML(html))
    else:
        if title:
            print(f"\n{title}")
            print("-" * len(title))
        for key, value in stats_dict.items():
            print(f"  {key}: {value}")


def section_header(title, level=1):
    """
    Display a section header in the notebook.
    """
    if IN_NOTEBOOK:
        symbols = {1: "═", 2: "─", 3: "·"}
        symbol = symbols.get(level, "─")
        display(HTML(f"<h{level+1}>{symbol*3} {title} {symbol*3}</h{level+1}>"))
    else:
        width = 60
        if level == 1:
            print(f"\n{'#' * width}")
            print(f"# {title.upper()}")
            print(f"{'#' * width}\n")
        else:
            print(f"\n{'=' * width}")
            print(f"{title}")
            print(f"{'=' * width}\n")


# Convenience function to wrap existing analysis functions
def with_inline_display(analysis_func):
    """
    Decorator to wrap analysis functions and display their outputs inline.
    
    Usage:
    ------
    @with_inline_display
    def my_analysis(data, output_dir):
        fig, ax = plt.subplots()
        # ... do analysis ...
        return {'figure': fig, 'stats': {...}}
    """
    def wrapper(*args, **kwargs):
        result = analysis_func(*args, **kwargs)
        
        if isinstance(result, dict):
            if 'figure' in result and result['figure'] is not None:
                display(result['figure'])
                plt.close(result['figure'])
            if 'stats' in result:
                show_statistics(result['stats'])
            if 'dataframe' in result:
                show_dataframe(result['dataframe'])
        
        return result
    
    return wrapper
