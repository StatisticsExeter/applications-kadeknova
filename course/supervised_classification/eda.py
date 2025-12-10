import pandas as pd
import plotly.express as px
from pathlib import Path
from course.utils import find_project_root
from plotly.subplots import make_subplots
import plotly.graph_objects as go




VIGNETTE_DIR = Path('data_cache') / 'vignettes' / 'supervised_classification'


def plot_scatter():
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'energy.csv')
    outpath = base_dir / VIGNETTE_DIR / 'scatterplot.html'
    title = "Energy variables showing different built_age type"
    fig = scatter_onecat(df, 'built_age', title)
    fig.write_html(outpath)


def scatter_onecat(df, cat_column, title):
    """Return a plotly express figure which is a scatterplot of all numeric columns in df
    with markers/colours given by the text in column cat_column
    and overall title specfied by title"""
    numeric_cols = df.select_dtypes(include="number").columns
    fig = px.scatter_matrix(df, dimensions=numeric_cols, color=cat_column, title=title)
    return fig


def get_frequencies(df, cat_column):
    return df[cat_column].value_counts()


def get_grouped_stats(df, cat_column):
    numeric_cols = df.select_dtypes(include='number').columns
    grouped_stats = df.groupby(cat_column)[numeric_cols].describe()
    grouped_stats.columns = ['{}_{}'.format(var, stat) for var, stat in grouped_stats.columns]
    return grouped_stats.transpose()


def get_summary_stats():
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'energy.csv')
    cat_column = 'built_age'
    frequencies = get_frequencies(df, cat_column)
    outpath_f = base_dir / VIGNETTE_DIR / 'frequencies.csv'
    frequencies.to_csv(outpath_f)
    summary_stats = get_grouped_stats(df, cat_column)
    outpath_s = base_dir / VIGNETTE_DIR / 'grouped_stats.csv'
    summary_stats.to_csv(outpath_s)


def plot_histograms_combined():
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'energy.csv')

    numeric_cols = df.select_dtypes(include="number").columns

    # Buat layout grid (2 kolom)
    n = len(numeric_cols)
    rows = (n + 1) // 2    # baris cukup untuk menampung 2 kolom
    cols = 2

    # Membuat subplot figure
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"{col}" for col in numeric_cols],
        horizontal_spacing=0.12,
        vertical_spacing=0.12
    )

    row = 1
    col = 1

    for col_name in numeric_cols:

        # Generate histogram plotly express
        h = px.histogram(
            df, x=col_name, color='built_age',
            barmode='overlay'
        )

        # Tambahkan semua trace histogram ke subplot
        for trace in h.data:
            fig.add_trace(trace, row=row, col=col)

        # Update lokasi subplot
        col += 1
        if col > cols:
            col = 1
            row += 1

    # Layout styling
    fig.update_layout(
        height=300 * rows,
        width=900,
        title="Combined Histograms of Energy Variables by Built Age",
        showlegend=False  # legend umum biasanya mengganggu tata letak
    )

    # Output path
    outpath = base_dir / VIGNETTE_DIR / 'histograms_combined.html'
    fig.write_html(outpath)

    return fig

