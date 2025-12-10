import plotly.express as px
import pandas as pd
from pathlib import Path
from course.utils import find_project_root

VIGNETTE_DIR = Path('data_cache') / 'vignettes' / 'regression'


def _boxplot(df, x_var, y_var, title):
    """Given a data frame 'df' containing categorical variable 'x_var'
    and outcome variable 'y_var' produce a box plot of the distribution of the y_variable
    for different levels of y_var. The box plot should have title 'title'"""
    fig = px.box(df, x=x_var, y=y_var, title=title)
    return fig


def boxplot_age():
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'la_energy.csv')
    fig = _boxplot(df, 'age', 'shortfall', 'Shortfall by Age Category')
    fig.write_html(VIGNETTE_DIR / 'boxplot_age.html')


def boxplot_rooms():
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'la_energy.csv')
    fig = _boxplot(df, 'n_rooms', 'shortfall', 'Shortfall by Number of rooms')
    fig.write_html(VIGNETTE_DIR / 'boxplot_rooms.html')
    

def shortfall_distribution():
    """Histogram of the distribution of shortfall (CO2 emissions)."""
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'la_energy.csv')

    fig = px.histogram(df, x='shortfall', nbins=50,
                       title='Distribution of Shortfall (CO2 Emissions)')
    fig.write_html(VIGNETTE_DIR / 'shortfall_distribution.html')


def scatter_rooms_shortfall():
    """Scatter plot of number of rooms vs shortfall with OLS trendline."""
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'la_energy.csv')

    fig = px.scatter(df, x='n_rooms', y='shortfall',
                     trendline='ols',
                     title='Shortfall vs Number of Rooms')
    fig.write_html(VIGNETTE_DIR / 'scatter_rooms_shortfall.html')


def shortfall_by_authority():
    """Boxplot showing variation of shortfall across local authorities."""
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'la_energy.csv')

    fig = px.box(df, 
                 x='local_authority_code', 
                 y='shortfall',
                 title='Variation of Shortfall Across Local Authorities')
    
    # Hide labels (too many authorities → unreadable)
    fig.update_xaxes(showticklabels=False)

    fig.write_html(VIGNETTE_DIR / 'shortfall_by_authority.html')


def mean_shortfall_by_authority():
    """Bar plot of mean shortfall per local authority."""
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'la_energy.csv')

    df_mean = df.groupby('local_authority_code')['shortfall'].mean().reset_index()

    fig = px.bar(df_mean,
                 x='local_authority_code',
                 y='shortfall',
                 title='Mean Shortfall per Local Authority')
    
    fig.update_xaxes(showticklabels=False)
    fig.write_html(VIGNETTE_DIR / 'mean_shortfall_by_authority.html')



def mean_shortfall_age():
    """Bar plot of mean shortfall by age group (Pre1930 vs Recent)."""
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'la_energy.csv')

    df_mean = df.groupby('age')['shortfall'].mean().reset_index()

    fig = px.bar(df_mean, x='age', y='shortfall',
                 title='Mean Shortfall by Age Category')
    fig.write_html(VIGNETTE_DIR / 'mean_shortfall_age.html')
