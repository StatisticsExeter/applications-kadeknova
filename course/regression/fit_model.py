import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path
from course.utils import find_project_root
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats

VIGNETTE_DIR = Path('data_cache') / 'vignettes' / 'regression'


def _fit_model(df):
    """Given data frame df containing columns 'shortfall', 'n_rooms', 'age' and
    'local_authority_code'
    Fit a linear mixed model with shortfall as the response variable
    n_rooms and age as fixed predictors
    with local_authority_code as a random effect"""
    model = smf.mixedlm(
      "shortfall ~ n_rooms + age",
      data=df,
      groups=df["local_authority_code"]
    )
    results = model.fit()
    return results


def _save_model_summary(model, outpath):
    with open(outpath, "w") as f:
        f.write(model.summary().as_text())


def _random_effects(results):
    re_df = pd.DataFrame(results.random_effects).T
    re_df.columns = ['Intercept'] + [f"Slope_{i}" for i in range(len(re_df.columns)-1)]
    re_df['group'] = re_df.index
    stderr = np.sqrt(results.cov_re.iloc[0, 0])
    re_df['lower'] = re_df['Intercept'] - 1.96 * stderr
    re_df['upper'] = re_df['Intercept'] + 1.96 * stderr
    re_df = re_df.sort_values('Intercept')
    return re_df


def fit_model():
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'la_energy.csv')
    results = _fit_model(df)
    outpath = VIGNETTE_DIR / 'model_fit.txt'
    _random_effects(results).to_csv(base_dir / 'data_cache' / 'models' / 'reffs.csv')
    _save_model_summary(results, outpath)
    _residual_diagnostics(results, df)
    
    
def _residual_diagnostics(results, df):
    """Generate residual diagnostic plots and save them as HTML files."""

    # Extract residuals and fitted values
    df['fitted'] = results.fittedvalues
    df['residuals'] = results.resid

    # 1. Histogram of residuals
    fig_hist = px.histogram(df, x='residuals', nbins=40,
                            title="Residual Histogram")
    fig_hist.write_html(VIGNETTE_DIR / "residual_histogram.html")

    # 2. QQ-plot
    qq = stats.probplot(df['residuals'], dist="norm")
    fig_qq = go.Figure()
    fig_qq.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1],
                                mode='markers',
                                name='Residuals'))
    fig_qq.add_trace(go.Scatter(x=qq[0][0], y=qq[0][0],
                                mode='lines',
                                name='45-degree line'))
    fig_qq.update_layout(title="QQ Plot of Residuals",
                         xaxis_title="Theoretical Quantiles",
                         yaxis_title="Sample Quantiles")
    fig_qq.write_html(VIGNETTE_DIR / "residual_qqplot.html")

    # 3. Residuals vs Fitted
    fig_fitted = px.scatter(df, x='fitted', y='residuals',
                            title="Residuals vs Fitted Values")
    fig_fitted.add_hline(y=0, line_dash="dash")
    fig_fitted.write_html(VIGNETTE_DIR / "residuals_vs_fitted.html")

    # 4. Residuals vs n_rooms
    fig_rooms = px.scatter(df, x='n_rooms', y='residuals',
                           title="Residuals vs Number of Rooms")
    fig_rooms.add_hline(y=0, line_dash="dash")
    fig_rooms.write_html(VIGNETTE_DIR / "residuals_vs_rooms.html")

    # 5. Residuals vs age
    fig_age = px.box(df, x='age', y='residuals',
                     title="Residuals by Age Category")
    fig_age.write_html(VIGNETTE_DIR / "residuals_vs_age.html")

