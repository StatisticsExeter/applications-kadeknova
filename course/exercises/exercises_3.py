import plotly.express as px


def box_plot(df, cat_var, cont_var):
    """Given
      - data frame df,
      - a string 'cat_var' denoting a categorical variable and
      - a string 'continuous_var' denoting a continuous variable as well as
    return a box plot as a plotly express object
    which summarises the distribution of the continuous variable for
    different levels of cat_var."""
    fig = px.box(df, x=cat_var, y=cont_var)
    return fig


def scatterplot(df, xvar, yvar):
    """Given
      - data frame df,
      - a string 'xvar' denoting a continuous variable and
      - a string 'yvar' denoting a continuous variable
    return a scatterplot plot as a plotly express object
    of the x variable against the y variable."""
    fig = px.scatter(df, x=xvar, y=yvar)
    return fig


def scatterplot_groups(df, xvar, yvar, groups):
    """Given
      - data frame df,
      - a string 'xvar' denoting a continuous variable and
      - a string 'yvar' denoting a continuous variable and
      - a string 'groups' denoting a categorical variable
    return a scatterplot plot as a plotly express object
    of the x variable against the y variable that has
    markers colours for different levels of the grouping variable."""
    fig = px.scatter(df, x=xvar, y=yvar, color=groups)
    return fig


def scatterplot_matrix(df, numeric_cols):
    """Given
      - data frame df,
      - a list 'numeric_cols' denoting several of the continuous variables
    return a scatterplot plotmatrix as a plotly express object
    plotting each continuous variable against the others."""
    fig = px.scatter_matrix(df, dimensions=numeric_cols)
    return fig


def bar_chart_means(df, cat_var, continuous_var, labels):
    """Given
      - data frame df,
      - a string 'cat_var' denoting a categorical variable and
      - a string 'continuous_var' denoting a continuous variable as well as
      - a dictionary 'labels' containing a description for the axis
        of the contents of cat_var and continuous_var
    return a bar chart as a plotly express object
    which summarises the mean of the continuous variable by different levels
    of cat_var and labels the axes using the labels dict."""
    df_means = df.groupby(cat_var)[continuous_var].mean().reset_index()
    fig = px.bar(df_means, x=cat_var, y=continuous_var, labels=labels,
                 title=f"Average {continuous_var} by {cat_var}")
    return fig


def stacked_bar_counts(df, cat_var_1, cat_var_2, labels):
    """Given
      - data frame df,
      - a string 'cat_var_1' denoting a categorical variable for the x axis
      - a string 'cat_var_2' denoting another categorical variable for the legend
      - a dictionary 'labels' containing a description for the axis
        of the contents of cat_var_1 and cat_var_2
    return a bar chart as a plotly express object
    which summarises the mean of the continuous variable by different levels
    of cat_var and labels the axes using the labels dict."""
    df_counts = df.groupby([cat_var_1, cat_var_2]).size().reset_index(name='count')
    fig = px.bar(df_counts, x=cat_var_1, y='count', color=cat_var_2, labels=labels)
    return fig
