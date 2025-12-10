from scipy.cluster.hierarchy import linkage, fcluster
import plotly.figure_factory as ff
import plotly.express as px
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path
from course.utils import find_project_root
from sklearn.mixture import GaussianMixture
from course.unsupervised_classification.utils_cluster import summarize_clusters

VIGNETTE_DIR = Path('data_cache') / 'vignettes' / 'unsupervised_classification'


def hcluster_analysis():
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'la_collision.csv')
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    linked = _fit_dendrogram(df_scaled)  # Z dengan ward
    outpath = base_dir / VIGNETTE_DIR / 'dendrogram.html'
    fig = _plot_dendrogram(linked)
    fig.write_html(outpath)


def hierarchical_groups(height):
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'la_collision.csv')
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    linked = _fit_dendrogram(df_scaled)
    clusters = _cutree(linked, height)  # adjust this value based on dendrogram scale
    df_plot = _pca(df_scaled)
    df_plot['cluster'] = clusters.astype(str)  # convert to string for color grouping
    outpath = base_dir / VIGNETTE_DIR / 'hscatter.html'
    fig = _scatter_clusters(df_plot)
    fig.write_html(outpath)
    df_original = pd.read_csv(base_dir / 'data_cache' / 'la_collision.csv')
    summarize_clusters(df_original, clusters['cluster'], filename="hierarchical_summary.csv")


def _fit_dendrogram(df):
    """Given a dataframe containing only suitable values
    Return a scipy.cluster.hierarchy hierarchical clustering solution to these data"""
    Z = linkage(df, method='ward')  # menghitung linkage untuk dendrogram
    return Z


def _plot_dendrogram(Z):
    """Plot dendrogram using Plotly Figure Factory with correct linkage"""
    tree = ff.create_dendrogram(
        Z,
        orientation='bottom',
        labels=[str(i) for i in range(Z.shape[0]+1)],  # jumlah objek = n_samples
        linkagefun=lambda x: Z  # pakai linkage matrix yang sudah dihitung
    )
    tree.update_layout(title_text='Interactive Hierarchical Clustering Dendrogram')
    return tree


def _cutree(tree, height):
    """Given a scipy.cluster.hierarchy hierarchical clustering solution and a float of the height
    Cut the tree at that hight and return the solution (cluster group membership) as a
    data frame with one column called 'cluster'"""
    clusters = fcluster(tree, t=height, criterion='distance')
    return pd.DataFrame({'cluster': clusters})


def _pca(df):
    """Given a dataframe of only suitable variables
    return a dataframe of the first two pca predictions (z values) with columns 'PC1' and 'PC2'"""
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(df)
    return pd.DataFrame(pcs, columns=['PC1', 'PC2'])


def _scatter_clusters(df):
    """Given a data frame containing columns 'PC1' and 'PC2' and 'cluster'
      (the first two principal component projections and the cluster groups)
    return a plotly express scatterplot of PC1 versus PC2
    with marks to denote cluster group membership"""
    fig = px.scatter(
      df,
      x='PC1',
      y='PC2',
      color='cluster',
      title='PCA Scatter Plot Colored by Cluster Labels'
    )
    return fig

# adding new clustering methods


def _gmm_clusters(df_scaled, n_components=4):
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    labels = gmm.fit_predict(df_scaled)
    return pd.DataFrame({'cluster': labels.astype(str)})


def gmm_analysis(n_components=4):
    base = find_project_root()
    df = pd.read_csv(base / "data_cache" / "la_collision.csv")

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    clusters = _gmm_clusters(df_scaled, n_components)
    df_pca = _pca(df_scaled)
    df_pca['cluster'] = clusters['cluster']

    outpath = base / VIGNETTE_DIR / "gmm_scatter.html"
    fig = _scatter_clusters(df_pca)
    fig.write_html(outpath)
    summarize_clusters(df, clusters['cluster'], filename="gmm_summary.csv")
