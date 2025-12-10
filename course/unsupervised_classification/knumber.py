# course/unsupervised_classification/knumber.py

import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage


# =========================================================
# ensure ouput directory
# =========================================================
def _ensure_outdir():
    """
    Ensure output directory exists:
    data_cache/vignettes/unsupervised_classification
    """
    outdir = Path("data_cache/vignettes/unsupervised_classification")
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


# =========================================================
# 1. K-MEANS ELBOW PLOT (WSS)
# =========================================================
def kmeans_elbow(df, k_range=range(2, 11),
                 filename="kmeans_elbow.html"):

    outdir = _ensure_outdir()
    wss = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(df)
        wss.append(km.inertia_)

    fig = px.line(
        x=list(k_range),
        y=wss,
        markers=True,
        title="K-Means Elbow Plot (WSS)",
        labels={"x": "Number of Clusters (K)", "y": "Within-Cluster Sum of Squares (WSS)"}
    )

    fig.update_layout(width=900, height=500)

    # Save HTML
    outpath = outdir / filename
    fig.write_html(outpath)

    return wss



# =========================================================
# 2. SILHOUETTE SCORES FOR K-MEANS
# =========================================================
def kmeans_silhouette(df, k_range=range(2, 11),
                      filename="kmeans_silhouette.html"):

    outdir = _ensure_outdir()
    sil_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(df)
        sil = silhouette_score(df, labels)
        sil_scores.append(sil)

    fig = px.line(
        x=list(k_range),
        y=sil_scores,
        markers=True,
        title="K-Means Silhouette Score",
        labels={"x": "Number of Clusters (K)", "y": "Average Silhouette Score"}
    )

    fig.update_layout(width=900, height=500)

    # Save HTML
    outpath = outdir / filename
    fig.write_html(outpath)

    return sil_scores



# =========================================================
# 3. GMM BIC & AIC SCORES
# =========================================================
def gmm_bic_aic(df, k_range=range(2, 11),
                filename="gmm_bic_aic.html"):

    outdir = _ensure_outdir()
    bic_scores = []
    aic_scores = []

    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(df)
        bic_scores.append(gmm.bic(df))
        aic_scores.append(gmm.aic(df))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(k_range), y=bic_scores,
        mode="lines+markers", name="BIC"
    ))
    fig.add_trace(go.Scatter(
        x=list(k_range), y=aic_scores,
        mode="lines+markers", name="AIC"
    ))

    fig.update_layout(
        title="GMM Model Selection: BIC & AIC",
        xaxis_title="Number of Components (K)",
        yaxis_title="Score (Lower is Better)",
        width=900,
        height=500
    )

    # Save HTML
    outpath = outdir / filename
    fig.write_html(outpath)

    return bic_scores, aic_scores
