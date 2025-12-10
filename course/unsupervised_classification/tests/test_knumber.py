# course/unsupervised_classification/tests/test_knumber.py

import numpy as np
import pandas as pd
from pathlib import Path

from course.unsupervised_classification.knumber import (
    kmeans_elbow,
    kmeans_silhouette,
    gmm_bic_aic
)

# Output folder path
OUTDIR = Path("data_cache/vignettes/unsupervised_classification")


# --------------------------------------------------------------------
# Helper: small synthetic dataset for stable testing
# --------------------------------------------------------------------
def small_test_df():
    np.random.seed(123)
    df = pd.DataFrame({
        "x1": np.random.normal(0, 1, 25),
        "x2": np.random.normal(0, 1, 25),
        "x3": np.random.normal(0, 1, 25),
    })
    return df

# --------------------------------------------------------------------
# Test 1: K-means WSS HTML creation
# --------------------------------------------------------------------


def test_kmeans_elbow_html():

    df = small_test_df()
    filename = "test_elbow.html"

    wss = kmeans_elbow(df, k_range=range(2, 6), filename=filename)

    # Correct length
    assert len(wss) == 4

    # Numeric values
    assert all(isinstance(v, float) for v in wss)

    # File exists
    assert (OUTDIR / filename).exists()


# --------------------------------------------------------------------
# Test 2: Silhouette scores HTML creation
# --------------------------------------------------------------------
def test_kmeans_silhouette_html():

    df = small_test_df()
    filename = "test_silhouette.html"

    scores = kmeans_silhouette(df, k_range=range(2, 6), filename=filename)

    assert len(scores) == 4
    assert all(isinstance(v, float) for v in scores)
    assert (OUTDIR / filename).exists()


# --------------------------------------------------------------------
# Test 3: GMM BIC & AIC HTML creation
# --------------------------------------------------------------------
def test_gmm_bic_aic_html():

    df = small_test_df()
    filename = "test_gmm.html"

    bic, aic = gmm_bic_aic(df, k_range=range(2, 6), filename=filename)

    assert len(bic) == 4
    assert len(aic) == 4
    assert all(isinstance(v, float) for v in bic)
    assert all(isinstance(v, float) for v in aic)
    assert (OUTDIR / filename).exists()
