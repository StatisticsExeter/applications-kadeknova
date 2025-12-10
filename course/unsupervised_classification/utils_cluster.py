# course/unsupervised_classification/utils_cluster.py

from pathlib import Path


def summarize_clusters(df, labels, filename="cluster_summary.csv"):
    outdir = Path("data_cache/vignettes/unsupervised_classification")
    outdir.mkdir(parents=True, exist_ok=True)

    df_temp = df.copy()
    df_temp["cluster"] = labels

    summary = df_temp.groupby("cluster").mean()

    outpath = outdir / filename
    summary.to_csv(outpath)

    return summary
