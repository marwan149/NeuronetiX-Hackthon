import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from typing import List, Optional, Tuple


def select_cluster_features(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    return df[features].copy()


def elbow_plot(X: pd.DataFrame, max_clusters: int = 10) -> None:
    inertias = []
    cluster_range = list(range(2, max_clusters + 1))

    for n_clusters in cluster_range:
        model = KMeans(n_clusters=n_clusters, random_state=33, n_init=10)
        model.fit(X)
        inertias.append(model.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(cluster_range, inertias, marker="o")
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.show()


def run_kmeans(X: pd.DataFrame, n_clusters: int = 6, random_state: int = 33) -> KMeans:
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    model.fit(X)
    return model


def run_agglomerative(X: pd.DataFrame, n_clusters: int = 6) -> AgglomerativeClustering:
    model = AgglomerativeClustering(n_clusters=n_clusters)
    model.fit(X)
    return model


def plot_cluster_distribution(
    X: pd.DataFrame,
    labels: pd.Series,
    x_col: str = "MonthlyCharges",
    y_col: str = "tenure",
) -> None:
    df = X.copy()
    df["cluster_label"] = labels.values

    plt.figure(figsize=(10, 6))
    for cluster_id in sorted(df["cluster_label"].unique()):
        subset = df[df["cluster_label"] == cluster_id]
        plt.scatter(
            subset[x_col],
            subset[y_col],
            label=f"Cluster {cluster_id}",
            alpha=0.6,
        )

    plt.title("Cluster Distribution")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend(title="Cluster")
    plt.show()


def add_cluster_labels(df: pd.DataFrame, labels: pd.Series, column: str = "cluster") -> pd.DataFrame:
    result = df.copy()
    result[column] = labels.values
    return result
