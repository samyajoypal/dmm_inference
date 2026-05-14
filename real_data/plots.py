# ============================================================
# real_data/plots.py
#
# Plotting utilities for real data analysis
# ============================================================

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# ------------------------------------------------------------
# Side-by-side t-SNE
# ------------------------------------------------------------

def plot_tsne_side_by_side(X, true_labels, pred_labels, save_path, random_state=42):
    """
    Create side-by-side t-SNE plots for true labels and predicted clusters.
    """
    X_array = np.asarray(X, dtype=float)

    tsne = TSNE(n_components=2, random_state=random_state)
    X_tsne = tsne.fit_transform(X_array)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    scatter1 = axes[0].scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        c=true_labels,
        edgecolors="k"
    )
    axes[0].set_title("t-SNE: True Labels")
    axes[0].set_xlabel("Component 1")
    axes[0].set_ylabel("Component 2")
    axes[0].legend(
        handles=scatter1.legend_elements()[0],
        title="True"
    )

    scatter2 = axes[1].scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        c=pred_labels,
        edgecolors="k"
    )
    axes[1].set_title("t-SNE: Predicted Clusters")
    axes[1].set_xlabel("Component 1")
    axes[1].set_ylabel("Component 2")
    axes[1].legend(
        handles=scatter2.legend_elements()[0],
        title="Cluster"
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------
# Data-driven gene difference plot
# ------------------------------------------------------------

def plot_gene_difference_bar(gene_diff_df, save_path, title="Top Gene Composition Differences"):
    """
    Horizontal bar plot of mu_cluster1 - mu_cluster2.
    """
    df = gene_diff_df.copy().sort_values("difference")

    plt.figure(figsize=(8, 5))
    plt.barh(df["Gene"], df["difference"])
    plt.axvline(x=0.0, linestyle="--")
    plt.xlabel("Difference in Estimated Mean Composition")
    plt.ylabel("Gene")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------
# Known gene grouped bar plot
# ------------------------------------------------------------

def plot_known_gene_grouped(gene_df, save_path, title="Known Gene Mean Composition by Cluster"):
    """
    Grouped bar plot for selected genes.
    Expects columns:
        Gene, mu_cluster1, mu_cluster2
    """
    df = gene_df.copy()

    x = np.arange(len(df))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, df["mu_cluster1"], width=width, label="Cluster 1")
    plt.bar(x + width / 2, df["mu_cluster2"], width=width, label="Cluster 2")

    plt.xticks(x, df["Gene"], rotation=45, ha="right")
    plt.ylabel("Estimated Mean Composition")
    plt.xlabel("Gene")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------
# CI plot for gene differences
# ------------------------------------------------------------

def plot_gene_difference_ci(gene_inf_df, save_path, title="Gene Difference with 95% CI"):
    """
    Forest-style plot for gene composition differences and 95% CIs.

    Expects columns:
        Gene, difference, CI_difference_lower, CI_difference_upper
    """
    df = gene_inf_df.copy().sort_values("difference")

    y = np.arange(len(df))
    est = df["difference"].values
    lo = df["CI_difference_lower"].values
    hi = df["CI_difference_upper"].values

    err_left = est - lo
    err_right = hi - est

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        est,
        y,
        xerr=[err_left, err_right],
        fmt="o",
        capsize=4
    )
    plt.axvline(x=0.0, linestyle="--")
    plt.yticks(y, df["Gene"])
    plt.xlabel("Difference in Estimated Mean Composition")
    plt.ylabel("Gene")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
