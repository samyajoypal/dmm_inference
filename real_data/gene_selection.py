# ============================================================
# real_data/gene_selection.py
#
# Gene selection for interpretation
# ============================================================

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Data-driven gene selection
# ------------------------------------------------------------

def select_top_genes_data_driven(mu_hat, gene_names, top_n=5):

    if mu_hat.shape[0] != 2:
        raise ValueError("Data-driven selection currently assumes K=2.")

    diff = np.abs(mu_hat[0] - mu_hat[1])

    idx = np.argsort(diff)[::-1][:top_n]

    genes = [gene_names[i] for i in idx]

    values = diff[idx]

    df = pd.DataFrame({
        "Gene": genes,
        "AbsDifference": values
    })

    return genes, df


# ------------------------------------------------------------
# Known gene selection
# ------------------------------------------------------------

def select_known_genes(gene_names, known_genes, top_n=5):

    present = [g for g in known_genes if g in gene_names]

    selected = present[:top_n]

    df = pd.DataFrame({
        "Gene": selected
    })

    return selected, df


# ------------------------------------------------------------
# Gene difference summary
# ------------------------------------------------------------

def compute_gene_differences(mu_hat, gene_names, genes):

    gene_idx = [gene_names.index(g) for g in genes]

    mu1 = mu_hat[0, gene_idx]

    mu2 = mu_hat[1, gene_idx]

    diff = mu1 - mu2

    df = pd.DataFrame({
        "Gene": genes,
        "mu_cluster1": mu1,
        "mu_cluster2": mu2,
        "difference": diff
    })

    return df
