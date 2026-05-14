# ============================================================
# real_data/fit_model.py
#
# Fit DMM and extract parameters
# ============================================================

import numpy as np
import pandas as pd

from fmvmm.mixtures.DMM_Soft import DMM_Soft
from fmvmm.utils.utils_mixture import clustering_metrics
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score

# ------------------------------------------------------------
# Fit DMM
# ------------------------------------------------------------

def fit_dmm_model(X, n_clusters=2):

    model = DMM_Soft(
        n_clusters=n_clusters,
        method="highdimensional",
        initialization="kmeans",
        print_log_likelihood=True,
    )

    model.fit(X)

    return model


# ------------------------------------------------------------
# Extract parameters
# ------------------------------------------------------------

def extract_parameters(model, gene_names):

    pi_hat = np.array(model.pi_new)

    alpha_hat = np.array(model.alpha_new)

    K, p = alpha_hat.shape

    # cluster means
    mu_hat = alpha_hat / alpha_hat.sum(axis=1, keepdims=True)

    # cluster precisions
    s_hat = alpha_hat.sum(axis=1)

    mu_df = pd.DataFrame(mu_hat, columns=gene_names)

    alpha_df = pd.DataFrame(alpha_hat, columns=gene_names)

    params = {
        "pi_hat": pi_hat,
        "alpha_hat": alpha_hat,
        "mu_hat": mu_hat,
        "s_hat": s_hat,
        "mu_df": mu_df,
        "alpha_df": alpha_df,
    }

    return params


# ------------------------------------------------------------
# Clustering evaluation
# ------------------------------------------------------------

##def evaluate_clustering(true_labels, pred_labels):
##
##    metrics = clustering_metrics(true_labels, pred_labels)
##
##    metrics_df = pd.DataFrame([metrics])
##
##    return metrics_df

def evaluate_clustering(true_labels, pred_labels):
    
    metrics = clustering_metrics(true_labels, pred_labels)
    
    metrics_df = pd.DataFrame([metrics])
    
    # Safely append new metrics directly to the dataframe 
    # (prevents errors if clustering_metrics returns an immutable namedtuple)
    metrics_df["ARI"] = adjusted_rand_score(true_labels, pred_labels)
    metrics_df["NMI"] = normalized_mutual_info_score(true_labels, pred_labels)
    metrics_df["HMS"] = homogeneity_score(true_labels, pred_labels)
    
    return metrics_df

# ------------------------------------------------------------
# Full pipeline
# ------------------------------------------------------------

def fit_and_summarize(X, y, gene_names):

    model = fit_dmm_model(X)

    pred = model.predict()

    params = extract_parameters(model, gene_names)

    metrics_df = evaluate_clustering(y, pred)

    results = {
        "model": model,
        "pred_labels": pred,
        "params": params,
        "metrics": metrics_df,
    }

    return results
