# ============================================================
# real_data/preprocess.py
#
# Dataset preprocessing pipeline
# ============================================================

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

from fmvmm.utils.utils_dmm import multiplicative_replacement


# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

def load_dataset(path):

    data = pd.read_csv(path)

    return data


# ------------------------------------------------------------
# Remove zero columns
# ------------------------------------------------------------

def remove_zero_columns(df):

    return df.loc[:, (df != 0).all(axis=0)]


# ------------------------------------------------------------
# Encode labels
# ------------------------------------------------------------

def encode_labels(labels):

    le = LabelEncoder()

    y = le.fit_transform(labels)

    return y, le


# ------------------------------------------------------------
# Compositional transform
# ------------------------------------------------------------

def compositional_transform(df):

    X = multiplicative_replacement(df)

    return pd.DataFrame(X, columns=df.columns)


# ------------------------------------------------------------
# Full preprocessing pipeline
# ------------------------------------------------------------

def preprocess_dataset(data):

    # ------------------------------------
    # Separate labels
    # ------------------------------------

    labels = data["Class"]

    X = data.drop(["Class"], axis=1)

    n_samples = X.shape[0]
    n_genes_before = X.shape[1]

    # ------------------------------------
    # Remove zero columns
    # ------------------------------------

    X = remove_zero_columns(X)

    n_genes_after = X.shape[1]

    # ------------------------------------
    # Encode labels
    # ------------------------------------

    y, label_encoder = encode_labels(labels)

    # ------------------------------------
    # Compositional transform
    # ------------------------------------

    X_comp = compositional_transform(X)

    gene_names = list(X_comp.columns)

    summary = {
        "n_samples": n_samples,
        "n_genes_before": n_genes_before,
        "n_genes_after": n_genes_after,
        "class_counts": dict(pd.Series(y).value_counts()),
    }

    return {
        "X": X_comp,
        "y": y,
        "gene_names": gene_names,
        "summary": summary,
        "label_encoder": label_encoder,
    }
