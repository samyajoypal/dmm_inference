# ============================================================
# real_data/tables.py
#
# Table formatting and export helpers
# ============================================================

import pandas as pd

from io_utils import save_csv, save_latex


# ------------------------------------------------------------
# Generic formatter
# ------------------------------------------------------------

def round_dataframe(df, digits=4):
    """
    Round numeric columns only.
    """
    out = df.copy()

    numeric_cols = out.select_dtypes(include=["number"]).columns
    out[numeric_cols] = out[numeric_cols].round(digits)

    return out


# ------------------------------------------------------------
# Save both CSV and LaTeX
# ------------------------------------------------------------

def save_table(df, csv_path, latex_path, digits=4):
    """
    Save a table to CSV and LaTeX after rounding.
    """
    out = round_dataframe(df, digits=digits)
    save_csv(out, csv_path)
    save_latex(out, latex_path)


# ------------------------------------------------------------
# Data summary table
# ------------------------------------------------------------

def make_data_summary_table(summary_dict):
    """
    Convert summary dictionary into a flat table.
    """
    class_counts = summary_dict.get("class_counts", {})

    rows = [{
        "n_samples": summary_dict.get("n_samples"),
        "n_genes_before": summary_dict.get("n_genes_before"),
        "n_genes_after": summary_dict.get("n_genes_after"),
        "class_0_count": class_counts.get(0, None),
        "class_1_count": class_counts.get(1, None),
    }]

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Parameter summary tables
# ------------------------------------------------------------

def make_pi_table(pi_hat):
    return pd.DataFrame({
        "Cluster": [f"Cluster_{i+1}" for i in range(len(pi_hat))],
        "pi_hat": pi_hat
    })


def make_precision_table(s_hat):
    return pd.DataFrame({
        "Cluster": [f"Cluster_{i+1}" for i in range(len(s_hat))],
        "s_hat": s_hat
    })
