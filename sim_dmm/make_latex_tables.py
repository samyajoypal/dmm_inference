# ============================================================
# sim_dmm/make_latex_tables.py
#
# Create clean LaTeX tables per dataset:
#   (1) pi + alpha
#   (2) pi + mu + tau
#
# Uses existing summary CSV output.
# Does NOT modify simulation code.
# ============================================================

import os
import pandas as pd
import numpy as np


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

INPUT_FILE = "out/clean/paper_summary_all.csv"   # <-- change if needed
OUTPUT_DIR = "out/paper/latex_tables"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------

def fmt(x):
    """Nice numeric formatting for paper."""
    if abs(x) < 1e-3:
        return f"{x:.2e}"
    if abs(x) >= 100:
        return f"{x:.2f}"
    if abs(x) >= 10:
        return f"{x:.3f}"
    return f"{x:.4f}"


def clean_param_name(p):
    """Ensure LaTeX formatting is preserved."""
    return p


def write_latex_table(df, filename, caption):
    """
    Write standalone LaTeX table.
    """
    lines = []

    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\hline")
    lines.append("Parameter & True & Mean & Bias & RMSE & Cover \\\\")
    lines.append("\\hline")

    for _, row in df.iterrows():
        lines.append(
            f"{clean_param_name(row['Parameter'])} & "
            f"{fmt(row['true'])} & "
            f"{fmt(row['mean'])} & "
            f"{fmt(row['bias'])} & "
            f"{fmt(row['rmse'])} & "
            f"{fmt(row['cover'])} \\\\"
        )

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append("\\end{table}")

    with open(os.path.join(OUTPUT_DIR, filename), "w") as f:
        f.write("\n".join(lines))


# ------------------------------------------------------------
# Main logic
# ------------------------------------------------------------

def main():

    df = pd.read_csv(INPUT_FILE)

    datasets = sorted(df["Dataset"].unique())

    for d in datasets:

        sub = df[df["Dataset"] == d].copy()

        # ----------------------------------------------------
        # Table 1: pi + alpha
        # ----------------------------------------------------

        mask_alpha = (
            sub["Parameter"].str.contains("\\\\pi") |
            sub["Parameter"].str.contains("\\\\alpha")
        )

        df_alpha = sub[mask_alpha].copy()

        write_latex_table(
            df_alpha,
            filename=f"table_dataset{d}_alpha.tex",
            caption=f"Simulation results for Dataset {d}: mixture weights and Dirichlet parameters."
        )

        # ----------------------------------------------------
        # Table 2: pi + mu + tau
        # ----------------------------------------------------

        mask_mu_tau = (
            sub["Parameter"].str.contains("\\\\pi") |
            sub["Parameter"].str.contains("\\\\mu") |
            sub["Parameter"].str.contains("\\\\tau")
        )

        df_mu_tau = sub[mask_mu_tau].copy()

        write_latex_table(
            df_mu_tau,
            filename=f"table_dataset{d}_mu_tau.tex",
            caption=f"Simulation results for Dataset {d}: mixture weights, means and precision parameters."
        )

    print("LaTeX tables written to:")
    print("  out/latex_tables/")


# ------------------------------------------------------------
if __name__ == "__main__":
    main()
