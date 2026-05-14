# sim_dmm/tables.py

import pandas as pd


def df_to_latex_multirow(df: pd.DataFrame,
                         dataset_col="Dataset",
                         float_fmt="{:.4f}"):
    """
    Expects df columns:
      Dataset, Parameter,
      true, mean, bias, rmse,
      mean_se, emp_se, mean_ci_len, cover

    Produces a LaTeX tabular with multirow blocks per dataset.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Sort for clean block structure
    # --------------------------------------------------------
    df = df.sort_values([dataset_col, "Parameter"])

    # --------------------------------------------------------
    # Format numeric columns
    # --------------------------------------------------------
    numeric_cols = [
        "true",
        "mean",
        "bias",
        "rmse",
        "mean_se",
        "emp_se",
        "mean_ci_len",
        "cover"
    ]

    for c in numeric_cols:
        if c in df.columns:
            df[c] = df[c].map(
                lambda x: float_fmt.format(x) if pd.notnull(x) else ""
            )

    # --------------------------------------------------------
    # Begin LaTeX table
    # --------------------------------------------------------
    lines = []
    lines.append(r"\begin{tabular}{llrrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"Dataset & Parameter & True & Mean & Bias & RMSE & Mean(SE) & Emp(SE) & Mean(CI) & Cover \\"
    )
    lines.append(r"\midrule")

    # --------------------------------------------------------
    # Multirow per dataset
    # --------------------------------------------------------
    for ds, g in df.groupby(dataset_col, sort=False):

        g = g.reset_index(drop=True)
        m = len(g)

        for i in range(m):
            row = g.loc[i]

            ds_cell = rf"\multirow{{{m}}}{{*}}{{{ds}}}" if i == 0 else ""

            lines.append(
                rf"{ds_cell} & {row['Parameter']} & "
                rf"{row['true']} & {row['mean']} & {row['bias']} & {row['rmse']} & "
                rf"{row['mean_se']} & {row['emp_se']} & "
                rf"{row['mean_ci_len']} & {row['cover']} \\"
            )

        lines.append(r"\midrule")

    # Replace final midrule with bottomrule
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")

    return "\n".join(lines)
