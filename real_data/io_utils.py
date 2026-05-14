# ============================================================
# real_data/io_utils.py
#
# File I/O helpers
# ============================================================

import os
import pandas as pd


# ------------------------------------------------------------
# Directory utilities
# ------------------------------------------------------------

def ensure_dirs(dir_dict):

    for d in dir_dict.values():
        os.makedirs(d, exist_ok=True)


# ------------------------------------------------------------
# Save CSV
# ------------------------------------------------------------

def save_csv(df, path):

    df.to_csv(path, index=False)


# ------------------------------------------------------------
# Save LaTeX
# ------------------------------------------------------------

def save_latex(df, path, float_format="%.4f"):

    latex_str = df.to_latex(
        index=False,
        float_format=float_format
    )

    with open(path, "w") as f:
        f.write(latex_str)
