# ============================================================
# real_data/config.py
#
# Global configuration for real data analysis
# ============================================================

import os

# ------------------------------------------------------------
# Dataset paths
# ------------------------------------------------------------

DATASETS = {
    "luad": "tcga_luad_filtered.csv",
    "brca": "tcga_brca_filtered2.csv",
}

# ------------------------------------------------------------
# Model settings
# ------------------------------------------------------------

N_CLUSTERS = 2
DMM_METHOD = "highdimensional"

# ------------------------------------------------------------
# Gene selection settings
# ------------------------------------------------------------

TOP_DATA_DRIVEN_GENES = 5
TOP_KNOWN_GENES = 5

# ------------------------------------------------------------
# Random seeds
# ------------------------------------------------------------

RANDOM_STATE = 42

# ------------------------------------------------------------
# Output directories
# ------------------------------------------------------------

BASE_OUTPUT_DIR = "real_data/out"

def dataset_output_dirs(dataset_name):

    base = os.path.join(BASE_OUTPUT_DIR, dataset_name)

    dirs = {
        "base": base,
        "tables": os.path.join(base, "tables"),
        "latex": os.path.join(base, "latex"),
        "figures": os.path.join(base, "figures"),
        "model": os.path.join(base, "model"),
    }

    return dirs
