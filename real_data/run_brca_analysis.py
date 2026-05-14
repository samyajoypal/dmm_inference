# ============================================================
# real_data/run_brca_analysis.py
# ============================================================

from config import DATASETS
from run_real_data_analysis import run_real_data_analysis


if __name__ == "__main__":
    run_real_data_analysis(
        dataset_name="brca",
        dataset_path=DATASETS["brca"]
    )
