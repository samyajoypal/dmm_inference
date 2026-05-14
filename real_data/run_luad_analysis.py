# ============================================================
# real_data/run_luad_analysis.py
# ============================================================

from config import DATASETS
from run_real_data_analysis import run_real_data_analysis


if __name__ == "__main__":
    run_real_data_analysis(
        dataset_name="luad",
        dataset_path=DATASETS["luad"]
    )
