# ============================================================
# real_data/run_real_data_analysis.py
#
# Main reusable pipeline for TCGA real data analysis
# ============================================================

import os
import pandas as pd

from config import (
    N_CLUSTERS,
    TOP_DATA_DRIVEN_GENES,
    TOP_KNOWN_GENES,
    RANDOM_STATE,
    dataset_output_dirs,
)

from known_genes import get_known_genes, filter_known_genes_present
from io_utils import ensure_dirs, save_csv
from preprocess import load_dataset, preprocess_dataset
from fit_model import fit_and_summarize
from gene_selection import (
    select_top_genes_data_driven,
    select_known_genes,
    compute_gene_differences,
)
from inference_utils import (
    run_global_tests,
    gene_difference_inference,
    summarize_pi_inference,
    summarize_precision_inference,
)
from plots import (
    plot_tsne_side_by_side,
    plot_gene_difference_bar,
    plot_known_gene_grouped,
    plot_gene_difference_ci,
)
from tables import (
    save_table,
    make_data_summary_table,
    make_pi_table,
    make_precision_table,
)


# ------------------------------------------------------------
# Main analysis function
# ------------------------------------------------------------

def run_real_data_analysis(dataset_name, dataset_path):

    # ------------------------------------
    # Output directories
    # ------------------------------------

    dirs = dataset_output_dirs(dataset_name)
    ensure_dirs(dirs)

    # ------------------------------------
    # Load and preprocess data
    # ------------------------------------

    data = load_dataset(dataset_path)

    prep = preprocess_dataset(data)

    X = prep["X"]
    y = prep["y"]
    gene_names = prep["gene_names"]
    summary = prep["summary"]

    # ------------------------------------
    # Fit DMM
    # ------------------------------------

    fit_res = fit_and_summarize(
        X=X,
        y=y,
        gene_names=gene_names
    )

    model = fit_res["model"]
    pred_labels = fit_res["pred_labels"]
    params = fit_res["params"]
    metrics_df = fit_res["metrics"]

    mu_hat = params["mu_hat"]
    pi_hat = params["pi_hat"]
    s_hat = params["s_hat"]
    alpha_df = params["alpha_df"]
    mu_df = params["mu_df"]

    # ------------------------------------
    # Gene selection
    # ------------------------------------

    data_driven_genes, data_driven_df = select_top_genes_data_driven(
        mu_hat,
        gene_names,
        top_n=TOP_DATA_DRIVEN_GENES
    )

    known_gene_list = get_known_genes(dataset_name)
    known_gene_present = filter_known_genes_present(known_gene_list, gene_names)

    selected_known_genes, known_gene_df = select_known_genes(
        gene_names,
        known_gene_present,
        top_n=TOP_KNOWN_GENES
    )

    # ------------------------------------
    # Gene difference summaries
    # ------------------------------------

    data_driven_diff_df = compute_gene_differences(
        mu_hat,
        gene_names,
        data_driven_genes
    )

    known_gene_diff_df = compute_gene_differences(
        mu_hat,
        gene_names,
        selected_known_genes
    )

    # ------------------------------------
    # Inference
    # ------------------------------------

    global_tests_df = run_global_tests(model, X)

    pi_inf_df = summarize_pi_inference(model)
    s_inf_df = summarize_precision_inference(model)

    data_driven_inf_df = gene_difference_inference(
        model,
        gene_names,
        data_driven_genes
    )

    known_gene_inf_df = gene_difference_inference(
        model,
        gene_names,
        selected_known_genes
    )

    # ------------------------------------
    # Save model summaries
    # ------------------------------------

    save_csv(alpha_df, os.path.join(dirs["model"], "alpha_hat.csv"))
    save_csv(mu_df, os.path.join(dirs["model"], "mu_hat.csv"))

    pi_table = make_pi_table(pi_hat)
    s_table = make_precision_table(s_hat)

    save_csv(pi_table, os.path.join(dirs["model"], "pi_hat.csv"))
    save_csv(s_table, os.path.join(dirs["model"], "s_hat.csv"))

    # ------------------------------------
    # Save tables
    # ------------------------------------

    data_summary_df = make_data_summary_table(summary)

    save_table(
        data_summary_df,
        os.path.join(dirs["tables"], "data_summary.csv"),
        os.path.join(dirs["latex"], "data_summary.tex")
    )

    save_table(
        metrics_df,
        os.path.join(dirs["tables"], "clustering_metrics.csv"),
        os.path.join(dirs["latex"], "clustering_metrics.tex")
    )

    save_table(
        global_tests_df,
        os.path.join(dirs["tables"], "global_tests.csv"),
        os.path.join(dirs["latex"], "global_tests.tex")
    )

    save_table(
        pi_inf_df,
        os.path.join(dirs["tables"], "pi_inference.csv"),
        os.path.join(dirs["latex"], "pi_inference.tex")
    )

    save_table(
        s_inf_df,
        os.path.join(dirs["tables"], "precision_inference.csv"),
        os.path.join(dirs["latex"], "precision_inference.tex")
    )

    save_table(
        data_driven_diff_df,
        os.path.join(dirs["tables"], "top_genes_data_driven.csv"),
        os.path.join(dirs["latex"], "top_genes_data_driven.tex")
    )

    save_table(
        known_gene_diff_df,
        os.path.join(dirs["tables"], "top_genes_known.csv"),
        os.path.join(dirs["latex"], "top_genes_known.tex")
    )

    save_table(
        data_driven_inf_df,
        os.path.join(dirs["tables"], "gene_inference_data_driven.csv"),
        os.path.join(dirs["latex"], "gene_inference_data_driven.tex")
    )

    save_table(
        known_gene_inf_df,
        os.path.join(dirs["tables"], "gene_inference_known.csv"),
        os.path.join(dirs["latex"], "gene_inference_known.tex")
    )

    # ------------------------------------
    # Save plots
    # ------------------------------------

    plot_tsne_side_by_side(
        X,
        y,
        pred_labels,
        save_path=os.path.join(dirs["figures"], "tsne_true_vs_pred.png"),
        random_state=RANDOM_STATE
    )

    plot_gene_difference_bar(
        data_driven_diff_df,
        save_path=os.path.join(dirs["figures"], "top_genes_difference.png"),
        title=f"{dataset_name.upper()}: Top Data-Driven Gene Differences"
    )

    if len(known_gene_diff_df) > 0:
        plot_known_gene_grouped(
            known_gene_diff_df,
            save_path=os.path.join(dirs["figures"], "known_genes_grouped.png"),
            title=f"{dataset_name.upper()}: Known Gene Mean Composition"
        )

    if len(data_driven_inf_df) > 0:
        plot_gene_difference_ci(
            data_driven_inf_df,
            save_path=os.path.join(dirs["figures"], "gene_difference_ci_data_driven.png"),
            title=f"{dataset_name.upper()}: Data-Driven Gene Differences with 95% CI"
        )

    if len(known_gene_inf_df) > 0:
        plot_gene_difference_ci(
            known_gene_inf_df,
            save_path=os.path.join(dirs["figures"], "gene_difference_ci_known.png"),
            title=f"{dataset_name.upper()}: Known Gene Differences with 95% CI"
        )

    # ------------------------------------
    # Return all results
    # ------------------------------------

    return {
        "data_summary": data_summary_df,
        "metrics": metrics_df,
        "global_tests": global_tests_df,
        "pi_inference": pi_inf_df,
        "precision_inference": s_inf_df,
        "top_genes_data_driven": data_driven_diff_df,
        "top_genes_known": known_gene_diff_df,
        "gene_inference_data_driven": data_driven_inf_df,
        "gene_inference_known": known_gene_inf_df,
        "output_dirs": dirs,
    }
