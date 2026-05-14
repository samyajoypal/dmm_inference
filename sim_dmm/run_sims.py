# ============================================================
# sim_dmm/run_sims.py
# TRUE-FINAL VERSION
# - Correct component alignment
# - Correct covariance permutation
# - Correct alpha / mu / tau SE
# ============================================================

import os
import numpy as np
import pandas as pd
from itertools import permutations
from joblib import Parallel, delayed

from scenarios import make_scenarios
from simulate import simulate_dmm
from fit import fit_soft_dmm
from metrics import summarize_scalar, alpha_to_mu_tau
from tables import df_to_latex_multirow

from fmvmm.inference import inference_dmm as inf
from pi_delta import pi_delta_inference


# ============================================================
# Utilities
# ============================================================

def ensure_dirs():
    for d in ["out/raw", "out/clean", "out/latex"]:
        os.makedirs(d, exist_ok=True)


def get_info_and_cov(model, info_method="louis"):
    I, _ = model.get_info_mat(method=info_method)
    cov = inf.cov_from_info(I)
    return I, cov


# ============================================================
# Alignment returning permutation
# ============================================================

def align_to_true(pi_hat, alpha_hat, alpha_true):
    """
    Align estimated components to true components.
    Returns perm such that:
        alpha_hat_aligned[k] matches alpha_true[k]
    """
    K = alpha_true.shape[0]
    best_perm = None
    best_score = np.inf

    for perm in permutations(range(K)):
        score = sum(
            np.linalg.norm(alpha_hat[perm[k]] - alpha_true[k])
            for k in range(K)
        )
        if score < best_score:
            best_score = score
            best_perm = perm

    pi_hat = pi_hat[list(best_perm)]
    alpha_hat = alpha_hat[list(best_perm)]

    return pi_hat, alpha_hat, best_perm


# ============================================================
# Single replicate
# ============================================================

def _single_replicate(
    r,
    sc,
    K,
    p,
    pi_true,
    alpha_true,
    info_method,
    seed_offset,
    max_iter,
    tol,
    verbose,
):

    rs = sc.seed + seed_offset + r
    X, _ = simulate_dmm(sc.N, pi_true, alpha_true, random_state=rs)

    model = fit_soft_dmm(
        X,
        K=K,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose
    )

    pi_hat_raw, alpha_hat_raw = model.get_params()

    # ---- ALIGN PARAMETERS ----
    pi_hat, alpha_hat, perm = align_to_true(
        pi_hat_raw, alpha_hat_raw, alpha_true
    )

    mu_hat, tau_hat = alpha_to_mu_tau(alpha_hat)

    # ---- INFORMATION MATRIX ----
    I_hat, cov_hat = get_info_and_cov(model, info_method)

    # ---- PERMUTE COVARIANCE BLOCKS ----
    # Structure: (eta | alpha_1 | alpha_2 | ... | alpha_K)
    offset = K - 1

    # Build permutation matrix for alpha blocks
    P = np.zeros((K * p, K * p))
    for new_k, old_k in enumerate(perm):
        P[new_k*p:(new_k+1)*p,
          old_k*p:(old_k+1)*p] = np.eye(p)

    # Extract and permute alpha covariance
    cov_alpha_raw = cov_hat[offset:, offset:]
    cov_alpha = P @ cov_alpha_raw @ P.T

    # ---- PI DELTA SE ----
    se_pi_full, _ = pi_delta_inference(pi_hat, I_hat)

    # ---- ALPHA SE ----
    se_alpha_r = np.sqrt(
        np.clip(np.diag(cov_alpha), 0.0, np.inf)
    )

    # ---- DELTA FOR MU AND TAU ----
    eps = 1e-6
    a0 = alpha_hat.reshape(-1)

    def derive_from_a(a_flat):
        a = a_flat.reshape(K, p)
        mu, tau = alpha_to_mu_tau(a)
        return np.concatenate([mu.reshape(-1), tau])

    y0 = derive_from_a(a0)
    J = np.zeros((len(y0), len(a0)))

    for j in range(len(a0)):
        a1 = a0.copy()
        a1[j] += eps
        y1 = derive_from_a(a1)
        J[:, j] = (y1 - y0) / eps

    cov_derived = J @ cov_alpha @ J.T
    se_derived = np.sqrt(np.clip(np.diag(cov_derived), 0.0, np.inf))

    se_mu_r = se_derived[:K*p]
    se_tau_r = se_derived[K*p:]

    return {
        "pi_hat": pi_hat,
        "alpha_hat": alpha_hat,
        "mu_hat": mu_hat,
        "tau_hat": tau_hat,
        "se_pi": se_pi_full,
        "se_alpha": se_alpha_r,
        "se_mu": se_mu_r,
        "se_tau": se_tau_r,
    }


# ============================================================
# Main Simulation
# ============================================================

def main(R=100,
         info_method="louis",
         seed_offset=10000,
         max_iter=200,
         tol=1e-6,
         verbose=False,
         n_jobs=-1):

    ensure_dirs()
    scenarios = make_scenarios(N=3000)

    paper_rows = []

    for s_idx, sc in enumerate(scenarios, start=1):

        print(f"\n=== Running {sc.name} (Dataset {s_idx}) ===")

        K, p = sc.K, sc.p
        pi_true = sc.pi
        alpha_true = sc.alpha
        mu_true, tau_true = alpha_to_mu_tau(alpha_true)

        results = Parallel(n_jobs=n_jobs)(
            delayed(_single_replicate)(
                r=r,
                sc=sc,
                K=K,
                p=p,
                pi_true=pi_true,
                alpha_true=alpha_true,
                info_method=info_method,
                seed_offset=seed_offset,
                max_iter=max_iter,
                tol=tol,
                verbose=verbose,
            )
            for r in range(R)
        )

        est_pi = np.asarray([res["pi_hat"] for res in results])
        est_alpha = np.asarray([res["alpha_hat"] for res in results])
        est_mu = np.asarray([res["mu_hat"] for res in results])
        est_tau = np.asarray([res["tau_hat"] for res in results])

        se_pi = np.asarray([res["se_pi"] for res in results])
        se_alpha = np.asarray([res["se_alpha"] for res in results])
        se_mu = np.asarray([res["se_mu"] for res in results])
        se_tau = np.asarray([res["se_tau"] for res in results])

        # ---- PI ----
        for k in range(K):
            out = summarize_scalar(pi_true[k], est_pi[:, k], se_pi[:, k])
            paper_rows.append({"Dataset": s_idx,
                               "Parameter": rf"$\pi_{{{k+1}}}$",
                               **out})

        # ---- ALPHA ----
        for k in range(K):
            for m in range(p):
                idx = k*p + m
                out = summarize_scalar(alpha_true[k,m],
                                       est_alpha[:,k,m],
                                       se_alpha[:,idx])
                paper_rows.append({"Dataset": s_idx,
                                   "Parameter": rf"$\alpha_{{{k+1}{m+1}}}$",
                                   **out})

        # ---- MU ----
        for k in range(K):
            for m in range(p):
                idx = k*p + m
                out = summarize_scalar(mu_true[k,m],
                                       est_mu[:,k,m],
                                       se_mu[:,idx])
                paper_rows.append({"Dataset": s_idx,
                                   "Parameter": rf"$\mu_{{{k+1},{m+1}}}$",
                                   **out})

        # ---- TAU ----
        for k in range(K):
            out = summarize_scalar(tau_true[k],
                                   est_tau[:,k],
                                   se_tau[:,k])
            paper_rows.append({"Dataset": s_idx,
                               "Parameter": rf"$\tau_{{{k+1}}}$",
                               **out})

    df_sum = pd.DataFrame(paper_rows)
    df_sum.to_csv("out/clean/paper_summary_all.csv", index=False)

    tex = df_to_latex_multirow(df_sum)

    with open("out/latex/table_main.tex", "w", encoding="utf-8") as f:
        f.write(tex)

    print("\nWrote:")
    print("  out/clean/paper_summary_all.csv")
    print("  out/latex/table_main.tex")


if __name__ == "__main__":
    main(R=100, verbose=False, n_jobs=-1)
