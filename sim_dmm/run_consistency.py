# ============================================================
# sim_dmm/run_consistency.py
# Parallelized via joblib
# FIXED:
# - True component alignment (no canonical mis-ordering)
# - Plot 3 pis + 9 alphas for Dataset 1
# - 4x3 layout
# - Save as PNG
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import permutations
from joblib import Parallel, delayed

from scenarios import make_scenarios
from simulate import simulate_dmm
from fit import fit_soft_dmm
from metrics import alpha_to_mu_tau


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def ensure_dirs():
    for d in ["out/raw", "out/clean", "out/figs"]:
        os.makedirs(d, exist_ok=True)


def rmse(true, est):
    est = np.asarray(est, float)
    return float(np.sqrt(np.mean((est - true) ** 2)))


# ------------------------------------------------------------
# TRUE alignment (robust)
# ------------------------------------------------------------

def align_to_true(pi_hat, alpha_hat, alpha_true):
    K = alpha_true.shape[0]
    best_perm = None
    best_score = np.inf

    for perm in permutations(range(K)):
        score = 0.0
        for k in range(K):
            score += np.linalg.norm(alpha_hat[perm[k]] - alpha_true[k])
        if score < best_score:
            best_score = score
            best_perm = perm

    alpha_hat = alpha_hat[list(best_perm)]
    pi_hat = pi_hat[list(best_perm)]

    return pi_hat, alpha_hat


# ------------------------------------------------------------
# Single replicate
# ------------------------------------------------------------

def _single_consistency_rep(
    N,
    r,
    sc,
    K,
    p,
    pi_true,
    alpha_true,
    max_iter,
    tol
):
    rs = sc.seed + 50000 + 1000 * N + r
    X, _ = simulate_dmm(N, pi_true, alpha_true, random_state=rs)

    model = fit_soft_dmm(
        X,
        K=K,
        max_iter=max_iter,
        tol=tol,
        verbose=False
    )

    pi_hat, alpha_hat = model.get_params()

    # 🔥 TRUE ALIGNMENT (correct)
    pi_hat, alpha_hat = align_to_true(
        pi_hat, alpha_hat, alpha_true
    )

    mu_hat, tau_hat = alpha_to_mu_tau(alpha_hat)

    return {
        "N": N,
        "rep": r,
        "pi_hat": pi_hat.tolist(),
        "alpha_hat": alpha_hat.tolist(),
        "mu_hat": mu_hat.tolist(),
        "tau_hat": tau_hat.tolist(),
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main(
    which_dataset=1,
    Ns=(500, 1000, 5000, 10000),
    R=100,
    max_iter=200,
    tol=1e-6,
    n_jobs=-1
):

    ensure_dirs()

    sc = make_scenarios(N=3000)[which_dataset - 1]
    K, p = sc.K, sc.p

    pi_true = sc.pi
    alpha_true = sc.alpha
    mu_true, tau_true = alpha_to_mu_tau(alpha_true)

    # --------------------------------------------------------
    # Parallel execution
    # --------------------------------------------------------

    tasks = [(N, r) for N in Ns for r in range(R)]

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_single_consistency_rep)(
            N=N,
            r=r,
            sc=sc,
            K=K,
            p=p,
            pi_true=pi_true,
            alpha_true=alpha_true,
            max_iter=max_iter,
            tol=tol
        )
        for (N, r) in tasks
    )

    df_raw = pd.DataFrame(results)
    df_raw.to_csv(
        f"out/raw/consistency_dataset{which_dataset}.csv",
        index=False
    )

    # --------------------------------------------------------
    # RMSE summaries
    # --------------------------------------------------------

    out_rows = []

    for N in Ns:
        sub = df_raw[df_raw["N"] == N]

        pi_est = np.stack(sub["pi_hat"].apply(np.array).to_numpy())
        alpha_est = np.stack(
            sub["alpha_hat"]
            .apply(lambda A: np.array(A).reshape(K, p))
            .to_numpy()
        )

        # ----- pi (ALL K) -----
        for k in range(K):
            out_rows.append({
                "N": N,
                "param": f"pi_{k+1}",
                "rmse": rmse(pi_true[k], pi_est[:, k])
            })

        # ----- alpha -----
        for k in range(K):
            for m in range(p):
                out_rows.append({
                    "N": N,
                    "param": f"alpha_{k+1}{m+1}",
                    "rmse": rmse(alpha_true[k, m], alpha_est[:, k, m])
                })

    df_rmse = pd.DataFrame(out_rows)
    df_rmse.to_csv(
        f"out/clean/consistency_rmse_dataset{which_dataset}.csv",
        index=False
    )

    # --------------------------------------------------------
    # 4x3 Plot for Scenario 1
    # Row 1: pi_1, pi_2, pi_3
    # Rows 2-4: alpha rows
    # --------------------------------------------------------

    if which_dataset == 1 and K == 3 and p == 3:

        fig, axes = plt.subplots(
            4, 3, figsize=(12, 10), constrained_layout=True
        )

        # ---- Row 1: pis ----
        for k in range(3):
            ax = axes[0, k]
            param = f"pi_{k+1}"
            tmp = df_rmse[df_rmse["param"] == param].sort_values("N")

            ax.plot(tmp["N"], tmp["rmse"], marker="o")
            ax.set_xscale("log")
            ax.set_xticks(Ns)
            ax.set_xticklabels([str(n) for n in Ns])
            ax.set_title(rf"$\pi_{{{k+1}}}$")
            ax.set_xlabel("N")
            ax.set_ylabel("RMSE")

        # ---- Rows 2-4: alphas ----
        for k in range(3):
            for m in range(3):
                ax = axes[k+1, m]
                param = f"alpha_{k+1}{m+1}"
                tmp = df_rmse[df_rmse["param"] == param].sort_values("N")

                ax.plot(tmp["N"], tmp["rmse"], marker="o")
                ax.set_xscale("log")
                ax.set_xticks(Ns)
                ax.set_xticklabels([str(n) for n in Ns])
                ax.set_title(rf"$\alpha_{{{k+1}{m+1}}}$")
                ax.set_xlabel("N")
                ax.set_ylabel("RMSE")

        fig.savefig(
            f"out/figs/consistency_full_rmse_dataset{which_dataset}.png",
            dpi=300
        )
        plt.close(fig)

    print("Wrote:")
    print(f"  out/raw/consistency_dataset{which_dataset}.csv")
    print(f"  out/clean/consistency_rmse_dataset{which_dataset}.csv")
    if which_dataset == 1:
        print(f"  out/figs/consistency_full_rmse_dataset{which_dataset}.png")


# ------------------------------------------------------------
if __name__ == "__main__":
    main(which_dataset=1, R=100, n_jobs=-1)
