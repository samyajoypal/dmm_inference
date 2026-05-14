# ============================================================
# sim_dmm/run_power_curves.py
#
# Power curves for DMM (Soft EM)
#
# Tests included:
#   T1_identical_precision
#   T2_identical_pi
#   T3_uniform_alpha_comp0
#   T4_fixed_alpha1_all1
#
# For each test:
#   - LRT
#   - Wald (multivariate delta)
#   - Score
#
# Parallelized via joblib
# Saves CSV + PNG curves
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2
from joblib import Parallel, delayed

from scenarios import make_scenarios
from simulate import simulate_dmm
from fit import fit_soft_dmm, reorder_components

from fmvmm.inference import inference_dmm as inf
from fmvmm.utils.utils_dmm import alr_inverse

from fmvmm.mixtures.DMM_Soft_NullModels import (
    DMM_Soft_IdenticalPrecision,
    DMM_Soft_IdenticalPi,
    DMM_Soft_UniformAlphaComponent,
    DMM_Soft_FixedUniformAlphaComponent,
)


# ============================================================
# Multivariate delta-Wald
# ============================================================

def _delta_wald_vector(g_fun, theta, cov, df, eps=1e-6, ridge=1e-12):

    theta = np.asarray(theta, dtype=float)
    cov = np.asarray(cov, dtype=float)

    g0 = np.asarray(g_fun(theta)).reshape(-1)
    q = g0.size

    G = np.zeros((q, theta.size))
    for j in range(theta.size):
        th = theta.copy()
        th[j] += eps
        G[:, j] = (np.asarray(g_fun(th)).reshape(-1) - g0) / eps

    V = G @ cov @ G.T
    V = 0.5 * (V + V.T)
    V += ridge * np.eye(q)

    try:
        stat = float(g0.T @ np.linalg.solve(V, g0))
    except np.linalg.LinAlgError:
        stat = float(g0.T @ (np.linalg.pinv(V) @ g0))

    return float(chi2.sf(stat, df))


# ============================================================
# Power DGP (mirrors run_tests_suite_full.py)
# ============================================================

def make_dgp_under_power(pi_base, alpha_base, which_test, effect_size):

    pi = np.asarray(pi_base).copy()
    alpha = np.asarray(alpha_base).copy()

    K, p = alpha.shape
    pi, alpha, _ = reorder_components(pi, alpha, sort_by="pi")

    es = float(effect_size)

    if which_test == "T1_identical_precision":
        alpha[0, :] *= (1.0 + es)

    elif which_test == "T2_identical_pi":
        pi[0] += es
        pi[1] -= es
        pi = np.clip(pi, 1e-6, None)
        pi /= pi.sum()

    elif which_test == "T3_uniform_alpha_comp0":
        s0 = alpha[0].sum()
        base = np.full(p, s0 / p)
        base[0] *= np.exp(+es)
        base[1] *= np.exp(-es)
        base *= s0 / base.sum()
        alpha[0, :] = base

    elif which_test == "T4_fixed_alpha1_all1":
        a1 = np.ones(p)
        a1[0] = np.exp(+es)
        a1[1] = np.exp(-es)
        alpha[1, :] = a1

    else:
        raise ValueError("Unknown test.")

    pi = np.clip(pi, 1e-12, None)
    pi /= pi.sum()
    alpha = np.clip(alpha, 1e-12, None)

    return pi, alpha


# ============================================================
# Single replicate
# ============================================================

def _single_power_replicate(
    r, sc, K, p, which_test,
    pi_gen, alpha_gen,
    seed_offset, levels
):

    rs = sc.seed + seed_offset + r
    X, _ = simulate_dmm(sc.N, pi_gen, alpha_gen, random_state=rs)

    # FULL
    full = fit_soft_dmm(X, K=K, canonical_sort="pi", verbose=False)
    ll_full = float(full.log_likelihood_new)

    I_full, _ = full.get_info_mat(method="louis")
    cov_full = inf.cov_from_info(I_full)
    theta = inf.theta_tilde_from_model(full)

    # Select null model and constraint
    if which_test == "T1_identical_precision":
        null = DMM_Soft_IdenticalPrecision(K, verbose=False)
        null.fit(X)
        df = K - 1

        def g_fun(th):
            alpha_vec = th[(K - 1):]
            alpha_mat = alpha_vec.reshape(K, p)
            s = alpha_mat.sum(axis=1)
            return s[1:] - s[0]

        idx = []
        for j in range(1, K):
            idx.extend(inf.build_test_indices_alpha(null, j))

    elif which_test == "T2_identical_pi":
        null = DMM_Soft_IdenticalPi(K, verbose=False)
        null.fit(X)
        df = K - 1

        def g_fun(th):
            eta = th[:K - 1]
            pi_hat = alr_inverse(eta)
            return pi_hat[1:] - pi_hat[0]

        idx = inf.build_test_indices_eta(null)

    elif which_test == "T3_uniform_alpha_comp0":
        null = DMM_Soft_UniformAlphaComponent(K, component_index=0, verbose=False)
        null.fit(X)
        df = p - 1

        def g_fun(th):
            alpha_vec = th[(K - 1):]
            alpha_mat = alpha_vec.reshape(K, p)
            row = alpha_mat[0]
            return row[:-1] - row[-1]

        idx = inf.build_test_indices_alpha(null, 0)

    elif which_test == "T4_fixed_alpha1_all1":
        null = DMM_Soft_FixedUniformAlphaComponent(K, component_index=1, verbose=False)
        null.fit(X)
        df = p

        def g_fun(th):
            alpha_vec = th[(K - 1):]
            alpha_mat = alpha_vec.reshape(K, p)
            return alpha_mat[1, :] - 1.0

        idx = inf.build_test_indices_alpha(null, 1)

    else:
        raise ValueError("Unknown test.")

    
    ll_null = float(null.log_likelihood_new)

    # LRT
    lrt_p = chi2.sf(2 * (ll_full - ll_null), df)

    # Wald
    wald_p = _delta_wald_vector(g_fun, theta, cov_full, df)

    # Score
    score_p = float(inf.score_test_fixed(null, idx).pvalue)

    out = []
    for level in levels:
        out.append({
            "Level": float(level),
            "WaldReject": int(wald_p < level),
            "LRTReject": int(lrt_p < level),
            "ScoreReject": int(score_p < level),
        })
    return out


# ============================================================
# Power curve runner
# ============================================================

def run_power_curve(
    which_test,
    effect_grid=(0, 0.25, 0.5, 0.75, 1.0, 1.25),
    R=200,
    n_jobs=-1
):

    os.makedirs("out/tests", exist_ok=True)
    os.makedirs("out/figs", exist_ok=True)

    sc = make_scenarios(N=3000)[0]
    K, p = sc.K, sc.p
    pi_base, alpha_base, _ = reorder_components(sc.pi, sc.alpha, sort_by="pi")

    levels = [0.01, 0.05, 0.10]
    all_results = []

    for j, delta in enumerate(effect_grid):
        print(f"[{which_test}] Δ = {delta}")

        pi_gen, alpha_gen = make_dgp_under_power(
            pi_base, alpha_base, which_test, delta
        )

        results = Parallel(n_jobs=n_jobs)(
            delayed(_single_power_replicate)(
                r, sc, K, p, which_test,
                pi_gen, alpha_gen,
                seed_offset=70000 + 100000*j,
                levels=levels
            )
            for r in range(R)
        )

        flat = [item for sublist in results for item in sublist]
        df = pd.DataFrame(flat)

        for level in levels:
            sub = df[df["Level"] == level]
            all_results.append({
                "Test": which_test,
                "EffectSize": delta,
                "Level": level,
                "WaldPower": sub["WaldReject"].mean(),
                "LRTPower": sub["LRTReject"].mean(),
                "ScorePower": sub["ScoreReject"].mean(),
            })

    df_out = pd.DataFrame(all_results)
    csv_name = f"out/tests/power_curve_{which_test}.csv"
    df_out.to_csv(csv_name, index=False)

    # Plot
    for level in sorted(df_out["Level"].unique()):
        sub = df_out[df_out["Level"] == level].sort_values("EffectSize")

        plt.figure(figsize=(6, 4))
        plt.plot(sub["EffectSize"], sub["WaldPower"], marker="o")
        plt.plot(sub["EffectSize"], sub["LRTPower"], marker="s")
        plt.plot(sub["EffectSize"], sub["ScorePower"], marker="^")
        plt.axhline(y=level, linestyle="--")

        plt.xlabel("Effect Size Δ")
        plt.ylabel("Power")
        plt.title(f"{which_test} (α={level})")
        plt.legend(["Wald", "LRT", "Score", "Nominal"])
        plt.tight_layout()

        plt.savefig(
            f"out/figs/power_curve_{which_test}_alpha_{level}.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    print("Saved:", csv_name)


# ============================================================

if __name__ == "__main__":

    tests = [
        "T1_identical_precision",
        "T2_identical_pi",
        "T3_uniform_alpha_comp0",
        "T4_fixed_alpha1_all1",
    ]

    for t in tests:
        run_power_curve(t, R=200, n_jobs=-1)
