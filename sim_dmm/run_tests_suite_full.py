# ============================================================
# sim_dmm/run_tests_suite_full.py
#
# Multi-Level Size & Power Study for DMM (Soft EM)
#
# Tests included (default runner):
#   T1_identical_precision  : H0 s_1 = ... = s_K
#   T2_identical_pi         : H0 pi_k = 1/K
#   T3_uniform_alpha_comp0  : H0 alpha_{0,1}=...=alpha_{0,p} (free common value)
#   T4_fixed_alpha1_all1    : H0 alpha_{1,j}=1 for all j
#
# Rejection at 1%, 5%, 10%
# Parallelized via joblib
#
# Uses dedicated null models for valid LRT fits
# Includes LRT, Wald (delta), Score tests
# ============================================================

import os
import numpy as np
import pandas as pd
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
    DMM_Soft_IdenticalMean,              # kept (legacy)
    DMM_Soft_UniformAlphaComponent,      # NEW
    DMM_Soft_FixedUniformAlphaComponent, # NEW
)
from fmvmm.mixtures.DMM_Soft_MuEqualPair import DMM_Soft_MuEqualPair  # kept (legacy)


# ============================================================
# GENERAL MULTIVARIATE DELTA WALD
# ============================================================

def _delta_wald_vector(g_fun, theta, cov, df, eps=1e-6, ridge=1e-12):
    """
    Multivariate delta-Wald for g(theta)=0 with dim(g)=df.

    g_fun(theta) must return shape (df,).
    Uses forward differences and a small ridge for stability.

    Returns: p-value
    """
    theta = np.asarray(theta, dtype=float)
    cov = np.asarray(cov, dtype=float)

    g0 = np.asarray(g_fun(theta), dtype=float).reshape(-1)
    q = int(g0.size)
    if q != int(df):
        raise ValueError(f"g_fun returned dim {q}, but df={df}.")

    G = np.zeros((q, theta.size), dtype=float)
    for j in range(theta.size):
        th = theta.copy()
        th[j] += eps
        gj = np.asarray(g_fun(th), dtype=float).reshape(-1)
        G[:, j] = (gj - g0) / eps

    V = G @ cov @ G.T
    V = 0.5 * (V + V.T)  # symmetrize
    V = V + ridge * np.eye(q)

    try:
        stat = float(g0.T @ np.linalg.solve(V, g0))
    except np.linalg.LinAlgError:
        stat = float(g0.T @ (np.linalg.pinv(V) @ g0))

    return float(chi2.sf(stat, df))


# ============================================================
# DGP: NULL
# ============================================================

def make_dgp_under_null(pi_base, alpha_base, which_test):
    """
    Return (pi, alpha) under the null for the chosen test.
    """
    pi = np.asarray(pi_base, dtype=float).copy()
    alpha = np.asarray(alpha_base, dtype=float).copy()

    K, p = alpha.shape
    pi, alpha, _ = reorder_components(pi, alpha, sort_by="pi")

    if which_test == "T1_identical_precision":
        # enforce s_1=...=s_K while keeping each mu_k
        s_bar = float(np.mean(alpha.sum(axis=1)))
        mu = alpha / alpha.sum(axis=1, keepdims=True)
        alpha = mu * s_bar

    elif which_test == "T2_identical_pi":
        # enforce equal mixing weights
        pi[:] = 1.0 / float(K)

    elif which_test == "T3_uniform_alpha_comp0":
        # enforce alpha_{0,j} all equal (free c), keep sum fixed
        s0 = float(alpha[0].sum())
        alpha[0, :] = s0 / float(p)

    elif which_test == "T4_fixed_alpha1_all1":
        # enforce alpha_{1,j}=1 for all j (Dirichlet(1,...,1))
        alpha[1, :] = 1.0

    # ---- legacy nulls (kept, but not used in default runner) ----
    elif which_test == "T3_identical_mean":
        mu = alpha / alpha.sum(axis=1, keepdims=True)
        mu_common = mu.mean(axis=0)
        s = alpha.sum(axis=1)
        alpha = s[:, None] * mu_common[None, :]

    elif which_test == "T4_mu_pair_01":
        mu = alpha / alpha.sum(axis=1, keepdims=True)
        mu_common = 0.5 * (mu[0] + mu[1])
        s = alpha.sum(axis=1)
        alpha[0] = s[0] * mu_common
        alpha[1] = s[1] * mu_common

    else:
        raise ValueError(f"Unknown which_test='{which_test}'")

    alpha = np.clip(alpha, 1e-12, None)
    pi = np.clip(pi, 1e-12, None)
    pi /= pi.sum()

    return pi, alpha


# ============================================================
# DGP: POWER
# ============================================================

def make_dgp_under_power(pi_base, alpha_base, which_test, effect_size):
    """
    Return (pi, alpha) under the alternative for the chosen test.
    """
    pi = np.asarray(pi_base, dtype=float).copy()
    alpha = np.asarray(alpha_base, dtype=float).copy()

    K, p = alpha.shape
    pi, alpha, _ = reorder_components(pi, alpha, sort_by="pi")

    es = float(effect_size)

    if which_test == "T1_identical_precision":
        # break identical precision: change only component 0 precision, keep its mu_0
        alpha[0, :] *= (1.0 + es)

    elif which_test == "T2_identical_pi":
        # break equal pi (keep interior)
        if K < 2:
            raise ValueError("T2 requires K>=2.")
        pi[0] += es
        pi[1] -= es
        pi = np.clip(pi, 1e-6, None)
        pi /= pi.sum()

    elif which_test == "T3_uniform_alpha_comp0":
        # break within-component equality for component 0 but keep sum fixed
        s0 = float(alpha[0].sum())
        base = np.full(p, s0 / float(p), dtype=float)
        base[0] *= np.exp(+es)
        base[1] *= np.exp(-es)
        base = np.clip(base, 1e-12, None)
        base *= s0 / float(base.sum())
        alpha[0, :] = base

    elif which_test == "T4_fixed_alpha1_all1":
        # break alpha_{1,j}=1: perturb two coords of component 1 around 1
        if p < 2:
            raise ValueError("T4 requires p>=2.")
        a1 = np.ones(p, dtype=float)
        a1[0] = np.exp(+es)
        a1[1] = np.exp(-es)
        a1 = np.clip(a1, 1e-12, None)
        alpha[1, :] = a1

    # ---- legacy alternatives (kept) ----
    elif which_test in ["T3_identical_mean", "T4_mu_pair_01"]:
        alpha[1, 0] += es
        alpha[1] = np.clip(alpha[1], 1e-6, None)

    else:
        raise ValueError(f"Unknown which_test='{which_test}'")

    alpha = np.clip(alpha, 1e-12, None)
    pi = np.clip(pi, 1e-12, None)
    pi /= pi.sum()

    return pi, alpha


# ============================================================
# Single replicate
# ============================================================

def _single_test_replicate(r, sc, K, p, which_test, pi_gen, alpha_gen, seed_offset, levels):
    rs = int(sc.seed + seed_offset + r)
    X, _ = simulate_dmm(sc.N, pi_gen, alpha_gen, random_state=rs)

    # FULL
    full = fit_soft_dmm(X, K=K, canonical_sort="pi", verbose=False)
    ll_full = float(full.log_likelihood_new)

    I_full, _ = full.get_info_mat(method="louis")
    cov_full = inf.cov_from_info(I_full)
    theta = inf.theta_tilde_from_model(full)

    tests = {}

    # -----------------------------
    # T1: Identical Precision
    # -----------------------------
    if which_test == "T1_identical_precision":
        null = DMM_Soft_IdenticalPrecision(K, verbose=False)
        null.fit(X)
        df = K - 1

        tests["T1_LRT"] = float(chi2.sf(2.0 * (ll_full - float(null.log_likelihood_new)), df))

        # Wald: s_k - s_1 = 0 for k=2..K
        def g_T1(th):
            alpha_vec = th[(K - 1):]
            alpha_mat = alpha_vec.reshape(K, p)
            s = alpha_mat.sum(axis=1)
            return (s[1:] - s[0]).reshape(-1)

        tests["T1_Wald"] = _delta_wald_vector(g_T1, theta, cov_full, df)

        # Score: alpha indices for constrained directions (use component blocks 1..K-1 convention)
        idx = []
        for j in range(1, K):
            idx.extend(inf.build_test_indices_alpha(null, j))
        tests["T1_Score"] = float(inf.score_test_fixed(null, idx).pvalue)

    # -----------------------------
    # T2: Identical Pi
    # -----------------------------
    elif which_test == "T2_identical_pi":
        null = DMM_Soft_IdenticalPi(K, verbose=False)
        null.fit(X)
        df = K - 1

        tests["T2_LRT"] = float(chi2.sf(2.0 * (ll_full - float(null.log_likelihood_new)), df))

        # Wald: pi_k - pi_1 = 0 for k=2..K
        def g_T2(th):
            eta = th[:K - 1]
            pi_hat = alr_inverse(eta)
            return (pi_hat[1:] - pi_hat[0]).reshape(-1)

        tests["T2_Wald"] = _delta_wald_vector(g_T2, theta, cov_full, df)

        # Score: eta block
        idx = inf.build_test_indices_eta(null)
        tests["T2_Score"] = float(inf.score_test_fixed(null, idx).pvalue)

    # -----------------------------
    # T3: Uniform alpha inside component 0 (free common value)
    # -----------------------------
    elif which_test == "T3_uniform_alpha_comp0":
        k0 = 0
        null = DMM_Soft_UniformAlphaComponent(K, component_index=k0, verbose=False)
        null.fit(X)
        df = p - 1

        tests["T3_LRT"] = float(chi2.sf(2.0 * (ll_full - float(null.log_likelihood_new)), df))

        # Wald: alpha_{k0,j} - alpha_{k0,p} = 0 for j=1..p-1
        def g_T3(th):
            alpha_vec = th[(K - 1):]
            alpha_mat = alpha_vec.reshape(K, p)
            row = alpha_mat[k0, :]
            return (row[:-1] - row[-1]).reshape(-1)

        tests["T3_Wald"] = _delta_wald_vector(g_T3, theta, cov_full, df)

        # Score: test alpha block for component k0 (constrained row)
        idx = inf.build_test_indices_alpha(null, k0)
        tests["T3_Score"] = float(inf.score_test_fixed(null, idx).pvalue)

    # -----------------------------
    # T4: Fixed alpha in component 1 equals all ones
    # -----------------------------
    elif which_test == "T4_fixed_alpha1_all1":
        k1 = 1
        null = DMM_Soft_FixedUniformAlphaComponent(K, component_index=k1, verbose=False)
        null.fit(X)
        df = p  # p constraints: alpha_{k1,j} - 1 = 0

        tests["T4_LRT"] = float(chi2.sf(2.0 * (ll_full - float(null.log_likelihood_new)), df))

        # Wald: alpha_{k1,j} - 1 = 0 for all j
        def g_T4(th):
            alpha_vec = th[(K - 1):]
            alpha_mat = alpha_vec.reshape(K, p)
            return (alpha_mat[k1, :] - 1.0).reshape(-1)

        tests["T4_Wald"] = _delta_wald_vector(g_T4, theta, cov_full, df)

        # Score: alpha block for component k1
        idx = inf.build_test_indices_alpha(null, k1)
        tests["T4_Score"] = float(inf.score_test_fixed(null, idx).pvalue)

    # -----------------------------
    # Legacy options (kept)
    # -----------------------------
    elif which_test == "T3_identical_mean":
        null = DMM_Soft_IdenticalMean(K, verbose=False)
        null.fit(X)
        df = (K - 1) * (p - 1)

        tests["T3_LRT"] = float(chi2.sf(2.0 * (ll_full - float(null.log_likelihood_new)), df))

        def g_legacy_T3(th):
            alpha_vec = th[(K - 1):]
            alpha_mat = alpha_vec.reshape(K, p)
            mu = alpha_mat / alpha_mat.sum(axis=1, keepdims=True)
            diffs = []
            for k in range(1, K):
                diffs.append(mu[k, :-1] - mu[0, :-1])
            return np.concatenate(diffs)

        tests["T3_Wald"] = _delta_wald_vector(g_legacy_T3, theta, cov_full, df)

        idx = []
        for j in range(1, K):
            idx.extend(inf.build_test_indices_alpha(null, j))
        tests["T3_Score"] = float(inf.score_test_fixed(null, idx).pvalue)

    elif which_test == "T4_mu_pair_01":
        null = DMM_Soft_MuEqualPair(K, mu_equal_pair=(0, 1), verbose=False)
        null.fit(X)
        df = p - 1

        tests["T4_LRT"] = float(chi2.sf(2.0 * (ll_full - float(null.log_likelihood_new)), df))

        def g_legacy_T4(th):
            alpha_vec = th[(K - 1):]
            alpha_mat = alpha_vec.reshape(K, p)
            mu = alpha_mat / alpha_mat.sum(axis=1, keepdims=True)
            return (mu[1, :-1] - mu[0, :-1]).reshape(-1)

        tests["T4_Wald"] = _delta_wald_vector(g_legacy_T4, theta, cov_full, df)

        idx = inf.build_test_indices_alpha(null, 1)
        tests["T4_Score"] = float(inf.score_test_fixed(null, idx).pvalue)

    else:
        raise ValueError(f"Unknown which_test='{which_test}'")

    # Store
    records = []
    for name, pval in tests.items():
        for level in levels:
            records.append({
                "Rep": int(r),
                "Test": str(name),
                "Level": float(level),
                "Reject": int(float(pval) < float(level)),
            })
    return records


# ============================================================
# Runner
# ============================================================

def run_test_suite(
    R=300,
    mode="size",
    which_test="T1_identical_precision",
    effect_size=0.3,
    seed_offset=90000,
    n_jobs=-1,
):

    os.makedirs("out/tests", exist_ok=True)

    sc = make_scenarios(N=3000)[0]
    K, p = int(sc.K), int(sc.p)

    pi_base, alpha_base, _ = reorder_components(sc.pi, sc.alpha, sort_by="pi")

    if mode == "size":
        pi_gen, alpha_gen = make_dgp_under_null(pi_base, alpha_base, which_test)
        tag = f"SIZE_{which_test}"
    elif mode == "power":
        pi_gen, alpha_gen = make_dgp_under_power(pi_base, alpha_base, which_test, effect_size)
        tag = f"POWER_{which_test}_eff_{effect_size}"
    else:
        raise ValueError("mode must be 'size' or 'power'")

    levels = [0.01, 0.05, 0.10]

    all_results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_single_test_replicate)(
            r, sc, K, p, which_test, pi_gen, alpha_gen, seed_offset, levels
        )
        for r in range(int(R))
    )

    records = [item for sublist in all_results for item in sublist]
    df = pd.DataFrame(records)

    summary = df.groupby(["Test", "Level"]).agg(
        RejectionRate=("Reject", "mean"),
        MC_SE=("Reject", lambda x: np.sqrt(np.mean(x) * (1.0 - np.mean(x)) / len(x))),
    ).reset_index()

    summary["Mode"] = str(mode)
    summary["Tag"] = str(tag)
    summary["R"] = int(R)

    fname = f"out/tests/test_suite_{tag}_R_{int(R)}.csv"
    summary.to_csv(fname, index=False)

    print("\nSaved:", fname)
    print(summary)

    return summary


if __name__ == "__main__":

    # Default test battery (recommended)
    tests_to_run = [
        "T1_identical_precision",
        "T2_identical_pi",
        "T3_uniform_alpha_comp0",
        "T4_fixed_alpha1_all1",
    ]

    for which in tests_to_run:
        print(f"\n===== SIZE: {which} =====")
        run_test_suite(R=300, mode="size", which_test=which, n_jobs=-1)

        print(f"\n===== POWER: {which} =====")
        run_test_suite(R=300, mode="power", which_test=which, effect_size=0.3, n_jobs=-1)
