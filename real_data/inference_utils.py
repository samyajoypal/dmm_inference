# ============================================================
# real_data/inference_utils.py
#
# Inference helpers for real data DMM analysis
# ============================================================

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from statsmodels.stats.multitest import multipletests

from fmvmm.inference import inference_dmm as inf
from fmvmm.utils.utils_dmm import alr_inverse, jacobian_pi_wrt_eta
from fmvmm.mixtures.DMM_Soft_NullModels import (
    DMM_Soft_IdenticalPrecision,
    DMM_Soft_IdenticalPi,
)


# ------------------------------------------------------------
# Basic covariance helper
# ------------------------------------------------------------

def get_theta_and_cov(model, info_method="louis"):
    """
    Return unconstrained parameter vector theta_tilde and covariance matrix.
    """
    I, _ = model.get_info_mat(method=info_method)
    cov = inf.cov_from_info(I)
    theta = inf.theta_tilde_from_model(model)
    return theta, cov


# ------------------------------------------------------------
# Generic multivariate delta method
# ------------------------------------------------------------

def delta_method(g_fun, theta, cov, eps=1e-6):
    """
    Compute g(theta), Jacobian G, and covariance of g(theta).

    Parameters
    ----------
    g_fun : callable
        Returns scalar or vector function of theta.
    theta : ndarray
    cov : ndarray
    eps : float

    Returns
    -------
    g0 : ndarray shape (q,)
    G  : ndarray shape (q, d)
    V  : ndarray shape (q, q)
    """
    theta = np.asarray(theta, dtype=float)
    cov = np.asarray(cov, dtype=float)

    g0 = np.asarray(g_fun(theta), dtype=float).reshape(-1)
    q = g0.size
    d = theta.size

    G = np.zeros((q, d), dtype=float)

    for j in range(d):
        th = theta.copy()
        th[j] += eps
        gj = np.asarray(g_fun(th), dtype=float).reshape(-1)
        G[:, j] = (gj - g0) / eps

    V = G @ cov @ G.T
    V = 0.5 * (V + V.T)

    return g0, G, V


# ------------------------------------------------------------
# Wald helpers
# ------------------------------------------------------------

def wald_test_delta(g_fun, theta, cov, df, eps=1e-6, ridge=1e-12):
    """
    Multivariate delta-Wald test of H0: g(theta)=0.
    Returns statistic and p-value.
    """
    g0, _, V = delta_method(g_fun, theta, cov, eps=eps)
    q = g0.size

    if q != int(df):
        raise ValueError(f"g_fun dimension {q} does not match df={df}")

    V = V + ridge * np.eye(q)

    try:
        stat = float(g0.T @ np.linalg.solve(V, g0))
    except np.linalg.LinAlgError:
        stat = float(g0.T @ (np.linalg.pinv(V) @ g0))

    pvalue = float(chi2.sf(stat, df))

    return {
        "stat": stat,
        "df": int(df),
        "pvalue": pvalue,
    }


def ci_from_est_se(est, se, alpha=0.05):
    """
    Wald-type confidence interval.
    """
    z = norm.ppf(1 - alpha / 2.0)
    return est - z * se, est + z * se


# ------------------------------------------------------------
# Global tests
# ------------------------------------------------------------

def test_equal_pi_wald(model, info_method="louis"):
    """
    Wald test for H0: pi_1 = pi_2, assuming K=2 or using first two clusters.
    """
    theta, cov = get_theta_and_cov(model, info_method=info_method)
    K = len(model.alpha_new)

    def g_fun(th):
        eta = th[:K - 1]
        pi_hat = alr_inverse(eta)
        return np.array([pi_hat[0] - pi_hat[1]])

    res = wald_test_delta(g_fun, theta, cov, df=1)

    return pd.DataFrame([{
        "Test": "pi1_eq_pi2",
        "Method": "Wald",
        "Statistic": res["stat"],
        "DF": res["df"],
        "PValue": res["pvalue"],
    }])


def test_equal_pi_score(X, K=2):
    """
    Score test for H0: pi_1 = ... = pi_K = 1/K using dedicated null model.
    """
    null = DMM_Soft_IdenticalPi(K, verbose=False)
    null.fit(X)

    eta_idx = inf.build_test_indices_eta(null)
    res = inf.score_test_fixed(null, eta_idx)

    return pd.DataFrame([{
        "Test": "pi1_eq_pi2",
        "Method": "Score",
        "Statistic": res.stat,
        "DF": res.df,
        "PValue": res.pvalue,
    }])


def test_equal_precision_wald(model, info_method="louis"):
    """
    Wald test for H0: s_1 = s_2, where s_k = sum_j alpha_{k,j}.
    """
    theta, cov = get_theta_and_cov(model, info_method=info_method)
    K, p = np.asarray(model.alpha_new).shape

    def g_fun(th):
        alpha_vec = th[(K - 1):]
        alpha_mat = alpha_vec.reshape(K, p)
        s = alpha_mat.sum(axis=1)
        return np.array([s[0] - s[1]])

    res = wald_test_delta(g_fun, theta, cov, df=1)

    return pd.DataFrame([{
        "Test": "s1_eq_s2",
        "Method": "Wald",
        "Statistic": res["stat"],
        "DF": res["df"],
        "PValue": res["pvalue"],
    }])


def test_equal_precision_score(X, K=2):
    """
    Score test for H0: identical precision using dedicated null model.
    """
    null = DMM_Soft_IdenticalPrecision(K, verbose=False)
    null.fit(X)

    idx = []
    for j in range(1, K):
        idx.extend(inf.build_test_indices_alpha(null, j))

    res = inf.score_test_fixed(null, idx)

    return pd.DataFrame([{
        "Test": "s1_eq_s2",
        "Method": "Score",
        "Statistic": res.stat,
        "DF": res.df,
        "PValue": res.pvalue,
    }])


def run_global_tests(model, X, info_method="louis"):
    """
    Run the two recommended global tests for real data:
      - pi_1 = pi_2
      - s_1 = s_2

    Returns a single DataFrame.
    """
    K = len(model.alpha_new)

    out = [
        test_equal_pi_wald(model, info_method=info_method),
        test_equal_pi_score(X, K=K),
        test_equal_precision_wald(model, info_method=info_method),
        test_equal_precision_score(X, K=K),
    ]

    df = pd.concat(out, ignore_index=True)

    return df


# ------------------------------------------------------------
# Gene-level inference: mu_{1j}, mu_{2j}, difference
# ------------------------------------------------------------

def gene_difference_inference(model, gene_names, selected_genes, alpha=0.05, info_method="louis"):
    """
    For each selected gene j, compute:
      - mu_1j
      - mu_2j
      - difference mu_1j - mu_2j
      - SEs via delta method
      - CIs
      - Wald test for H0: mu_1j = mu_2j

    Notes
    -----
    This routine currently assumes K=2.
    """
    theta, cov = get_theta_and_cov(model, info_method=info_method)

    alpha_hat = np.asarray(model.alpha_new, dtype=float)
    K, p = alpha_hat.shape

    if K != 2:
        raise ValueError("gene_difference_inference currently assumes K=2.")

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    records = []

    for gene in selected_genes:
        if gene not in gene_to_idx:
            continue

        j = gene_to_idx[gene]

        def g_fun(th):
            alpha_vec = th[(K - 1):]
            alpha_mat = alpha_vec.reshape(K, p)
            mu = alpha_mat / alpha_mat.sum(axis=1, keepdims=True)

            return np.array([
                mu[0, j],              # mu_1j
                mu[1, j],              # mu_2j
                mu[0, j] - mu[1, j],   # diff
            ])

        g0, _, V = delta_method(g_fun, theta, cov, eps=1e-6)

        se1 = float(np.sqrt(max(V[0, 0], 0.0)))
        se2 = float(np.sqrt(max(V[1, 1], 0.0)))
        sed = float(np.sqrt(max(V[2, 2], 0.0)))

        ci1_l, ci1_u = ci_from_est_se(g0[0], se1, alpha=alpha)
        ci2_l, ci2_u = ci_from_est_se(g0[1], se2, alpha=alpha)
        cid_l, cid_u = ci_from_est_se(g0[2], sed, alpha=alpha)

        wald_stat = float((g0[2] ** 2) / max(V[2, 2], 1e-12))
        wald_p = float(chi2.sf(wald_stat, 1))

        records.append({
            "Gene": gene,
            "mu_cluster1": float(g0[0]),
            "mu_cluster2": float(g0[1]),
            "difference": float(g0[2]),
            "SE_mu_cluster1": se1,
            "SE_mu_cluster2": se2,
            "SE_difference": sed,
            "CI_mu_cluster1_lower": float(ci1_l),
            "CI_mu_cluster1_upper": float(ci1_u),
            "CI_mu_cluster2_lower": float(ci2_l),
            "CI_mu_cluster2_upper": float(ci2_u),
            "CI_difference_lower": float(cid_l),
            "CI_difference_upper": float(cid_u),
            "WaldStatistic": wald_stat,
            "WaldPValue": wald_p,
        })

    df = pd.DataFrame(records)

    if len(df) > 0:
        reject, p_adj, _, _ = multipletests(df["WaldPValue"].values, method="fdr_bh")
        df["WaldPValueAdj"] = p_adj
        df["RejectFDR"] = reject.astype(int)

    return df


# ------------------------------------------------------------
# Optional summaries for pi and s
# ------------------------------------------------------------

def summarize_pi_inference(model, alpha=0.05, info_method="louis"):
    """
    Estimate, SE, and CI for pi_1 and pi_2 via delta method.
    """
    theta, cov = get_theta_and_cov(model, info_method=info_method)
    K = len(model.alpha_new)

    def g_fun(th):
        eta = th[:K - 1]
        pi_hat = alr_inverse(eta)
        return pi_hat[:2]

    g0, _, V = delta_method(g_fun, theta, cov, eps=1e-6)

    records = []

    for k in range(2):
        se = float(np.sqrt(max(V[k, k], 0.0)))
        lo, hi = ci_from_est_se(g0[k], se, alpha=alpha)

        records.append({
            "Parameter": f"pi_{k+1}",
            "Estimate": float(g0[k]),
            "SE": se,
            "CI_Lower": float(lo),
            "CI_Upper": float(hi),
        })

    return pd.DataFrame(records)


def summarize_precision_inference(model, alpha=0.05, info_method="louis"):
    """
    Estimate, SE, and CI for s_1 and s_2 via delta method.
    """
    theta, cov = get_theta_and_cov(model, info_method=info_method)
    K, p = np.asarray(model.alpha_new).shape

    def g_fun(th):
        alpha_vec = th[(K - 1):]
        alpha_mat = alpha_vec.reshape(K, p)
        s = alpha_mat.sum(axis=1)
        return s[:2]

    g0, _, V = delta_method(g_fun, theta, cov, eps=1e-6)

    records = []

    for k in range(2):
        se = float(np.sqrt(max(V[k, k], 0.0)))
        lo, hi = ci_from_est_se(g0[k], se, alpha=alpha)

        records.append({
            "Parameter": f"s_{k+1}",
            "Estimate": float(g0[k]),
            "SE": se,
            "CI_Lower": float(lo),
            "CI_Upper": float(hi),
        })

    return pd.DataFrame(records)
