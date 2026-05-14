# sim_dmm/fit.py

import numpy as np
from fmvmm.mixtures.DMM_Soft import DMM_Soft

_EPS = 1e-15

def alpha_to_mu_tau(alpha: np.ndarray):
    alpha = np.asarray(alpha, float)
    tau = alpha.sum(axis=1)
    mu = alpha / tau[:, None]
    return mu, tau

def reorder_components(pi: np.ndarray, alpha: np.ndarray, sort_by: str = "pi"):
    """
    sort_by: "pi" or "mu1"
    """
    pi = np.asarray(pi, float)
    alpha = np.asarray(alpha, float)
    mu, _ = alpha_to_mu_tau(alpha)

    if sort_by == "pi":
        key = -pi
    elif sort_by == "mu1":
        key = -mu[:, 0]
    else:
        raise ValueError("sort_by must be 'pi' or 'mu1'")

    perm = np.argsort(key)
    return pi[perm], alpha[perm], perm

def post_m_step_canonical(pi, alpha, sort_by="pi"):
    pi2, alpha2, _ = reorder_components(pi, alpha, sort_by=sort_by)
    # safeguard pi positivity
    pi2 = np.clip(pi2, _EPS, 1.0)
    pi2 = pi2 / pi2.sum()
    alpha2 = np.clip(alpha2, _EPS, np.inf)
    return pi2, alpha2

def fit_soft_dmm(X: np.ndarray, K: int, max_iter: int = 200, tol: float = 1e-6,
                 init: str = "kmeans", method: str = "meanprecision",
                 canonical_sort: str = "pi", verbose: bool = False):
    model = DMM_Soft(
        n_clusters=K,
        tol=tol,
        initialization=init,
        method=method,
        print_log_likelihood=False,
        max_iter=max_iter,
        verbose=verbose
    )
    model.fit(X, post_m_step=lambda pi, alpha: post_m_step_canonical(pi, alpha, canonical_sort))
    return model
