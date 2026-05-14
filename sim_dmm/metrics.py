# sim_dmm/metrics.py

import numpy as np


# ------------------------------------------------------------
# Vectorization helpers
# ------------------------------------------------------------

def vec_alpha(alpha: np.ndarray) -> np.ndarray:
    # order: alpha_11,...,alpha_1p, alpha_21,...,alpha_Kp
    return np.asarray(alpha, float).reshape(-1)


def alpha_to_mu_tau(alpha: np.ndarray):
    alpha = np.asarray(alpha, float)
    tau = alpha.sum(axis=1)
    mu = alpha / tau[:, None]
    return mu, tau


# ------------------------------------------------------------
# Wald CI
# ------------------------------------------------------------

def wald_ci(theta_hat: np.ndarray,
            se_hat: np.ndarray,
            z: float = 1.959963984540054):
    lo = theta_hat - z * se_hat
    hi = theta_hat + z * se_hat
    return lo, hi


# ------------------------------------------------------------
# Simulation summary for one scalar parameter
# ------------------------------------------------------------

def summarize_scalar(true_val: float,
                     est: np.ndarray,
                     se: np.ndarray):

    est = np.asarray(est, float)
    se = np.asarray(se, float)

    # --- Mean estimate ---
    mean_est = float(np.mean(est))

    # --- Bias ---
    bias = float(mean_est - true_val)

    # --- RMSE ---
    rmse = float(np.sqrt(np.mean((est - true_val) ** 2)))

    # --- Empirical SE ---
    emp_se = float(np.std(est, ddof=1)) if len(est) > 1 else float("nan")

    # --- Mean model-based SE ---
    mean_se = float(np.mean(se))

    # --- Wald CI ---
    lo, hi = wald_ci(est, se)

    # --- Mean CI length ---
    mean_ci_len = float(np.mean(hi - lo))

    # --- Coverage ---
    cover = float(np.mean((lo <= true_val) & (true_val <= hi)))

    return {
        "true": true_val,
        "mean": mean_est,           # NEW COLUMN
        "bias": bias,
        "rmse": rmse,
        "mean_se": mean_se,
        "emp_se": emp_se,
        "mean_ci_len": mean_ci_len,
        "cover": cover,
    }
