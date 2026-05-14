# sim_dmm/pi_delta.py

import numpy as np
from fmvmm.utils.utils_dmm import jacobian_pi_wrt_eta, alr_transform

def pi_delta_inference(pi_hat, info_matrix):
    """
    Compute SE and Wald CIs for ALL K mixture weights
    using delta method from ALR parameterization.

    Parameters
    ----------
    pi_hat : (K,)
    info_matrix : ((K-1)+Kp , (K-1)+Kp)

    Returns
    -------
    se_pi : (K,)
    ci_pi : list of (lo,hi)
    """

    pi_hat = np.asarray(pi_hat, float)
    K = len(pi_hat)

    # Covariance on unconstrained scale
    try:
        cov = np.linalg.inv(info_matrix)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(info_matrix)

    # Extract eta covariance block
    cov_eta = cov[:K-1, :K-1]

    # Jacobian d pi / d eta
    J = jacobian_pi_wrt_eta(pi_hat)  # shape (K, K-1)

    cov_pi = J @ cov_eta @ J.T
    var_pi = np.clip(np.diag(cov_pi), 0.0, np.inf)
    se_pi = np.sqrt(var_pi)

    z = 1.959963984540054
    ci_pi = [(pi_hat[k] - z*se_pi[k],
              pi_hat[k] + z*se_pi[k]) for k in range(K)]

    return se_pi, ci_pi
