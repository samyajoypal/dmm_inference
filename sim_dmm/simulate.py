# sim_dmm/simulate.py

import numpy as np
from fmvmm.distributions import dirichlet
from fmvmm.utils.utils_mixture import sample_mixture_distribution

def simulate_dmm(N: int, pi: np.ndarray, alpha: np.ndarray, random_state: int):
    """
    alpha: shape (K,p)
    Returns X (N,p), z (N,)
    """
    K, p = alpha.shape
    alphas_true = [[alpha[k].tolist()] for k in range(K)]  # matches your sampler signature

    X, z = sample_mixture_distribution(
        N,
        dirichlet.rvs,
        pi.tolist(),
        alphas_true,
        mixture_type="identical",
        random_state=random_state
    )
    return np.asarray(X, float), np.asarray(z, int)
