# sim_dmm/scenarios.py

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    name: str
    N: int
    K: int
    p: int
    pi: np.ndarray          # shape (K,)
    mu: np.ndarray          # shape (K,p), rows sum to 1
    tau: np.ndarray         # shape (K,)
    seed: int = 1

    @property
    def alpha(self) -> np.ndarray:
        return (self.tau[:, None] * self.mu).astype(float)


def _normalize_rows(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, float)
    return M / M.sum(axis=1, keepdims=True)


def make_scenarios(N: int = 3000):

    # =========================================================
    # Scenario 1: K=3, p=3
    # Strongly separated simplex vertices
    # Moderate tau
    # =========================================================
    pi1 = np.array([0.35, 0.40, 0.25])

    mu1 = np.array([
        [0.80, 0.10, 0.10],
        [0.10, 0.80, 0.10],
        [0.10, 0.15, 0.75],
    ])
    mu1 = _normalize_rows(mu1)

    tau1 = np.array([80.0, 90.0, 100.0])

    # =========================================================
    # Scenario 2: K=4, p=5
    # Each component dominant in one coordinate
    # High tau for strong curvature
    # =========================================================
    pi2 = np.array([0.30, 0.25, 0.25, 0.20])

    mu2 = np.array([
        [0.70, 0.10, 0.10, 0.05, 0.05],
        [0.10, 0.70, 0.10, 0.05, 0.05],
        [0.10, 0.10, 0.70, 0.05, 0.05],
        [0.05, 0.05, 0.10, 0.70, 0.10],
    ])
    mu2 = _normalize_rows(mu2)

    tau2 = np.array([120.0, 130.0, 125.0, 140.0])

    # =========================================================
    # Scenario 3: K=3, p=20
    # Block-separated high dimensions
    # No symmetry
    # =========================================================
    pi3 = np.array([0.30, 0.45, 0.25])

    mu3 = np.full((3, 20), 0.01)

    # Component 1 dominant in dims 0–4
    mu3[0, 0:5] += 0.12

    # Component 2 dominant in dims 7–11
    mu3[1, 7:12] += 0.12

    # Component 3 dominant in dims 14–18
    mu3[2, 14:19] += 0.12

    mu3 = _normalize_rows(mu3)

    tau3 = np.array([200.0, 220.0, 210.0])

    # =========================================================
    # Scenario 4: K=5, p=10
    # Each component dominant in two coordinates
    # Strong separation
    # High tau
    # =========================================================
    pi4 = np.array([0.20, 0.20, 0.20, 0.20, 0.20])

    mu4 = np.full((5, 10), 0.02)

    dominant_pairs = [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (8, 9),
    ]

    for k, (i, j) in enumerate(dominant_pairs):
        mu4[k, i] += 0.35
        mu4[k, j] += 0.20

    mu4 = _normalize_rows(mu4)

    tau4 = np.array([160.0, 170.0, 180.0, 175.0, 165.0])

    return [
        Scenario("S1_K3_p3_well_sep", N, 3, 3, pi1, mu1, tau1, seed=1),
        Scenario("S2_K4_p5_well_sep", N, 4, 5, pi2, mu2, tau2, seed=2),
        Scenario("S3_K3_p20_block_sep", N, 3, 20, pi3, mu3, tau3, seed=3),
        Scenario("S4_K5_p10_strong_sep", N, 5, 10, pi4, mu4, tau4, seed=4),
    ]
