import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class RLSConfig:
    """Configuration for Recursive Least Squares."""
    p_order: int = 5          # AR order
    q_order: int = 5          # MA order
    my: int = 1               # Number of outputs
    mu: int = 1               # Number of inputs
    forgetting_factor: float = 0.98
    initial_covariance: float = 1000.0

    @property
    def n_theta(self) -> int:
        return self.my**2 * self.p_order + self.my * self.mu * self.q_order


class RecursiveLeastSquares:
    """Online parameter estimator using RLS with forgetting factor."""

    def __init__(self, config: RLSConfig):
        self.cfg = config
        self.theta = 0.05 * np.random.randn(config.n_theta)
        self.P = config.initial_covariance * np.eye(config.n_theta)

        self.Y_buffer = np.zeros((config.p_order, config.my))
        self.U_buffer = np.zeros((config.q_order, config.mu))

    def _build_regressor(self) -> np.ndarray:
        phi = np.concatenate([
            self.Y_buffer[::-1].ravel(),
            self.U_buffer[::-1].ravel()
        ])
        return phi

    def update(self, y: float, u: float) -> float:
        phi = self._build_regressor()

        if np.linalg.norm(self.P) > 1e10 or np.any(np.isnan(self.P)):
            self.P = self.cfg.initial_covariance * np.eye(self.cfg.n_theta)

        denom = self.cfg.forgetting_factor + phi @ self.P @ phi
        L = (self.P @ phi) / denom
        self.P = (1.0 / self.cfg.forgetting_factor) * (self.P - np.outer(L, phi @ self.P))

        y_pred = phi @ self.theta
        error = y - y_pred
        self.theta = self.theta + L * error

        if np.any(np.isnan(self.theta)) or np.linalg.norm(self.theta) > 1e5:
            self.theta = 0.05 * np.random.randn(self.cfg.n_theta)

        self.Y_buffer = np.vstack([self.Y_buffer[1:], y])
        self.U_buffer = np.vstack([self.U_buffer[1:], u])

        return error

    def get_polynomials(self) -> Tuple[np.ndarray, np.ndarray]:
        A_poly = self.theta[:self.cfg.p_order]
        B_poly = self.theta[self.cfg.p_order:]
        return A_poly, B_poly
