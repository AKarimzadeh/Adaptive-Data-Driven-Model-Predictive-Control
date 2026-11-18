import numpy as np
import osqp
from scipy import sparse
from dataclasses import dataclass
from src.rls import RecursiveLeastSquares

@dataclass
class MPCConfig:
    """Model Predictive Control configuration."""
    Ny: int = 80               # Prediction horizon
    Nu: int = 70               # Control horizon
    u_min: float = -5.0
    u_max: float = 5.0
    du_max: float = 0.5
    Ts: float = 0.009
    regularization: float = 1e-4

class AdaptiveMPC:
    """Data-driven Adaptive MPC using OSQP."""

    def __init__(self, mpc_cfg: MPCConfig, rls: RecursiveLeastSquares):
        self.cfg = mpc_cfg
        self.rls = rls
        self.u_prev = 0.0

    def _compute_free_response_and_G(self, dist_est: float):
        A_poly, B_poly = self.rls.get_polynomials()
        p, q = self.rls.cfg.p_order, self.rls.cfg.q_order
        Ny, Nu = self.cfg.Ny, self.cfg.Nu

        g = np.zeros(Ny)
        for k in range(Ny):
            phi = []
            for i in range(1, p+1):
                idx = k - i
                if idx >= 0:
                    phi.append(g[idx])
                else:
                    phi.append(self.rls.Y_buffer[p + idx, 0])
            for i in range(1, q+1):
                idx = k - i
                if idx >= 0:
                    phi.append(0.0)
                else:
                    phi.append(self.rls.U_buffer[q + idx, 0])

            phi = np.asarray(phi)
            val = self.rls.theta @ phi + dist_est
            g[k] = 0.0 if np.isnan(val) else val

        h = np.zeros(Ny)
        for k in range(Ny):
            val = 0.0
            for i in range(1, p+1):
                if (k - i) >= 0:
                    val += A_poly[i-1] * h[k - i]
            for i in range(1, q+1):
                if k == (i - 1):
                    val += B_poly[i-1]
            h[k] = val

        G = np.zeros((Ny, Nu))
        for c in range(Nu):
            G[c:, c] = h[:Ny-c]

        return g, G

    def solve(self, y_current: float, reference: float, dist_est: float) -> float:
        g, G = self._compute_free_response_and_G(dist_est)

        ref_vec = reference * np.ones(self.cfg.Ny)
        H = 2.0 * (G.T @ G)
        H = 0.5 * (H + H.T) + self.cfg.regularization * np.eye(self.cfg.Nu)
        f = 2.0 * G.T @ (g - ref_vec)

        Nu = self.cfg.Nu
        A_amp = np.eye(Nu)
        l_amp = np.full(Nu, self.cfg.u_min)
        u_amp = np.full(Nu, self.cfg.u_max)

        A_rate = np.eye(Nu) - np.diag(np.ones(Nu-1), k=-1)
        rate = self.cfg.du_max * self.cfg.Ts
        l_rate = np.full(Nu, -rate)
        u_rate = np.full(Nu, rate)
        l_rate[0] += self.u_prev
        u_rate[0] += self.u_prev

        A_qp = np.vstack([A_amp, A_rate])
        l_qp = np.concatenate([l_amp, l_rate])
        u_qp = np.concatenate([u_amp, u_rate])

        u_opt = self.u_prev

        if not (np.any(np.isnan(H)) or np.any(np.isnan(f))):
            try:
                solver = osqp.OSQP()
                solver.setup(
                    sparse.csc_matrix(H), f,
                    sparse.csc_matrix(A_qp), l_qp, u_qp,
                    verbose=False, eps_abs=1e-4, eps_rel=1e-4, max_iter=4000
                )
                res = solver.solve()
                if res.info.status_val in (1, 2):
                    u_opt = float(res.x[0])
            except Exception:
                pass

        u_opt = float(np.clip(u_opt, self.cfg.u_min, self.cfg.u_max))
        self.u_prev = u_opt
        return u_opt
