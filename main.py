# ---------------------------------------------------------
# Developed by Abdollah Karimzadeh - 2026
# Simulates the closed-loop response of an AVR system using 
# a data-driven Adaptive MPC based on online ARMAX identification.
# ---------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import time

from src.avr_system import AVRParameters, build_continuous_plant, discretize_plant
from src.rls import RLSConfig, RecursiveLeastSquares
from src.mpc import MPCConfig, AdaptiveMPC

def run_simulation():
    np.random.seed(42)
    params = AVRParameters()
    Ts = 0.009
    A_c, B_c, C_c, D_c = build_continuous_plant(params)
    A, B, C, D = discretize_plant(A_c, B_c, C_c, D_c, Ts)
    n_x = A.shape[0]

    Tsim = 200.0
    t_vec = np.arange(0, Tsim + Ts/2, Ts)
    N = len(t_vec)

    split_idx = int(0.6 * N)
    seg = split_idx // 6
    ref_step = np.concatenate([
        0.0 * np.ones(seg), 0.8 * np.ones(seg), 1.0 * np.ones(seg),
        1.2 * np.ones(seg), 0.9 * np.ones(seg), np.ones(split_idx - 5 * seg)
    ])
    
    t_sin_local = t_vec[split_idx:] - t_vec[split_idx]
    ref_sin = 1.0 + 0.1 * np.sin(2 * np.pi * 0.08 * t_sin_local)
    reference = np.concatenate([ref_step, ref_sin])

    rls_cfg = RLSConfig()
    mpc_cfg = MPCConfig(Ts=Ts)
    rls = RecursiveLeastSquares(rls_cfg)
    controller = AdaptiveMPC(mpc_cfg, rls)

    print("Running initial excitation for RLS...")
    init_samples = int(0.2 / Ts)
    x = np.zeros(n_x)
    for k in range(1, init_samples):
        u = np.clip(0.5 * np.random.randn(), mpc_cfg.u_min, mpc_cfg.u_max)
        x = A @ x + B.ravel() * u
        y = (C @ x).item()
        rls.update(y, u)

    print("Starting closed-loop Adaptive MPC simulation...")
    t_start = time.time()
    y_log = np.zeros(N)
    u_log = np.zeros(N)
    x = np.zeros(n_x)

    for k in range(N):
        y = 0.0 if k == 0 else (C @ x).item()
        y_log[k] = y

        if k > 0:
            error = rls.update(y, u_log[k-1])
        else:
            error = 0.0

        u = controller.solve(y, reference[k], dist_est=error)
        u_log[k] = u
        x = A @ x + B.ravel() * u

        if (k + 1) % 500 == 0:
            print(f"  Step {k+1:5d}/{N} | y = {y:.3f} | ref = {reference[k]:.3f} | u = {u:.3f}")

    print(f"Simulation completed in {time.time() - t_start:.2f} seconds.")

    plot_results(t_vec, reference, y_log, u_log, mpc_cfg)

def plot_results(t_vec, reference, y_log, u_log, mpc_cfg):
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['axes.linewidth'] = 1.5
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['xtick.major.size'] = 6
    mpl.rcParams['ytick.major.size'] = 6
    mpl.rcParams['xtick.major.width'] = 1.5
    mpl.rcParams['ytick.major.width'] = 1.5
    mpl.rcParams['font.size'] = 14
    mpl.rcParams['axes.labelsize'] = 16
    mpl.rcParams['axes.titlesize'] = 18
    mpl.rcParams['legend.fontsize'] = 14

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t_vec, reference, 'k--', linewidth=2.0, label='Reference')
    axes[0].plot(t_vec, y_log, color='#0072BD', linewidth=2.2, label='Terminal Voltage $V_t$')
    axes[0].set_ylabel('Voltage [p.u.]')
    axes[0].set_title('Adaptive Data-Driven MPC: Closed-loop Response', pad=15)
    axes[0].legend(loc='upper right', framealpha=1.0, edgecolor='black')
    axes[0].set_ylim([-0.3, 1.5])
    axes[0].grid(True, which='major', linestyle='--', linewidth=0.8, alpha=0.7)

    axes[1].step(t_vec, u_log, color='#D95319', where='post', linewidth=2.0, label='Control Input $u$')
    axes[1].axhline(mpc_cfg.u_min, color='gray', linestyle=':', linewidth=1.5, alpha=0.8, label='Actuator Limits')
    axes[1].axhline(mpc_cfg.u_max, color='gray', linestyle=':', linewidth=1.5, alpha=0.8)
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Control Input')
    axes[1].legend(loc='upper right', framealpha=1.0, edgecolor='black')
    axes[1].set_ylim([mpc_cfg.u_min - 0.8, mpc_cfg.u_max + 0.8])
    axes[1].grid(True, which='major', linestyle='--', linewidth=0.8, alpha=0.7)

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/adaptive_mpc_avr_results.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/adaptive_mpc_avr_results.pdf', format='pdf', bbox_inches='tight')
    print("Figures saved in 'results/' directory.")

if __name__ == "__main__":
    run_simulation()
