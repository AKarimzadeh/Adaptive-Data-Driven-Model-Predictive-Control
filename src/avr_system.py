import numpy as np
from scipy import signal
from dataclasses import dataclass

@dataclass
class AVRParameters:
    """Nominal parameters of the AVR system."""
    Ka: float = 11.0
    Ta: float = 0.06
    Ke: float = 3.0
    Te: float = 0.7
    Kg: float = 0.85
    Tg: float = 1.5
    Ks: float = 1.0
    Tsensor: float = 0.0305


def build_continuous_plant(params: AVRParameters):
    """Build continuous-time state-space model of the AVR."""
    A = np.array([
        [-1/params.Ta, 0, 0, -params.Ka/params.Ta],
        [params.Ke/params.Te, -1/params.Te, 0, 0],
        [0, params.Kg/params.Tg, -1/params.Tg, 0],
        [0, 0, params.Ks/params.Tsensor, -1/params.Tsensor]
    ])
    B = np.array([[params.Ka/params.Ta], [0.], [0.], [0.]])
    C = np.array([[0., 0., 1., 0.]])  # Terminal voltage Vt
    D = np.array([[0.]])
    return A, B, C, D


def discretize_plant(A, B, C, D, Ts: float):
    """Discretize the continuous plant using zero-order hold."""
    sys_c = signal.StateSpace(A, B, C, D)
    sys_d = sys_c.to_discrete(Ts)
    return sys_d.A, sys_d.B, sys_d.C, sys_d.D
