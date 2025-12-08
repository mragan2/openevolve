"""
Initial program for missing-term reconstruction in H(a).

Defines:
    - H_LCDM(a)
    - correction_term(a)
    - prediction(a)

Fully compatible with STRICT evaluator and STRICT prompt.
"""

from __future__ import annotations
import numpy as np

# ------------------------------------------------------------
# Default cosmological parameters (must match evaluator)
# ------------------------------------------------------------
H0_DEFAULT = 2.2e-18
OMEGA_M_DEFAULT = 0.3
OMEGA_R_DEFAULT = 9e-5
OMEGA_L_DEFAULT = 1.0 - OMEGA_M_DEFAULT - OMEGA_R_DEFAULT


# ------------------------------------------------------------
# ΛCDM expansion history (unchanged)
# ------------------------------------------------------------
def H_LCDM(a: np.ndarray,
           H0: float = H0_DEFAULT,
           Omega_m: float = OMEGA_M_DEFAULT,
           Omega_r: float = OMEGA_R_DEFAULT,
           Omega_L: float = OMEGA_L_DEFAULT) -> np.ndarray:

    a = np.asarray(a, dtype=float)
    return H0 * np.sqrt(Omega_r / a**4 + Omega_m / a**3 + Omega_L)


# ------------------------------------------------------------
# EVOLVE BLOCK — ONLY THIS WILL BE REWRITTEN
# ------------------------------------------------------------
def correction_term(a):
    """
    Strictly constrained neutral δ(a) starting point.
    Only allowed basis functions are present.
    Coefficients are tiny, safe, and monotonic.
    """

    a = np.asarray(a, dtype=float)

    BASIS_1 = (a**2) / (1.0 + a**2)
    BASIS_2 = (a**3) / (1.0 + a**3)
    BASIS_3 = np.log1p(a)
    BASIS_4 = a
    BASIS_5 = a**2

    # Tiny neutral coefficients — evolution will rewrite them.
    c1 = 0.0
    c2 = 0.0
    c3 = 0.0
    c4 = 0.0
    c5 = 0.0

    delta = (
        c1 * BASIS_1 +
        c2 * BASIS_2 +
        c3 * BASIS_3 +
        c4 * BASIS_4 +
        c5 * BASIS_5
    )

    return delta
# EVOLVE-BLOCK-END


# ------------------------------------------------------------
# Full reconstructed expansion history
# ------------------------------------------------------------
def prediction(a: np.ndarray,
               H0: float = H0_DEFAULT,
               Omega_m: float = OMEGA_M_DEFAULT,
               Omega_r: float = OMEGA_R_DEFAULT,
               Omega_L: float = OMEGA_L_DEFAULT) -> np.ndarray:

    a = np.asarray(a, dtype=float)
    base = H_LCDM(a, H0, Omega_m, Omega_r, Omega_L)
    delta = correction_term(a)
    return base * (1.0 + delta)