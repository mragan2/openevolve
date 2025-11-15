"""
Initial program for discovering missing terms in the Hubble expansion history.

The evaluator for this task:
  • Stitches ONLY the EVOLVE block below
  • Expects prediction(a) to exist
  • Uses a massive-graviton H(a) as the target truth
  • Scores how well correction_term(a) reproduces that behaviour

Evolution is allowed to edit ONLY the function correction_term(a)
located between:

    # EVOLVE-BLOCK-START
    ...
    # EVOLVE-BLOCK-END
"""

from __future__ import annotations

import numpy as np
from typing import Union

ArrayLike = Union[float, np.ndarray]

# =============================================================
#  Physical constants and baseline cosmology
# =============================================================

H0_DEFAULT = 2.2e-18   # s^-1 (matches evaluator)
OMEGA_M_DEFAULT = 0.3
OMEGA_R_DEFAULT = 9.0e-5
OMEGA_L_DEFAULT = 1.0 - OMEGA_M_DEFAULT - OMEGA_R_DEFAULT


def H_LCDM(a: ArrayLike,
           H0: float = H0_DEFAULT,
           Omega_m: float = OMEGA_M_DEFAULT,
           Omega_r: float = OMEGA_R_DEFAULT,
           Omega_L: float = OMEGA_L_DEFAULT) -> ArrayLike:
    """
    Baseline ΛCDM expansion history.
    Evolvable corrections multiply this baseline.

    This must match EXACTLY the evaluator’s version.
    """
    a = np.asarray(a, dtype=float)
    a = np.clip(a, 1e-9, None)

    return H0 * np.sqrt(
        Omega_r * a**(-4) +
        Omega_m * a**(-3) +
        Omega_L
    )


# =============================================================
#  EVOLVE BLOCK — ONLY THIS PART IS MUTABLE
# =============================================================

# EVOLVE-BLOCK-START
def correction_term(a: ArrayLike) -> ArrayLike:
    """
    Evolvable missing term δ(a).

    Initial baseline: no correction. The evaluator teaches evolution to
    replace this with a small, smooth, dimensionless function that makes:

        prediction(a) = H_LCDM(a) * (1 + δ(a))

    approximate the H(a) produced by a massive-graviton model.
    """
    a = np.asarray(a, dtype=float)
    return np.zeros_like(a)
# EVOLVE-BLOCK-END


# =============================================================
#  Public API used by evaluator
# =============================================================

def prediction(a: ArrayLike,
               H0: float = H0_DEFAULT,
               Omega_m: float = OMEGA_M_DEFAULT,
               Omega_r: float = OMEGA_R_DEFAULT,
               Omega_L: float = OMEGA_L_DEFAULT) -> ArrayLike:
    """
    Final predicted expansion history:
        H_total(a) = H_LCDM(a) * (1 + correction_term(a))

    The evaluator compares this curve to the massive-graviton H(a)
    to discover the effective missing physical term.
    """
    a = np.asarray(a, dtype=float)
    base = H_LCDM(a, H0, Omega_m, Omega_r, Omega_L)
    delta = correction_term(a)

    return base * (1.0 + delta)
