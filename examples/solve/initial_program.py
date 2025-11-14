"""
Missing-term discovery scaffold for OpenEvolve.

Public API expected by the evaluator:

- prediction(a: np.ndarray) -> np.ndarray

  Given an array of scale factors a (dimensionless),
  returns the predicted H(a) including a baseline
  ΛCDM part plus an evolvable correction term.

The only part that evolution is allowed to modify is
the function `correction_term(a)` inside the EVOLVE block.
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np

ArrayLike = Union[float, np.ndarray]

# ------------------------------------------------------------
# Baseline cosmology parameters (can be tuned separately)
# ------------------------------------------------------------
H0_DEFAULT = 70.0           # km/s/Mpc, just a placeholder
OMEGA_M_DEFAULT = 0.3
OMEGA_R_DEFAULT = 9.0e-5
OMEGA_L_DEFAULT = 1.0 - OMEGA_M_DEFAULT - OMEGA_R_DEFAULT


def H_LCDM(a: ArrayLike,
           H0: float = H0_DEFAULT,
           Omega_m: float = OMEGA_M_DEFAULT,
           Omega_r: float = OMEGA_R_DEFAULT,
           Omega_L: float = OMEGA_L_DEFAULT) -> ArrayLike:
    """
    Baseline ΛCDM H(a) in units of H0.

    This is the "trusted" part of the model. Evolution is not allowed
    to modify this function; it only modifies the correction_term
    defined in the EVOLVE block.
    """
    a = np.asarray(a, dtype=float)
    # Avoid division by zero at a = 0
    a = np.clip(a, 1e-6, None)

    return H0 * np.sqrt(
        Omega_r * a**(-4) +
        Omega_m * a**(-3) +
        Omega_L
    )


# ------------------------------------------------------------
# EVOLVE BLOCK: missing term / correction factor
# ------------------------------------------------------------

# EVOLVE-BLOCK-START
def correction_term(a: ArrayLike) -> ArrayLike:
    """
    Evolvable correction δ(a) to the baseline model.

    Definition:
        H_total(a) = H_LCDM(a) * (1.0 + correction_term(a))

    Requirements for evolution:
    - a is dimensionless scale factor (0 < a <= 1 for real data).
    - This function should return a dimensionless correction.
    - For physical plausibility, evolution should prefer small
      corrections at early times (a << 1) unless strongly supported
      by data.

    Initial baseline: no missing term, δ(a) = 0.

    Evolution is free to replace this with a more complex expression,
    for example:
        return eps0 + eps2 * a**2
    or any other smooth, low-amplitude function that improves the fit.
    """
    a = np.asarray(a, dtype=float)
    # No correction in the initial program
    return np.zeros_like(a)
# EVOLVE-BLOCK-END


# ------------------------------------------------------------
# Public prediction API used by the evaluator
# ------------------------------------------------------------

def prediction(a: ArrayLike,
               H0: float = H0_DEFAULT,
               Omega_m: float = OMEGA_M_DEFAULT,
               Omega_r: float = OMEGA_R_DEFAULT,
               Omega_L: float = OMEGA_L_DEFAULT) -> ArrayLike:
    """
    Full model prediction H(a) including the baseline ΛCDM part
    and the evolvable correction term.

    The evaluator will call this function on an array of scale
    factors a_data and compare the result to observed H_data.

    Parameters can be fixed here or tuned in the evaluator.
    """
    a = np.asarray(a, dtype=float)
    H_base = H_LCDM(a, H0=H0, Omega_m=Omega_m,
                    Omega_r=Omega_r, Omega_L=Omega_L)
    delta = correction_term(a)
    return H_base * (1.0 + delta)
