"""
Given an evolved missing-term correction δ(a),
this script computes the best-fit graviton mass m_g
consistent with your massive-graviton H_mg_phenomenological model.

The procedure is:

1. Compute δ(a) from the evolved program.
2. Compute H_LCDM(a).
3. Infer the effective massive-gravity correction:

       H_mg_eff(a) = H_LCDM(a)^2 * ( (1+δ(a))^2 - 1 )

   (this is exact to linear order: H_total^2 = H_LCDM^2 (1 + 2δ + δ^2))

4. Fit:

       H_mg_eff(a)  ≈  H_mg_phenomenological(a, m_g, H0)

5. Return best-fit m_g.
"""

import numpy as np
from scipy.optimize import minimize_scalar
import importlib.util
import pathlib

# ----------------------------------------------------------
# Import your massive-gravity model (same file you already use)
# ----------------------------------------------------------
from examples.solve.initial_program import H_mg_phenomenological, M_G_REF_global as M_G_REF

# ----------------------------------------------------------
# Default cosmology
# ----------------------------------------------------------
H0_DEFAULT = 2.2e-18
OMEGA_M_DEFAULT = 0.3
OMEGA_R_DEFAULT = 9e-5
OMEGA_L_DEFAULT = 1.0 - OMEGA_M_DEFAULT - OMEGA_R_DEFAULT


# ----------------------------------------------------------
# Load an evolved candidate program
# ----------------------------------------------------------
def load_candidate(path):
    """
    Load a Python module purely by path.
    """
    path = pathlib.Path(path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------
# Compute effective massive-gravity correction
# ----------------------------------------------------------
def compute_Hmg_eff(candidate, a):
    """
    Computes H_mg_eff(a) from:
       H_total(a) = H_LCDM(a) * (1 + δ(a))
       H_total^2 = H_LCDM^2 (1 + 2δ + δ^2)
       H_mg_eff = H_total^2 - H_LCDM^2
    """
    H_LCDM = candidate.H_LCDM(a)
    delta = candidate.correction_term(a)

    H_tot_sq = (H_LCDM * (1 + delta)) ** 2
    H_LCDM_sq = H_LCDM**2

    return H_tot_sq - H_LCDM_sq


# ----------------------------------------------------------
# Loss function for fitting m_g
# ----------------------------------------------------------
def mg_loss(m_g, a, Hmg_eff, H0):
    """
    L2 loss between the effective correction curve and the
    model H_mg_phenomenological(a, m_g, H0).
    """
    pred = np.array([H_mg_phenomenological(ai, m_g, H0) for ai in a])
    return np.mean((Hmg_eff - pred) ** 2)


# ----------------------------------------------------------
# Fit the best graviton mass
# ----------------------------------------------------------
def fit_graviton_mass(candidate_path):
    """
    Given a candidate evolved program, compute best-fit m_g.
    """

    candidate = load_candidate(candidate_path)

    # grid
    a = np.linspace(0.05, 1.0, 100)

    # compute effective massive gravity term
    Hmg_eff = compute_Hmg_eff(candidate, a)

    # search range:
    # 10^-75 kg to 10^-28 kg (covers all physically relevant masses)
    result = minimize_scalar(
        mg_loss, bounds=(1e-75, 1e-28), args=(a, Hmg_eff, H0_DEFAULT), method="bounded"
    )

    best_mg = result.x
    return {"best_m_g": best_mg, "best_m_g_over_Mref": best_mg / M_G_REF, "loss": result.fun}


# ----------------------------------------------------------
# Script entry
# ----------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python graviton_mass_fit.py path/to/candidate.py")
        exit(1)

    path = sys.argv[1]
    result = fit_graviton_mass(path)

    print("\n===== MASSIVE GRAVITY FIT RESULTS =====")
    print(f" Best-fit graviton mass m_g (kg):       {result['best_m_g']:.4e}")
    print(f" Relative to reference M_G_REF:         {result['best_m_g_over_Mref']:.4e}")
    print(f" Fit loss:                              {result['loss']:.4e}")
    print("=========================================\n")
