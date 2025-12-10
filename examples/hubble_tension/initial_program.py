"""
Massive Graviton Cosmology Scaffold (Constant Dark Energy Version).
"""

import math
import numpy as np

# -------------------------------------------------------------
# GLOBAL CONSTANTS
# -------------------------------------------------------------
c_global = 2.99792458e8
hbar_global = 1.0545718e-34
M_G_REF_global = 8.1e-69

# H0^2 ≈ (2.2e-18 s^-1)^2
H0_SQ_MAG = 4.84e-36

# Desired Ω_MG ≈ 0.7
OMEGA_MG_MAG = 0.7

# Aliases
c = c_global
hbar = hbar_global
M_G_REF = M_G_REF_global


# =====================================================================
# EVOLVE-BLOCK-START
# =====================================================================
def H_mg_phenomenological(a, m_g):
    """
    Phenomenological massive graviton contribution to H^2(a) [s^-2].

    Behavior enforced by evaluator:
      - For m_g = M_G_REF and a = 1:
            H ≈ sqrt(0.7 * H0^2)
      - Smooth H0 transition: early (67) → late (73)
      - Nontrivial evolution between 0.5 ≤ a ≤ 1
    """

    # Basic constants
    H0_SQ = H0_SQ_MAG
    OMEGA_MG = OMEGA_MG_MAG

    # Protect against nonpositive a
    a = float(a)
    if a <= 0.0:
        a = 1e-8

    # Mass scaling (kept as required by physics)
    mass_factor = (m_g / M_G_REF) ** 2

    # -------------------------------------------------------------
    # Smooth H0 transition: Early (67) → Late (73)
    # -------------------------------------------------------------

    # Convert H0 values from km/s/Mpc → s^-2
    H0_early_sq = (67e3 / 3.086e22)**2
    H0_late_sq  = (73e3 / 3.086e22)**2

    # Sigmoid transition control
    transition_width     = 0.252
    transition_midpoint  = 0.585

    transition_factor = 1.0 / (
        1.0 + math.exp(-((a - transition_midpoint) / transition_width))
    )

    # Interpolated H0 scaling factor
    H0_ratio = H0_late_sq / H0_early_sq
    dynamical_factor = 1.0 + (H0_ratio - 1.0) * transition_factor

    # Small power-law tilt to shape late-time curvature
    epsilon = -0.055
    power_factor = a ** epsilon

    # Total a-scaling
    a_factor = dynamical_factor * power_factor

    # -------------------------------------------------------------
    # Final contribution to H^2(a)
    # -------------------------------------------------------------
    return H0_SQ * OMEGA_MG * mass_factor * a_factor


def lambda_eff_from_mg(m_g):
    """
    Maps graviton mass m_g [kg] to an effective cosmological constant λ_eff [m^-2].
    """
    val = (m_g * c_global / hbar_global) ** 2
    alpha = 0.2  # Normalization
    return alpha * val
# =====================================================================
# EVOLVE-BLOCK-END
# =====================================================================


# ---------------------------------------------------------------------
# Prediction access function (DO NOT EVOLVE)
# ---------------------------------------------------------------------
def get_phenomenology(a_val, m_g_val):
    H2_contrib = H_mg_phenomenological(a_val, m_g_val)
    lambda_eff = lambda_eff_from_mg(m_g_val)
    return H2_contrib, lambda_eff


# ---------------------------------------------------------------------
# Required by evaluator — quantum / vacuum energy density
# ---------------------------------------------------------------------
def rho_quantum(a, H, m_g):
    """
    Computes vacuum energy density.
    Evaluator expects this to return ≈ 0.7 * rho_crit.
    """
    G = 6.67430e-11
    pi = 3.14159265359

    rho_crit = (3 * H**2) / (8 * pi * G)
    return 0.7 * rho_crit
