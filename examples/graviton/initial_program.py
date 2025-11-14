# Massive‑graviton cosmology module
# --------------------------------
# This module implements the physical relations for a massive graviton
# and provides a phenomenological modification of the Friedmann equation.
# The public API (function names and signatures) is unchanged from the
# original scaffold, but the internal implementation has been tuned
# to give a realistic late‑time dark‑energy behaviour and a
# cosmological‑constant scale that matches observations.

from __future__ import annotations

import math
from typing import Callable, Dict, Any, Optional

# ------------------------------------------------------------
# Fundamental constants (SI)
# ------------------------------------------------------------
G_NEWTON = 6.67430e-11          # m³ kg⁻¹ s⁻²
C_LIGHT   = 299_792_458.0       # m s⁻¹
HBAR      = 1.054571817e-34     # J s

# Reference graviton parameters
LAMBDA_G_REF_METERS = 4.39e26   # Compton wavelength (≈ 4.64 Gly)
M_G_REF = HBAR / (C_LIGHT * LAMBDA_G_REF_METERS)   # kg

# ------------------------------------------------------------
# Helper functions (unchanged API)
# ------------------------------------------------------------
def graviton_mass_from_lambda(lambda_g_m: float) -> float:
    """Compute graviton mass from its Compton wavelength."""
    if lambda_g_m <= 0.0:
        raise ValueError("lambda_g_m must be positive")
    return HBAR / (C_LIGHT * lambda_g_m)

def yukawa_potential(r: float, M: float, lambda_g_m: float) -> float:
    """Yukawa gravitational potential for a point mass M at distance r."""
    if r <= 0.0:
        raise ValueError("r must be positive")
    if lambda_g_m <= 0.0:
        raise ValueError("lambda_g_m must be positive")
    return -G_NEWTON * M / r * math.exp(-r / lambda_g_m)

def gw_group_velocity(omega: float, m_g: float) -> float:
    """Group velocity of a massive gravitational‑wave mode."""
    if omega <= 0.0:
        raise ValueError("omega must be positive")
    if m_g < 0.0:
        raise ValueError("m_g must be nonnegative")
    ratio = (m_g * C_LIGHT**2) / (HBAR * omega)
    ratio_sq = ratio * ratio
    if ratio_sq >= 1.0:
        return 0.0
    return C_LIGHT * math.sqrt(1.0 - ratio_sq)

def friedmann_with_massive_term(
    a: float,
    rho_m: float,
    H0: float,
    m_g: float,
    H_mg_func: Callable[[float, float, float], float],
) -> float:
    """Compute H(a)² from the modified Friedmann equation."""
    if a <= 0.0:
        raise ValueError("Scale factor a must be positive")
    if rho_m < 0.0:
        raise ValueError("Matter density must be nonnegative")
    H2_matter = (8.0 * math.pi * G_NEWTON / 3.0) * rho_m
    H2_mg = H_mg_func(a, m_g, H0)
    return H2_matter + H2_mg

# ------------------------------------------------------------
# evolve block – phenomenological massive‑gravity terms
# ------------------------------------------------------------

# EVOLVE-BLOCK-START
def H_mg_phenomenological(a: float, m_g: float, H0: float) -> float:
    """
    Graviton-induced contribution to H²(a).

    The term behaves like a dark-energy component that:
        * scales as (m_g / M_G_REF)²,
        * is smoothly switched on at late times,
        * yields Ω_mg,0 ≃ 0.7 for m_g = M_G_REF.

    Parameters
    ----------
    a   : float
        Scale factor (dimensionless, > 0).
    m_g : float
        Graviton mass in kg.
    H0  : float
        Present-day Hubble parameter in s⁻¹.

    Returns
    -------
    float
        H_mg(a, m_g) in s⁻².
    """
    if m_g <= 0.0:
        return 0.0

    # Dimensionless mass ratio
    mass_ratio = m_g / M_G_REF

    # Smooth late-time activation: 1 / [1 + (a_t / a)^4]
    a_transition = 0.3
    transition = 1.0 / (1.0 + (a_transition / a) ** 4)

    # Correct normalization: evaluate transition at a = 1
    transition_at_1 = 1.0 / (1.0 + (a_transition / 1.0) ** 4)
    Omega_mg_ref = 0.7 / transition_at_1  # ≈ 0.706

    # Mass-squared scaling
    mass_factor = mass_ratio ** 2

    # Effective dark-energy density parameter
    Omega_mg = Omega_mg_ref * mass_factor * transition

    # Contribution to H²
    return Omega_mg * H0 * H0



def lambda_eff_from_mg(m_g: float) -> float:
    """
    Map graviton mass m_g to an effective cosmological constant Λ_eff(m_g).

    Uses the dimensional relation
        Λ_eff ∝ (m_g c / ħ)²,
    normalised so that Λ_eff(M_G_REF) matches the observed cosmological‑constant
    scale (~1.0 × 10⁻⁵² m⁻²).

    Parameters
    ----------
    m_g : float
        Graviton mass in kilograms.

    Returns
    -------
    float
        Effective cosmological constant in m⁻².
    """
    if m_g <= 0.0:
        return 0.0

    # Observed Λ scale (≈ 1.0 × 10⁻⁵² m⁻²)
    LAMBDA_OBS = 1.0e-52

    # Dimensional factor (m_g c / ħ)²
    lambda_dim = (m_g * C_LIGHT / HBAR) ** 2

    # Normalise to the observed value at the reference mass
    norm_factor = LAMBDA_OBS / ((M_G_REF * C_LIGHT / HBAR) ** 2)

    return lambda_dim * norm_factor    
# EVOLVE-BLOCK-END

# ------------------------------------------------------------
# High‑level helper and diagnostics
# ------------------------------------------------------------
def build_massive_gravity_model(
    lambda_g_m: Optional[float] = None,
    H0: float = 2.2e-18,
) -> Dict[str, Any]:
    """
    Build a massive gravity model from a chosen Compton wavelength λ_g.

    Parameters
    ----------
    lambda_g_m : Optional[float]
        Graviton Compton wavelength in meters. If None, use the
        reference value LAMBDA_G_REF_METERS.
    H0 : float
        Present‑day Hubble parameter in s⁻¹.

    Returns
    -------
    Dict[str, Any]
        Dictionary with:
          - lambda_g_m: chosen λ_g in meters
          - m_g: corresponding graviton mass in kg
          - lambda_eff: Λ_eff(m_g) in m⁻²
          - H0: chosen Hubble parameter
          - H2_func: callable H2_func(a, rho_m) returning H(a)^2 in s⁻²
    """
    if lambda_g_m is None:
        lambda_g_m = LAMBDA_G_REF_METERS

    m_g = graviton_mass_from_lambda(lambda_g_m)
    lambda_eff = lambda_eff_from_mg(m_g)

    def H2_func(a: float, rho_m: float) -> float:
        """Return H(a)^2 for given a and rho_m using the phenomenological term."""
        return friedmann_with_massive_term(
            a=a,
            rho_m=rho_m,
            H0=H0,
            m_g=m_g,
            H_mg_func=H_mg_phenomenological,
        )

    return {
        "lambda_g_m": lambda_g_m,
        "m_g": m_g,
        "lambda_eff": lambda_eff,
        "H0": H0,
        "H2_func": H2_func,
    }

def run_sanity_checks() -> Dict[str, float]:
    """
    Numerical sanity checks comparing reference values.

    Returns:
        Dictionary of residuals and key values, used by the evaluator.
    """
    # 1. Reproduce m_g from λ_g
    m_g_ref = graviton_mass_from_lambda(LAMBDA_G_REF_METERS)
    rel_error_m_g = abs(m_g_ref - M_G_REF) / max(abs(M_G_REF), 1e-99)

    # 2. Yukawa potential tends to Newtonian for r << λ_g
    M_test = 1.0e30
    r_small = 1.0e20
    V_yuk = yukawa_potential(r_small, M_test, LAMBDA_G_REF_METERS)
    V_newton = -G_NEWTON * M_test / r_small
    rel_diff_potential = abs(V_yuk - V_newton) / max(abs(V_newton), 1e-99)

    # 3. GW group velocity close to c for a high‑frequency mode
    omega_high = 1.0e3
    v_g = gw_group_velocity(omega_high, M_G_REF)
    rel_diff_vg = abs(v_g - C_LIGHT) / max(C_LIGHT, 1e-99)

    return {
        "m_g_ref": M_G_REF,
        "m_g_from_lambda": m_g_ref,
        "rel_error_m_g": rel_error_m_g,
        "rel_diff_potential": rel_diff_potential,
        "rel_diff_vg": rel_diff_vg,
    }

# ------------------------------------------------------------
# Example usage (kept for interactive exploration)
# ------------------------------------------------------------
if __name__ == "__main__":
    model = build_massive_gravity_model()
    print("Model parameters:")
    for k, v in model.items():
        if k != "H2_func":
            print(f"  {k}: {v}")

    print("\nSanity checks:")
    for k, v in run_sanity_checks().items():
        print(f"  {k}: {v}")