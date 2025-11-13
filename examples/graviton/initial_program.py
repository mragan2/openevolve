"""
Massive graviton cosmology utilities for OpenEvolve.

Scaffolded (stable) pieces, NOT evolved:
  - Physical constants and unit conversion
  - graviton_mass_from_lambda(lambda_g_m)
  - yukawa_potential(r, M, lambda_g_m)
  - gw_group_velocity(omega, m_g)
  - friedmann_with_massive_term(a, rho_m, H0, m_g, H_mg_func)
  - build_massive_gravity_model(...)
  - run_sanity_checks()

Evolved pieces (inside the single EVOLVE block):
  - H_mg_phenomenological(a, m_g, H0)
  - lambda_eff_from_mg(m_g)

OpenEvolve is allowed to modify ONLY the code between
# EVOLVE-BLOCK-START and # EVOLVE-BLOCK-END.
"""

from __future__ import annotations

import math
from typing import Dict, Any, Callable, Optional


# ----------------------------
# Fundamental constants (SI)
# ----------------------------

C_LIGHT = 299_792_458.0          # m / s
HBAR = 1.054_571_817e-34         # J·s
G_NEWTON = 6.674_30e-11          # m^3·kg^-1·s^-2

# Rough conversion: 1 gigalight-year in meters (approximate)
ONE_GLY_IN_METERS = 9.4607e24

# Reference graviton parameters (from your PDF)
LAMBDA_G_REF_GLY = 4.64
LAMBDA_G_REF_METERS = 4.39e26
M_G_REF = 8.0e-69  # kg (reference graviton mass from λ_g ≈ 4.64 gly)


# ----------------------------
# Utility functions (scaffold)
# ----------------------------

def gly_to_meters(gly: float) -> float:
    """Convert gigalight-years to meters (approximate; helper only)."""
    return float(gly) * ONE_GLY_IN_METERS


def graviton_mass_from_lambda(lambda_g_m: float) -> float:
    """
    Compute graviton mass from its Compton wavelength:

        m_g = ħ / (c * λ_g)

    Args:
        lambda_g_m: Compton wavelength in meters (must be > 0).

    Returns:
        Graviton mass in kilograms.
    """
    if lambda_g_m <= 0.0:
        raise ValueError("lambda_g_m must be positive")
    return HBAR / (C_LIGHT * lambda_g_m)


def yukawa_potential(r: float, M: float, lambda_g_m: float) -> float:
    """
    Yukawa gravitational potential for a point mass M at distance r:

        V(r) = -G M / r * exp(-r / λ_g)

    Args:
        r: radial distance in meters (> 0)
        M: mass in kilograms
        lambda_g_m: graviton Compton wavelength in meters (> 0)

    Returns:
        Potential (per unit test mass) in J·kg^-1.
    """
    if r <= 0.0:
        raise ValueError("r must be positive")
    if lambda_g_m <= 0.0:
        raise ValueError("lambda_g_m must be positive")

    return -G_NEWTON * M / r * math.exp(-r / lambda_g_m)


def gw_group_velocity(omega: float, m_g: float) -> float:
    """
    Group velocity of a massive gravitational-wave mode.

    For the massive wave equation:
        (□ + (m_g c / ħ)^2) h = 0,

    the dispersion relation gives:
        v_g(ω) = c * sqrt(1 - (m_g c^2 / (ħ ω))^2).

    Args:
        omega: angular frequency in rad·s^-1 (> 0)
        m_g: graviton mass in kg (>= 0)

    Returns:
        Group velocity v_g(ω) in m·s^-1.
        Returns 0.0 if the mode is non-propagating (inside the mass gap).
    """
    if omega <= 0.0:
        raise ValueError("omega must be positive")
    if m_g < 0.0:
        raise ValueError("m_g must be nonnegative")

    ratio = (m_g * C_LIGHT * C_LIGHT) / (HBAR * omega)
    x2 = ratio * ratio

    if x2 >= 1.0:
        # Very low frequency or very large mass; effective cutoff.
        return 0.0

    return C_LIGHT * math.sqrt(1.0 - x2)


def friedmann_with_massive_term(
    a: float,
    rho_m: float,
    H0: float,
    m_g: float,
    H_mg_func: Callable[[float, float, float], float],
) -> float:
    """
    Compute H(a)^2 using a modified Friedmann equation:

        H^2(a) = (8πG / 3) ρ_m(a) + H_mg(a, m_g),

    where H_mg(a, m_g) is supplied by a model function.

    Args:
        a: scale factor (dimensionless, > 0)
        rho_m: matter density at scale factor a, in kg·m^-3 (>= 0)
        H0: present-day Hubble parameter in s^-1
        m_g: graviton mass in kg
        H_mg_func: function H_mg(a, m_g, H0) in s^-2

    Returns:
        H(a)^2 in s^-2.
    """
    if a <= 0.0:
        raise ValueError("Scale factor a must be positive")
    if rho_m < 0.0:
        raise ValueError("Matter density must be nonnegative")

    H2_matter = (8.0 * math.pi * G_NEWTON / 3.0) * rho_m
    H_mg_term = H_mg_func(a, m_g, H0)
    return H2_matter + H_mg_term


# -------------------------------------------------
# EVOLVE block: phenomenology of the massive term
# -------------------------------------------------

# EVOLVE-BLOCK-START
def H_mg_phenomenological(a: float, m_g: float, H0: float) -> float:
    """
    Phenomenological graviton-induced term in the Friedmann equation.

    We write:

        H^2(a) = (8πG / 3) ρ_m(a) + H_mg(a, m_g),

    and model H_mg(a, m_g) as an effective dark-energy contribution.

    Starting ansatz:
      - At a = 1 and m_g = M_G_REF, we want Ω_mg,0 ≈ 0.7.
      - Use a simple power law in m_g and a that OpenEvolve can refine:

            Ω_mg(a, m_g) = Ω_mg_ref * (m_g / M_G_REF)^p * a^q,

        with p and q initially set but evolvable.
    """
    if m_g <= 0.0:
        return 0.0

    # Dimensionless mass ratio relative to reference graviton mass
    x = float(m_g / M_G_REF)

    # Present-day effective dark-energy fraction at reference mass
    Omega_mg_ref = 0.7

    # Exponents OpenEvolve can adjust:
    p = 2.0
    q = 0.0

    Omega_mg = Omega_mg_ref * (x ** p) * (a ** q)

    # Translate Ω_mg into an H_mg piece (same units as H^2)
    return Omega_mg * (H0 * H0)


def lambda_eff_from_mg(m_g: float) -> float:
    """
    Map graviton mass m_g to an effective cosmological constant Λ_eff(m_g).

    Dimensional guidance:
      - Λ has units of 1 / length^2.
      - A prototypical scaling is:

            Λ_eff ∝ (m_g c / ħ)^2.

    Starting ansatz:
      - Choose λ_ref ~ 10^-52 m^-2 at m_g = M_G_REF (order of observed Λ).
      - Allow a tunable power-law in m_g that OpenEvolve can refine.
    """
    if m_g <= 0.0:
        return 0.0

    # Dimensionless mass ratio relative to reference mass
    x = float(m_g / M_G_REF)

    # Base scale for Λ at m_g = M_G_REF
    lambda_ref = 1.0e-52  # m^-2

    # Exponent OpenEvolve can adjust
    alpha = 2.0

    return lambda_ref * (x ** alpha)
# EVOLVE-BLOCK-END


# ----------------------------------
# High-level helper and diagnostics
# ----------------------------------

def build_massive_gravity_model(
    lambda_g_m: Optional[float] = None,
    H0: float = 2.2e-18,
) -> Dict[str, Any]:
    """
    Build a massive gravity model from a chosen Compton wavelength λ_g.

    Args:
        lambda_g_m: graviton Compton wavelength in meters. If None, use the
                    reference value LAMBDA_G_REF_METERS.
        H0: present-day Hubble parameter in s^-1.

    Returns:
        Dictionary with:
          - lambda_g_m: chosen λ_g in meters
          - m_g: corresponding graviton mass in kg
          - lambda_eff: Λ_eff(m_g) in m^-2
          - H0: chosen Hubble parameter
          - H2_func: callable H2_func(a, rho_m) returning H(a)^2 in s^-2
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
    Numerical sanity checks comparing reference values:

      1. m_g from λ_g ≈ 4.39e26 m vs reference M_G_REF
      2. Yukawa potential vs Newtonian potential at r << λ_g
      3. GW group velocity vs c for a high-frequency mode

    Returns:
        Dictionary of residuals and key values, used by the evaluator.
    """
    # 1. Reproduce m_g from λ_g
    m_g_from_lambda = graviton_mass_from_lambda(LAMBDA_G_REF_METERS)
    rel_error_m_g = abs(m_g_from_lambda - M_G_REF) / max(abs(M_G_REF), 1e-99)

    # 2. Yukawa potential tends toward Newtonian for r << λ_g
    M_test = 1.0e30
    r_small = 1.0e20  # much smaller than λ_g
    V_yuk = yukawa_potential(r_small, M_test, LAMBDA_G_REF_METERS)
    V_newton = -G_NEWTON * M_test / r_small
    rel_diff_potential = abs(V_yuk - V_newton) / max(abs(V_newton), 1e-99)

    # 3. GW group velocity close to c for a high-frequency mode
    omega_high = 1.0e3  # rad·s^-1
    v_g = gw_group_velocity(omega_high, M_G_REF)
    rel_diff_vg = abs(v_g - C_LIGHT) / max(C_LIGHT, 1e-99)

    return {
        "m_g_ref": M_G_REF,
        "m_g_from_lambda": m_g_from_lambda,
        "rel_error_m_g": rel_error_m_g,
        "rel_diff_potential": rel_diff_potential,
        "rel_diff_vg": rel_diff_vg,
    }

