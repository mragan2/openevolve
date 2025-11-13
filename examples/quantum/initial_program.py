"""
Semiclassical cosmology scaffold for OpenEvolve

Goal
-----
We want to evolve an *effective quantum correction* ρ_q(a, H, m_g) that modifies
the Friedmann equation in a controlled, physically-motivated way.

The EVOLVE block contains a single function:

    rho_quantum(a: float, H_classical: float, m_g: float) -> float

OpenEvolve is allowed to rewrite ONLY that function. Everything else is
scaffolding that must remain stable.

Physical picture
----------------
We work with a flat FRW universe, with:

    H^2(a) = (8πG/3) [ρ_m(a) + ρ_r(a) + ρ_Λ + ρ_q(a)]

where:
    ρ_m  ~ a^-3      (pressureless matter)
    ρ_r  ~ a^-4      (radiation)
    ρ_Λ  = const     (cosmological constant–like term)
    ρ_q  = ρ_quantum (unknown quantum correction, to be evolved)

The evaluator can:
  - enforce H(a) > 0 for relevant a
  - require early times to be matter/radiation dominated
  - require late times to look like dark-energy acceleration
  - constrain |ρ_q| to be small compared to ρ_crit at a = 1
  - test smoothness and sign of ρ_q(a) over a grid
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List


# ==========================
# Fundamental constants (SI)
# ==========================

C = 2.99792458e8           # speed of light [m/s]
G = 6.67430e-11            # Newton's constant [m^3 kg^-1 s^-2]
H0_KM_S_MPC = 70.0         # "H0" in km/s/Mpc for intuition
MPC_IN_METERS = 3.085677581491367e22  # [m]

# Convert H0 to SI [s^-1]
H0_SI = (H0_KM_S_MPC * 1_000.0) / MPC_IN_METERS

# Reference graviton mass scale (you can adjust in evaluator if needed)
# Here we set ~ 10^-32 eV/c^2 in SI units:
ELECTRONVOLT_J = 1.602176634e-19  # [J]
EV_OVER_C2_TO_KG = ELECTRONVOLT_J / (C ** 2)  # 1 eV/c^2 in kg
M_G_REF_EV = 1e-32               # ~ phenomenological tiny graviton mass
M_G_REF = M_G_REF_EV * EV_OVER_C2_TO_KG  # reference graviton mass in kg


@dataclass
class CosmologyParams:
    """
    Simple parameter container for a flat FRW cosmology with a quantum correction.

    Attributes
    ----------
    H0 : float
        Present-day Hubble parameter in SI units [s^-1].
    Omega_m0 : float
        Present-day matter density fraction.
    Omega_r0 : float
        Present-day radiation density fraction.
    Omega_L0 : float
        Present-day cosmological constant / dark-energy fraction.
    m_g : float
        Graviton mass in kg (can be evolved or fixed by evaluator).
    """
    H0: float = H0_SI
    Omega_m0: float = 0.3
    Omega_r0: float = 5e-5
    Omega_L0: float = 0.7
    m_g: float = M_G_REF


def critical_density(H: float) -> float:
    """
    Critical density ρ_crit = 3 H^2 / (8πG).

    Parameters
    ----------
    H : float
        Hubble parameter in [s^-1].

    Returns
    -------
    float
        Critical density in [kg/m^3].
    """
    return 3.0 * H * H / (8.0 * math.pi * G)


def classical_background_densities(a: float, params: CosmologyParams) -> Dict[str, float]:
    """
    Compute classical (non-quantum) energy densities for matter, radiation, and Λ.

    Parameters
    ----------
    a : float
        Scale factor (a = 1 today).
    params : CosmologyParams
        Cosmological parameters.

    Returns
    -------
    dict
        Dictionary with keys 'rho_m', 'rho_r', 'rho_L', 'rho_crit0'.
    """
    if a <= 0.0:
        raise ValueError("Scale factor 'a' must be positive.")

    H0 = params.H0
    rho_crit0 = critical_density(H0)

    rho_m0 = params.Omega_m0 * rho_crit0
    rho_r0 = params.Omega_r0 * rho_crit0
    rho_L0 = params.Omega_L0 * rho_crit0

    rho_m = rho_m0 / (a ** 3)   # matter: a^-3
    rho_r = rho_r0 / (a ** 4)   # radiation: a^-4
    rho_L = rho_L0              # constant

    return {
        "rho_m": rho_m,
        "rho_r": rho_r,
        "rho_L": rho_L,
        "rho_crit0": rho_crit0,
    }


def classical_H_squared(a: float, params: CosmologyParams) -> float:
    """
    Standard Friedmann equation without quantum corrections:

        H_classical^2 = (8πG/3) (ρ_m + ρ_r + ρ_L)

    Parameters
    ----------
    a : float
        Scale factor.
    params : CosmologyParams
        Cosmological parameters.

    Returns
    -------
    float
        H_classical^2 in [s^-2].
    """
    dens = classical_background_densities(a, params)
    rho_total = dens["rho_m"] + dens["rho_r"] + dens["rho_L"]
    return (8.0 * math.pi * G / 3.0) * rho_total


# EVOLVE-BLOCK-START

def rho_quantum(a: float, H_classical: float, m_g: float) -> float:
    """
    Effective quantum correction ρ_q(a, H_classical, m_g).

    This implementation is tuned for the second phase of evolution. It retains
    physical plausibility while further suppressing the present-day and early
    contributions compared to earlier candidates. The correction scales with
    the squared mass ratio and decays exponentially with the scale factor. A
    very small amplitude ensures that the quantum term has a negligible
    influence on the Hubble parameter today, improving the H0 alignment and
    reducing the quantum fraction at a=1.

    Parameters
    ----------
    a : float
        Scale factor (a > 0).
    H_classical : float
        Classical Hubble parameter (s⁻¹) without quantum corrections.
    m_g : float
        Graviton mass (kg).

    Returns
    -------
    float
        Quantum energy density ρ_q in [kg/m³].
    """
    # Guard against non-physical inputs
    if a <= 0.0 or m_g <= 0.0:
        return 0.0

    # Local critical density at the given Hubble rate
    rho_crit = critical_density(H_classical)

    # Choose a very small amplitude so the quantum term remains subdominant
    amplitude = 1e-6

    # Exponential decay with scale factor; ensures the correction decreases
    # smoothly toward the future (a → ∞) and does not blow up at early times
    decay_factor = exp(-a)

    # Dimensionless mass ratio squared
    mass_ratio = (m_g / M_G_REF) ** 2

    # Compute the quantum density
    rho_q = amplitude * mass_ratio * decay_factor * rho_crit
    return rho_q

# EVOLVE-BLOCK-END

def H_squared_with_quantum(a: float, params: CosmologyParams) -> float:
    """
    Full H^2(a) including the quantum correction:

        H^2(a) = H_classical^2(a) + (8πG/3) ρ_q(a, H_classical, m_g)

    Parameters
    ----------
    a : float
        Scale factor.
    params : CosmologyParams
        Cosmological parameters.

    Returns
    -------
    float
        H^2(a) including quantum correction [s^-2].
    """
    H2_classical = classical_H_squared(a, params)
    if H2_classical <= 0.0:
        # Guard against numerical issues: return 0 to let evaluator penalize.
        return 0.0

    H_classical = math.sqrt(H2_classical)
    rho_q = rho_quantum(a, H_classical, params.m_g)

    H2_full = H2_classical + (8.0 * math.pi * G / 3.0) * rho_q
    return max(H2_full, 0.0)


def sample_H_of_a_grid(params: CosmologyParams, a_values: List[float]) -> Dict[float, float]:
    """
    Convenience function: compute H(a) on a grid of scale factors.

    Parameters
    ----------
    params : CosmologyParams
        Cosmological parameters.
    a_values : list of float
        Scale factor values.

    Returns
    -------
    dict
        Mapping a -> H(a) in [s^-1] including quantum correction.
    """
    result = {}
    for a in a_values:
        H2 = H_squared_with_quantum(a, params)
        if H2 <= 0.0:
            result[a] = 0.0
        else:
            result[a] = math.sqrt(H2)
    return result


def run_sanity_checks() -> Dict[str, float]:
    """
    Basic sanity checks for manual testing and for use by an evaluator.

    Returns a dictionary with:
      - H0_classical: H_classical(a=1)
      - H0_with_quantum: H(a=1) with quantum correction
      - ratio_H0: H_with_quantum / H_classical
      - rho_q_today_over_crit0: ρ_q(a=1)/ρ_crit0
      - H_at_early_a: H(a_early) with quantum (e.g. a=0.1)
    """
    params = CosmologyParams()
    a0 = 1.0
    a_early = 0.1

    H2_classical_today = classical_H_squared(a0, params)
    H2_full_today = H_squared_with_quantum(a0, params)

    if H2_classical_today <= 0.0:
        H0_classical = 0.0
    else:
        H0_classical = math.sqrt(H2_classical_today)

    if H2_full_today <= 0.0:
        H0_full = 0.0
    else:
        H0_full = math.sqrt(H2_full_today)

    dens0 = classical_background_densities(a0, params)
    rho_crit0 = dens0["rho_crit0"]
    rho_q_today = rho_quantum(a0, H0_classical, params.m_g)

    H2_early = H_squared_with_quantum(a_early, params)
    H_early = math.sqrt(H2_early) if H2_early > 0.0 else 0.0

    ratio_H0 = H0_full / H0_classical if H0_classical > 0.0 else 0.0
    rho_q_frac = rho_q_today / rho_crit0 if rho_crit0 > 0.0 else 0.0

    return {
        "H0_classical": H0_classical,
        "H0_with_quantum": H0_full,
        "ratio_H0": ratio_H0,
        "rho_q_today_over_crit0": rho_q_frac,
        "H_at_early_a": H_early,
    }


if __name__ == "__main__":
    # Manual quick test when running this file directly.
    checks = run_sanity_checks()
    print("Sanity checks for semiclassical cosmology with quantum correction:")
    for k, v in checks.items():
        print(f"  {k}: {v:.6e}")
