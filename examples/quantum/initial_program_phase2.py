"""
Semiclassical cosmology scaffold for OpenEvolve (Phase 2)

This file contains a copy of the original semiclassical cosmology scaffold used
for the quantum correction evolution example, but with an updated EVOLVE block
that serves as a new baseline for a more demanding second phase of evolution.

The goal of the second phase is to encourage non‑trivial quantum corrections
that remain small but non‑zero at the present epoch while respecting all
physical constraints.  The EVOLVE block below defines a baseline
``rho_quantum`` function that produces a smooth, positive correction which
peaks at early times at a few percent of the local critical density and
decays gently towards the future without vanishing exactly at a = 1.  This
baseline is intentionally conservative so that subsequent evolution can
explore departures from it in response to a stricter evaluator.

Only the code between the ``# EVOLVE-BLOCK-START`` and
``# EVOLVE-BLOCK-END`` markers may be modified by OpenEvolve.  The rest of
the file is the fixed scaffold used for cosmological computations and must
remain unchanged.
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
        Present‑day Hubble parameter in SI units [s⁻¹].
    Omega_m0 : float
        Present‑day matter density fraction.
    Omega_r0 : float
        Present‑day radiation density fraction.
    Omega_L0 : float
        Present‑day cosmological constant / dark‑energy fraction.
    m_g : float
        Graviton mass in kg (can be evolved or fixed by the evaluator).
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
        Hubble parameter in [s⁻¹].

    Returns
    -------
    float
        Critical density in [kg/m³].
    """
    return 3.0 * H * H / (8.0 * math.pi * G)


def classical_background_densities(a: float, params: CosmologyParams) -> Dict[str, float]:
    """
    Compute classical (non‑quantum) energy densities for matter, radiation,
    and Λ.

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

        H_classical^2 = (8πG/3) (ρ_m + ρ_r + ρ_Λ)

    Parameters
    ----------
    a : float
        Scale factor.
    params : CosmologyParams
        Cosmological parameters.

    Returns
    -------
    float
        H_classical^2 in [s⁻²].
    """
    dens = classical_background_densities(a, params)
    rho_total = dens["rho_m"] + dens["rho_r"] + dens["rho_L"]
    return (8.0 * math.pi * G / 3.0) * rho_total


# EVOLVE-BLOCK-START

def rho_quantum(a: float, H_classical: float, m_g: float) -> float:
    """
    Effective quantum correction ρ_q(a, H_classical, m_g) for phase 2.

    This baseline implementation is designed for the second evolution phase.
    It produces a smooth, positive quantum energy density that remains
    subdominant at all times but does not vanish identically at a = 1.
    The correction scales with the squared graviton mass ratio and a
    dimensionless decay factor that interpolates between roughly 3 % of
    ρ_crit at early times (a ≈ 0.05–0.3) and 3 % at the present epoch
    (a = 1).  The functional form combines a gentle rise with a decay
    to encourage non‑trivial variation across the scale factor range.  This
    provides a reasonable seed for evolution that will be encouraged to
    explore more complex profiles by the stricter evaluator in phase 2.

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
    # Guard against non‑physical inputs
    if a <= 0.0 or m_g <= 0.0:
        return 0.0

    # Local critical density at the given Hubble rate
    rho_crit = critical_density(H_classical)

    # Amplitude chosen so that the dimensionless ratio rho_q/rho_crit
    # is of order a few percent across the relevant scale factor range.
    amplitude = 0.06

    # Smooth decay factor combining a gentle rise with decay.  For a→0 it
    # approaches ~0.5, and for a→1 it equals 0.5.  Dividing by (1 + a^2)
    # suppresses the correction at late times while allowing a nonzero
    # present‑day contribution.  This functional form maintains monotonic
    # behaviour and ensures the quantum fraction remains bounded.
    decay_factor = (0.5 + 0.5 * a) / (1.0 + a * a)

    # Dimensionless mass ratio squared
    mass_ratio = (m_g / M_G_REF) ** 2

    # Compute the quantum energy density
    rho_q = amplitude * mass_ratio * decay_factor * rho_crit
    return rho_q

# EVOLVE-BLOCK-END

def H_squared_with_quantum(a: float, params: CosmologyParams) -> float:
    """
    Full H²(a) including the quantum correction:

        H²(a) = H_classical²(a) + (8πG/3) ρ_q(a, H_classical, m_g)

    Parameters
    ----------
    a : float
        Scale factor.
    params : CosmologyParams
        Cosmological parameters.

    Returns
    -------
    float
        H²(a) including the quantum correction [s⁻²].
    """
    H2_classical = classical_H_squared(a, params)
    if H2_classical <= 0.0:
        # Guard against numerical issues: return 0 to let the evaluator penalize.
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
    """
    results: Dict[float, float] = {}
    for a in a_values:
        H2 = H_squared_with_quantum(a, params)
        if H2 <= 0.0:
            results[a] = 0.0
        else:
            results[a] = math.sqrt(H2)
    return results


def run_sanity_checks() -> Dict[str, float]:
    """
    Perform basic sanity checks on the cosmological model to support the evaluator.

    Returns
    -------
    dict
        Dictionary containing:
            - ratio_H0: ratio of the full Hubble rate to the classical value at a=1.
            - rho_q_today_over_crit0: quantum density fraction at a=1 relative to the
              present‑day critical density.
            - H_at_early_a: full H(a) evaluated at a small scale factor (e.g. a=0.1).
    """
    params = CosmologyParams()
    a_today = 1.0
    a_early = 0.1

    # Compute full and classical H at a=1
    H2_classical_today = classical_H_squared(a_today, params)
    H_classical_today = math.sqrt(max(H2_classical_today, 0.0))
    H2_full_today = H_squared_with_quantum(a_today, params)
    H_full_today = math.sqrt(max(H2_full_today, 0.0))

    ratio_H0 = 0.0
    if H_classical_today > 0.0:
        ratio_H0 = H_full_today / H_classical_today

    # Quantum fraction today relative to rho_crit0
    rho_crit0 = critical_density(params.H0)
    rho_q_today = rho_quantum(a_today, H_classical_today, params.m_g)
    rho_q_today_over_crit0 = 0.0
    if rho_crit0 > 0.0:
        rho_q_today_over_crit0 = rho_q_today / rho_crit0

    # Early‑time full H(a) at a=0.1
    H2_full_early = H_squared_with_quantum(a_early, params)
    H_at_early_a = math.sqrt(max(H2_full_early, 0.0))

    return {
        "ratio_H0": float(ratio_H0),
        "rho_q_today_over_crit0": float(rho_q_today_over_crit0),
        "H_at_early_a": float(H_at_early_a),
    }
