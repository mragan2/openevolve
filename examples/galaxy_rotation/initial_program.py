"""
Galaxy rotation with unit-consistent Yukawa correction and Vainshtein screening.

This is a v2 seed based on the previous best program, modified to:
  - Preserve the same radial behaviour (screening and Yukawa structure).
  - Make units explicit and consistent:
        * Yukawa term is built in SI units [m^2/s^2]
        * Then converted to [km^2/s^2] before combining with v_baryonic^2
  - Expose a small set of parameters for OpenEvolve to tune.
"""

import math

# ----------------------------------------------------------------------
# Physical constants (fixed)
# ----------------------------------------------------------------------
G = 6.67430e-11        # Gravitational constant [m^3 kg^-1 s^-2]
KPC_TO_M = 3.086e19    # 1 kpc in meters

# ----------------------------------------------------------------------
# Massive graviton / screening scales (TUNABLE)
# ----------------------------------------------------------------------
LAMBDA_G = 1.4e26          # [m] Graviton Compton wavelength (~4.6 Gly)
R_VAIN_KPC = 25.0          # [kpc] Vainshtein radius (screening scale)
SCREENING_POWER = 2.0      # Power in screening factor (controls steepness)

# ----------------------------------------------------------------------
# Coupling strength (TUNABLE)
# ----------------------------------------------------------------------
# Rescaled so that after m^2->km^2 conversion, the numerical contribution
# approximately matches the earlier best program with YUKAWA_ALPHA ≈ 0.6.
ALPHA_YUKAWA_DIMLESS = 6.0e5

# ----------------------------------------------------------------------
# Inner-region suppression (TUNABLE)
# ----------------------------------------------------------------------
# Yukawa term is strongly suppressed inside INNER_CUTOFF_KPC, so that
# baryons dominate the inner rotation curve.
INNER_CUTOFF_KPC = 5.0         # Radius inside which Yukawa is suppressed
INNER_SUPPRESSION_POWER = 2.0  # Exponent in (r / INNER_CUTOFF_KPC)^power

# Threshold for "r << lambda" regime
RATIO_THRESHOLD = 1.0e-3


def calculate_rotation_velocity(r_kpc, v_baryonic, M_enclosed):
    """
    Compute total rotation velocity (km/s) including a Yukawa + Vainshtein
    massive-gravity correction.

    Parameters
    ----------
    r_kpc : float
        Radius in kiloparsecs.
    v_baryonic : float
        Baryonic-only circular velocity in km/s.
    M_enclosed : float
        Enclosed baryonic mass within r (kg).

    Returns
    -------
    float
        Total circular velocity in km/s.
    """
    # Basic safety checks
    if r_kpc <= 0.0 or M_enclosed <= 0.0:
        return float(v_baryonic)

    # Radius in meters
    r_m = r_kpc * KPC_TO_M
    v_bary = float(v_baryonic)

    # --------------------------------------------------------------
    # 1. Vainshtein-like screening (same structure as best program)
    #    screening_factor = 1 + (r / R_VAIN)^SCREENING_POWER
    # --------------------------------------------------------------
    r_vain_m = R_VAIN_KPC * KPC_TO_M
    screening_factor = 1.0 + (r_m / r_vain_m) ** SCREENING_POWER

    # --------------------------------------------------------------
    # 2. Yukawa "core" term in SI units [m^2/s^2]
    #    Core structure:
    #      ~ G * M_enclosed / LAMBDA_G with exponential suppression
    #      for r ≳ LAMBDA_G.
    # --------------------------------------------------------------
    ratio = r_m / LAMBDA_G

    if ratio < RATIO_THRESHOLD:
        # r << lambda: exponential ~ 1, scale-independent boost
        yukawa_core_SI = G * M_enclosed / LAMBDA_G
    else:
        # r ≳ lambda: include exponential Yukawa suppression
        exponential_factor = math.exp(-ratio)
        yukawa_core_SI = G * M_enclosed / LAMBDA_G * exponential_factor

    # Full Yukawa contribution in SI:
    #   v_yukawa_sq_SI ~ α * (G M / LAMBDA_G) * screening_factor
    v_yukawa_sq_SI = ALPHA_YUKAWA_DIMLESS * yukawa_core_SI * screening_factor

    # --------------------------------------------------------------
    # 3. Convert Yukawa term to (km/s)^2
    # --------------------------------------------------------------
    # 1 km^2/s^2 = 1e6 m^2/s^2  →  divide by 1e6
    v_yukawa_sq_km2 = v_yukawa_sq_SI / 1.0e6

    # Inner suppression: Yukawa should not dominate in inner galaxy
    if r_kpc < INNER_CUTOFF_KPC:
        # Factor goes from 0 at r=0 to 1 at r=INNER_CUTOFF_KPC
        factor = (r_kpc / INNER_CUTOFF_KPC) ** INNER_SUPPRESSION_POWER
        v_yukawa_sq_km2 *= factor

    if v_yukawa_sq_km2 < 0.0:
        v_yukawa_sq_km2 = 0.0

    # --------------------------------------------------------------
    # 4. Combine baryonic and Yukawa contributions
    # --------------------------------------------------------------
    v_total_sq = v_bary ** 2 + v_yukawa_sq_km2

    if v_total_sq <= 0.0:
        # Fallback: numerical guard
        return v_bary

    return math.sqrt(v_total_sq)
