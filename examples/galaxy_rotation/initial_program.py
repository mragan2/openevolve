"""
Galaxy rotation with unit-consistent Yukawa correction and Vainshtein screening.

v3 seed: cleaned version of the best evolved model.

- Preserves radial behaviour of the 0.797-score solution.
- Makes units explicit and consistent.
- Encodes inner and outer suppression as single, clear factors
  instead of duplicated code, while keeping the effective power.
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
# Tuned by OpenEvolve; this value matches the best evolved model.
ALPHA_YUKAWA_DIMLESS = 6.3e5  # Coupling strength

# ----------------------------------------------------------------------
# Inner-region suppression (TUNABLE)
# ----------------------------------------------------------------------
# In the best model, inner suppression was effectively applied twice with
# INNER_SUPPRESSION_POWER = 2.0 -> net exponent 4.0.
# Here we encode that directly as INNER_SUPPRESSION_POWER = 4.0 and apply
# the factor only once.
INNER_CUTOFF_KPC = 5.0          # Radius inside which Yukawa is suppressed
INNER_SUPPRESSION_POWER = 4.0   # Effective exponent in (r / INNER_CUTOFF_KPC)^power

# ----------------------------------------------------------------------
# Outer-region suppression (TUNABLE)
# ----------------------------------------------------------------------
OUTER_CUTOFF_KPC = 100.0        # Radius where outer damping starts to matter
OUTER_WIDTH_KPC = 30.0          # Width scale for Gaussian-like damping

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
    # 1. Vainshtein-like screening
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

    # --------------------------------------------------------------
    # 4. Inner suppression: Yukawa should not dominate in inner galaxy
    # --------------------------------------------------------------
    if r_kpc < INNER_CUTOFF_KPC:
        # Factor goes from 0 at r=0 to 1 at r=INNER_CUTOFF_KPC
        factor_inner = (r_kpc / INNER_CUTOFF_KPC) ** INNER_SUPPRESSION_POWER
        v_yukawa_sq_km2 *= factor_inner

    # --------------------------------------------------------------
    # 5. Outer suppression: avoid overcorrection at very large radii
    # --------------------------------------------------------------
    if r_kpc > OUTER_CUTOFF_KPC:
        x = (r_kpc - OUTER_CUTOFF_KPC) / OUTER_WIDTH_KPC
        outer_factor = math.exp(-x * x)
        v_yukawa_sq_km2 *= outer_factor

    if v_yukawa_sq_km2 < 0.0:
        v_yukawa_sq_km2 = 0.0

    # --------------------------------------------------------------
    # 6. Combine baryonic and Yukawa contributions
    # --------------------------------------------------------------
    v_total_sq = v_bary ** 2 + v_yukawa_sq_km2

    if v_total_sq <= 0.0:
        # Fallback: numerical guard
        return v_bary

    return math.sqrt(v_total_sq)
