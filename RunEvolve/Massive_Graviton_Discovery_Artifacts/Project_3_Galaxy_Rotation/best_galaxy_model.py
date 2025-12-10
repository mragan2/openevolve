"""
MOG/Yukawa galaxy rotation model with Vainshtein screening (best fit).
"""

import math

# Physical constants
G = 6.67430e-11      # Gravitational constant [m^3 kg^-1 s^-2]
KPC_TO_M = 3.086e19  # 1 kpc in meters

LAMBDA_G = 1.4e26          # Graviton Compton wavelength (~4.6 Gly) [m]
R_VAIN_KPC = 25.0          # Vainshtein screening radius [kpc]
SCREENING_POWER = 2.0      # Power in screening factor

ALPHA_YUKAWA_DIMLESS = 6.4e5   # Yukawa coupling strength

INNER_CUTOFF_KPC = 5.0         # Inner suppression radius [kpc]
INNER_SUPPRESSION_POWER = 2.0  # Power for inner suppression

OUTER_CUTOFF_KPC = 100.0       # Outer damping start radius [kpc]
OUTER_WIDTH_KPC = 30.0         # Width scale for outer damping [kpc]


def calculate_rotation_velocity(r_kpc, v_baryonic, M_enclosed):
    """
    Calculate total rotation velocity including Yukawa + Vainshtein correction.
    """
    if r_kpc <= 0.0 or M_enclosed <= 0.0:
        return float(v_baryonic)

    r_m = r_kpc * KPC_TO_M
    v_bary = float(v_baryonic)

    # 1. Vainshtein screening
    r_vain_m = R_VAIN_KPC * KPC_TO_M
    screening = 1.0 + (r_m / r_vain_m) ** SCREENING_POWER

    # 2. Yukawa core with exponential suppression
    ratio = r_m / LAMBDA_G
    yukawa_core = G * M_enclosed / LAMBDA_G

    if ratio > 20.0:
        exponential_factor = 0.0
    elif ratio > 1e-3:
        exponential_factor = math.exp(-ratio)
    else:
        exponential_factor = 1.0 - ratio + 0.5 * ratio * ratio
    yukawa_core *= exponential_factor

    # Yukawa contribution in (km/s)^2
    v_yukawa_sq = ALPHA_YUKAWA_DIMLESS * yukawa_core * screening / 1.0e6

    # 3. Inner suppression
    if r_kpc < 2.0 * INNER_CUTOFF_KPC:
        x = r_kpc / INNER_CUTOFF_KPC
        inner_suppression = x ** INNER_SUPPRESSION_POWER / (1.0 + x ** INNER_SUPPRESSION_POWER)
        v_yukawa_sq *= inner_suppression

    # 4. Outer suppression
    if r_kpc > OUTER_CUTOFF_KPC:
        x_outer = (r_kpc - OUTER_CUTOFF_KPC) / OUTER_WIDTH_KPC
        outer_suppression = math.exp(-x_outer * x_outer)
        if r_kpc > 1.5 * OUTER_CUTOFF_KPC:
            extra_suppression = math.exp(-(r_kpc - 1.5 * OUTER_CUTOFF_KPC) / OUTER_WIDTH_KPC)
            outer_suppression *= extra_suppression
        v_yukawa_sq *= outer_suppression

    v_total_sq = v_bary ** 2 + max(0.0, v_yukawa_sq)
    if v_total_sq <= 0.0:
        return v_bary

    return math.sqrt(v_total_sq)
