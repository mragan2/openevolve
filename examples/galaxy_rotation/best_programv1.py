"""
Simplified MOG/Yukawa galaxy rotation model with Vainshtein screening.

Core physics:
- Yukawa correction with massive graviton screening
- Vainshtein mechanism to suppress Yukawa in dense regions
- Physical suppression at small and large radii
"""

import math

# Physical constants
G = 6.67430e-11     # Gravitational constant [m^3 kg^-1 s^-2]
KPC_TO_M = 3.086e19 # 1 kpc in meters

# EVOLVE-BLOCK-START
# MOG parameters - focus on key physical scales
LAMBDA_G = 1.4e26       # Graviton Compton wavelength (~4.6 Gly)
R_VAIN_KPC = 25.0       # Vainshtein screening radius
SCREENING_POWER = 2.0   # Power in screening factor (controls steepness)
ALPHA_YUKAWA_DIMLESS = 6.3e5    # Yukawa coupling strength
INNER_CUTOFF_KPC = 5.0      # Inner suppression radius (kpc)
INNER_SUPPRESSION_POWER = 2.0   # Power for inner suppression
OUTER_CUTOFF_KPC = 100.0   # Outer damping start radius
OUTER_WIDTH_KPC = 30.0     # Width scale for damping
# EVOLVE-BLOCK-END

def calculate_rotation_velocity(r_kpc, v_baryonic, M_enclosed):
    """Calculate total rotation velocity with Yukawa + Vainshtein correction."""
    if r_kpc <= 0.0 or M_enclosed <= 0.0:
        return float(v_baryonic)
    
    r_m = r_kpc * KPC_TO_M
    v_bary = float(v_baryonic)
    
    # Vainshtein screening: suppresses Yukawa in dense regions
    r_vain_m = R_VAIN_KPC * KPC_TO_M
    screening = 1.0 + (r_m / r_vain_m) ** SCREENING_POWER
    
    # Yukawa core with exponential suppression
    ratio = r_m / LAMBDA_G
    yukawa_core = G * M_enclosed / LAMBDA_G
    # Unified treatment of exponential factor
    exponential_factor = math.exp(-ratio) if ratio > 1e-3 else 1.0
    yukawa_core *= exponential_factor
    
    # Yukawa contribution in km/s units
    v_yukawa_sq = ALPHA_YUKAWA_DIMLESS * yukawa_core * screening / 1.0e6
    
    # Inner suppression: Yukawa should be subdominant at small r
    if r_kpc < 2 * INNER_CUTOFF_KPC:
        x = r_kpc / INNER_CUTOFF_KPC
        inner_suppression = 1.0 - math.exp(-x ** INNER_SUPPRESSION_POWER)
        v_yukawa_sq *= inner_suppression
    
    # Inner suppression: Yukawa should be subdominant at small r
    if r_kpc < 2 * INNER_CUTOFF_KPC:
        x = r_kpc / INNER_CUTOFF_KPC
        inner_suppression = 1.0 - math.exp(-x ** INNER_SUPPRESSION_POWER)
        v_yukawa_sq *= inner_suppression
    
    # Outer suppression: prevent overcorrection at large radii
    if r_kpc > OUTER_CUTOFF_KPC:
        x_outer = (r_kpc - OUTER_CUTOFF_KPC) / OUTER_WIDTH_KPC
        v_yukawa_sq *= math.exp(-x_outer * x_outer)
    
    # Simplified combination with built-in non-negativity
    v_total_sq = v_bary ** 2 + max(0.0, v_yukawa_sq)
    return math.sqrt(v_total_sq)
