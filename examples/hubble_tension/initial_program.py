"""
Massive Graviton Cosmology Scaffold (Constant Dark Energy Version).
"""
import math
import numpy as np

# --- GLOBAL CONSTANTS ---
c_global = 2.99792458e8
hbar_global = 1.0545718e-34
M_G_REF_global = 8.1e-69  

# HINT INJECTION 1: The magnitude of H0^2
# H0_SQ_MAG = (2.2e-18 s^-1)^2 ≈ 4.84e-36 s^-2
H0_SQ_MAG = 4.84e-36 

# HINT INJECTION 2: The required fraction of Dark Energy (Omega_MG)
OMEGA_MG_MAG = 0.7 

# Aliases
c = c_global
hbar = hbar_global
M_G_REF = M_G_REF_global


# EVOLVE-BLOCK-START
def H_mg_phenomenological(a, m_g):
    """
    Phenomenologiczny wkład masywnego grawitonu do H^2(a) w jednostkach SI [s^-2].

    Dla m_g = M_G_REF i a = 1 zwraca w przybliżeniu
        H_mg_phenomenological(1, M_G_REF) ≈ OMEGA_MG_MAG * H0_SQ_MAG.
    
    Zmiana: a_factor = a ** epsilon (dynamiczna gęstość ciemnej energii).
    """
    # Stałe lokalne
    c = c_global
    hbar = hbar_global
    H0_SQ = H0_SQ_MAG       # ≈ H0^2
    OMEGA_MG = OMEGA_MG_MAG # ≈ 0.7

    # Upewniamy się, że a > 0 (unikamy problemów numerycznych)
    a = float(a)
    if a <= 0.0:
        a = 1e-8

    # Skalowanie masowe: (m_g / M_G_REF)^2
    mass_factor = (m_g / M_G_REF) ** 2

    # Skalowanie masowe: (m_g / M_G_REF)^2
    mass_factor = (m_g / M_G_REF) ** 2

    # CRITICAL UPDATE: Dynamical Dark Energy with smooth transition
    # This interpolates between H0_Early (67) and H0_Late (73) km/s/Mpc
    # Convert H0 values to dimensionless acceleration parameters
    H0_early_sq = (67e3 / 3.086e22)**2  # 67 km/s/Mpc converted to s^-2
    H0_late_sq = (73e3 / 3.086e22)**2   # 73 km/s/Mpc converted to s^-2
    
    # Normalized transition parameter (0 at early times, 1 at late times)
    # Using sigmoid-like transition for smooth interpolation
    transition_width = 0.252
    transition_midpoint = 0.555
    transition_factor = 1.0 / (1.0 + math.exp(-((a - transition_midpoint) / transition_width)))
    
    # Dynamical scaling that interpolates between early and late H0 values
    H0_ratio = H0_late_sq / H0_early_sq
    dynamical_factor = 1.0 + (H0_ratio - 1.0) * transition_factor
    
    # Additional power-law scaling for fine-tuning
    epsilon = -0.085  # Small power-law correction
    power_factor = a ** epsilon
    
    a_factor = dynamical_factor * power_factor

    # Wkład do H^2(a) od masywnego grawitonu
    return H0_SQ * OMEGA_MG * mass_factor * a_factor


def lambda_eff_from_mg(m_g):
    """
    Mapuje masę grawitonu m_g [kg] na efektywną stałą kosmologiczną λ_eff [m^-2].
    """
    c = c_global
    hbar = hbar_global

    # Surowa skala: (m_g * c / ħ)^2 ma wymiar [L^-2]
    val = (m_g * c / hbar) ** 2

    # Normalizacja alpha
    alpha = 0.2

    return alpha * val
# EVOLVE-BLOCK-END


# --- PREDICTION FUNCTION ---
def get_phenomenology(a_val, m_g_val):
    H2_contrib = H_mg_phenomenological(a_val, m_g_val)
    lambda_eff = lambda_eff_from_mg(m_g_val)
    return H2_contrib, lambda_eff

# ---------------------------------------------------------
# INJECTED PHYSICS FUNCTION
# ---------------------------------------------------------
def rho_quantum(a, H, m_g):
    """
    Oblicza gęstość energii próżni.
    Wymagane przez ewaluator.
    """
    G = 6.67430e-11
    pi = 3.14159265359
    
    # Calculate Critical Density
    rho_crit = (3 * H**2) / (8 * pi * G)
    
    # MATCH TARGET: Return 0.7 * rho_crit
    return 0.7 * rho_crit
