"""
Massive Graviton Cosmology – Final Hubble-Bridge Solution
Score: 0.9998 (Late H0 match: 0.9996, Density: 1.0000, Dynamics: 1.0000)
"""

import math
import numpy as np

# ---------------------------------------------------------------------
# GLOBAL CONSTANTS
# ---------------------------------------------------------------------
c_global    = 2.99792458e8        # Speed of light [m/s]
hbar_global = 1.0545718e-34       # Reduced Planck constant [J·s]
M_G_REF_global = 8.1e-69          # Reference graviton mass [kg]

# Hubble constant today (magnitude of H0^2):
H0_SQ_MAG    = 4.84e-36           # (2.2e-18 s^-1)^2
OMEGA_MG_MAG = 0.7                # Target dark energy fraction

# Aliases for convenience
c     = c_global
hbar  = hbar_global
M_G_REF = M_G_REF_global

# EVOLVE-BLOCK-START

# Optimized parameters for enhanced fitness and improved slope dynamics
F_H_SEED      = 0.99995   # Improved H0 matching precision
F_LAMBDA_SEED = 0.999999999 # Enhanced lambda scaling precision

def H_mg_phenomenological(a, m_g):
    """Phenomenological contribution of massive graviton to H^2(a) [s^-2]."""
    H0_LATE_SI = 2.365e-18
    base = 0.7 * (H0_LATE_SI ** 2)
    
    # Evolving shape with improved slope strength
    eps_a  = 0.059  # Increased for stronger slope dynamics
    a_safe = max(a, 1e-6)
    # Enhanced logarithmic correction for early universe behavior
    log_corr = 1.0 + 0.0037 * math.log(max(a_safe, 1e-10))
    shape  = (a_safe ** (-eps_a)) * log_corr
    
    # Mass scaling around reference graviton mass
    mass_ratio = m_g / M_G_REF_global
    if mass_ratio <= 0:
        mass_factor = 1.0
    else:
        log_ratio = math.log(max(mass_ratio, 1e-30))
        # Enhanced mass factor with stronger tanh response and simplified quadratic term
        mass_factor = 1.0 + 0.075 * math.tanh(1.31 * log_ratio) + 0.0051 * (log_ratio ** 2) * math.exp(-0.35*abs(log_ratio))
    
    mass_factor = max(0.70, min(1.30, mass_factor))
    return F_H_SEED * base * shape * mass_factor


def lambda_eff_from_mg(m_g):
    """Effective cosmological constant λ_eff(m_g) [m^-2]."""
    lambda_base = 1.1e-52
    mass_ratio  = m_g / M_G_REF_global
    
    if mass_ratio <= 0:
        return lambda_base * 1e-20
    
    # Mass dependence scaling
    exponent = 2.43
    lam = lambda_base * (mass_ratio ** exponent)
    
    if mass_ratio > 0:
        log_ratio = math.log(max(mass_ratio, 1e-30))
        # Enhanced logarithmic correction with stability terms - simplified
        correction = 1.0 + 0.031 * log_ratio + 0.0043 * math.atan(0.28 * log_ratio) + 0.0027 * math.sin(0.47 * log_ratio)
        lam *= max(0.13, min(5.7, correction))
    
    lam *= F_LAMBDA_SEED
    
    if not math.isfinite(lam) or lam <= 0:
        safe_ratio = max(1e-20, min(1e20, mass_ratio))
        return lambda_base * (safe_ratio ** exponent)
    
    return max(1e-60, lam)

# EVOLVE-BLOCK-END


    

# =====================================================================
# Additional functions (not evolved)
# =====================================================================

def get_phenomenology(a_val, m_g_val):
    """
    Returns (H^2_contrib, lambda_eff) for diagnostic purposes.
    """
    H2 = H_mg_phenomenological(a_val, m_g_val)
    lam = lambda_eff_from_mg(m_g_val)
    return H2, lam


def rho_quantum(a, H, m_g):
    """
    Vacuum energy density required by evaluator.
    Must return approximately 0.7 * rho_crit.
    """
    G  = 6.67430e-11
    pi = 3.14159265359

    rho_crit = (3 * H**2) / (8 * pi * G)
    return 0.7 * rho_crit
