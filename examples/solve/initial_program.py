"""
Massive Graviton Cosmology Scaffold (Honest Version).
"""
import math
import numpy as np

# --- GLOBAL CONSTANTS ---
c_global = 2.99792458e8
hbar_global = 1.0545718e-34

# CRITICAL UPDATE: 8.1e-69 kg corresponds to lambda_g = 4.6 GLY
M_G_REF_global = 8.1e-69  

# EVOLVE-BLOCK-START
def H_mg_phenomenological(a, m_g, H0):
    """
    Calculates the graviton-induced contribution to H^2(a).
    Must return a value with units [T^-2].
    """
    # LOCAL CONSTANTS
    c = c_global
    hbar = hbar_global
    
    # AI TASK: Find a stable modification to expansion history
    # Hint: Simply adding H0^2 is too unstable. 
    # Try scaling by a small dimensionless factor or coupling constant.
    return H0**2 * 0.0 # Placeholder: Currently does nothing

def lambda_eff_from_mg(m_g):
    """
    Maps graviton mass m_g (kg) to effective cosmological constant (m^-2).
    STRICT MODE: Must return physical calculation.
    """
    # LOCAL CONSTANTS
    c = c_global
    hbar = hbar_global
    
    # PHYSICS: The cosmological constant Lambda is related to the inverse 
    # square of the Compton wavelength.
    # Lambda ~ (1/lambda_g)^2 = (m_g * c / hbar)^2
    
    val = (m_g * c / hbar)**2
    
    # We allow the AI to tune a dimensionless pre-factor (coupling constant 'alpha')
    # but it MUST be proportional to val.
    alpha = 1.0 # AI can evolve this number
    
    return alpha * val
# EVOLVE-BLOCK-END

# --- PREDICTION FUNCTION ---
def get_phenomenology(a_val, m_g_val):
    H0_SI = 2.2e-18
    H2_contrib = H_mg_phenomenological(a_val, m_g_val, H0_SI)
    lambda_eff = lambda_eff_from_mg(m_g_val)
    return H2_contrib, lambda_eff