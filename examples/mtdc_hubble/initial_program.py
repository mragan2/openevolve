
"""
Massive Graviton Multi-Temporal Decay Cosmology (MTDC)
OpenEvolve Initial Program
"""

import math
import numpy as np
from scipy.integrate import quad

# Physical constants
c = 2.99792458e5        # km/s
c_SI = 2.99792458e8     # m/s
H0_to_SI = 1000.0 / 3.085677581e22
SI_to_H0 = 3.085677581e22 / 1000.0

lambda_g_Mpc = 4.64e9 / 3.26 / 1e6


Omega_r_default = 9e-5

# EVOLVE-BLOCK-START
def mg_refinement(Dc_Mpc, lambda_g_Mpc):
    """
    Refinement factor for the MTDC suppression.

    Default = 1.0
    Evolution may produce:
       1 + alpha (Dc / lambda_g)^n
       exp(beta Dc / lambda_g)
       logistic corrections
    etc.

    MUST return a positive scalar ~ O(1).
    """
    return 1.0
# EVOLVE-BLOCK-END


# ============================================================
# MTDC Hubble function
# ============================================================
def H_MTD_corrected(z, H0_km_s_Mpc, Omega_m):
    Omega_r = Omega_r_default
    H0_SI = H0_km_s_Mpc * H0_to_SI

    # Compute comoving distance Dc(z) in Mpc
    def integrand(zp):
        return c / H_LCDM(zp, H0_km_s_Mpc, Omega_m)

    Dc, _ = quad(integrand, 0.0, z, epsabs=1e-6, epsrel=1e-6)
    Dc_Mpc = Dc

    base = math.exp(-Dc_Mpc / lambda_g_Mpc)
    refine = mg_refinement(Dc_Mpc, lambda_g_Mpc)

    H2 = H0_SI**2 * (
        Omega_m * (1+z)**3 * base * refine +
        Omega_r * (1+z)**4
    )
    return math.sqrt(H2) * SI_to_H0  # return in km/s/Mpc


def H_LCDM(z, H0, Omega_m):
    Omega_r = Omega_r_default
    return H0 * math.sqrt(Omega_m*(1+z)**3 + Omega_r*(1+z)**4 + 1.0 - Omega_m - Omega_r)


def predict_H(z, H0, Omega_m):
    return H_MTD_corrected(z, H0, Omega_m)
