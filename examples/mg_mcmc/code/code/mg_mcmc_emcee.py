"""
MCMC for Multi-Temporal Decay Cosmology (MTDC)

- Parameter vector: [H0_km_s_Mpc, Omega_m]
- Replaces standard Lambda with an exponential suppression of matter density
  based on comoving distance.
- M_abs is FIXED to SH0ES: -19.253
"""

import numpy as np
from scipy.integrate import quad
import emcee
import math
import os

# ------------------------------------------------------------
# 0. GLOBAL CONSTANTS
# ------------------------------------------------------------

c_global = 2.99792458e8
c_km_s = 2.99792458e5

# Conversion: 1 Mpc ~ 3.26156 million light years
# Lambda_g = 4.64 billion light years
LAMBDA_G_LY = 4.64e9
LAMBDA_G_MPC = LAMBDA_G_LY / 3.26156e6  # ~1422 Mpc

M_ABS_FIXED = -19.253   # SH0ES absolute magnitude

# ------------------------------------------------------------
# 1. DATA: SN / BAO / CMB
# ------------------------------------------------------------

def load_pantheon_plus(path):
    arr = np.loadtxt(path, usecols=(0, 1, 2))
    return {"z": arr[:,0], "mu": arr[:,1], "sigma_mu": arr[:,2]}

PANTHEON_PATH = os.path.join("data", "pantheon_plus_sn.txt")
if os.path.exists(PANTHEON_PATH):
    SN_DATA = load_pantheon_plus(PANTHEON_PATH)
else:
    print("Warning: Pantheon data not found. Using mock data.")
    SN_DATA = {
        "z": np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]),
        "mu": np.array([33.2, 36.3, 37.8, 39.5, 40.6, 42.2, 43.3, 44.0]),
        "sigma_mu": np.array([0.15, 0.15, 0.15, 0.17, 0.18, 0.2, 0.22, 0.25]),
    }

BAO_DATA = {
    "z": np.array([0.38, 0.51, 0.61]),
    "DV_over_rd": np.array([9.89, 12.86, 14.51]),
    "sigma": np.array([0.15, 0.18, 0.21]),
    "rd": 147.09,
}

CMB_PRIOR = {
    "R_obs": 1.7502,
    "sigma_R": 0.0046,
    "z_star": 1089.92,
}

# ------------------------------------------------------------
# 2. MTDC MODEL H(z)
# ------------------------------------------------------------

def H_total_of_z(z, theta):
    """
    MTDC Modified Hubble function.
    theta = [H0_km_s_Mpc, Omega_m]
    """
    H0_km_s_Mpc, Omega_m = theta
    Omega_r = 9e-5
    
    # H0 in SI
    H0_si = H0_km_s_Mpc * 1000.0 / 3.085677581e22

    # --- Compute 'Reference' Comoving Distance for Suppression ---
    # We define the decay based on the distance in a Matter+Radiation universe 
    # (or the 'unsuppressed' metric).
    def E_inv_reference(zp):
        # Kernel: 1 / sqrt( Omega_m(1+z)^3 + Omega_r(1+z)^4 )
        # No Omega_L here.
        return 1.0 / math.sqrt(Omega_m * (1.0 + zp)**3 + Omega_r * (1.0 + zp)**4)

    # Calculate dimensionless distance Dc (units of Hubble radius)
    # z must be scalar here.
    Dc_dimless, _ = quad(E_inv_reference, 0.0, z, epsabs=1e-6, epsrel=1e-6)
    
    # Convert to physical Mpc
    Dc_phys_Mpc = (c_km_s / H0_km_s_Mpc) * Dc_dimless

    # --- MTDC Modification ---
    # Exponential suppression of the matter term
    suppression = math.exp(-Dc_phys_Mpc / LAMBDA_G_MPC)

    # Total H^2 (SI units)
    # Note: No explicit Omega_L. The suppression creates the acceleration effect.
    H2_si = H0_si**2 * (
        Omega_m * (1.0 + z)**3 * suppression +
        Omega_r * (1.0 + z)**4
    )
    
    if H2_si <= 0: return 1e-10

    H_si = math.sqrt(H2_si)
    
    # Convert back to km/s/Mpc
    return H_si * (3.085677581e22 / 1000.0)

# ------------------------------------------------------------
# 3. OBSERVABLE HELPERS
# ------------------------------------------------------------

def comoving_distance_Mpc(z, theta):
    """
    Computes comoving distance from the *resulting* H_total_of_z.
    Note: This involves a nested integral (quad inside quad).
    """
    integrand = lambda zp: c_km_s / H_total_of_z(zp, theta)
    
    if np.ndim(z) > 0:
        results = np.zeros_like(z, dtype=float)
        for i, val in enumerate(z):
            res, _ = quad(integrand, 0.0, val, epsabs=1e-5, epsrel=1e-5)
            results[i] = res
        return results
    else:
        chi, _ = quad(integrand, 0.0, z, epsabs=1e-5, epsrel=1e-5)
        return chi

def luminosity_distance_Mpc(z, theta):
    return (1.0 + z) * comoving_distance_Mpc(z, theta)

def angular_diameter_distance_Mpc(z, theta):
    dist = comoving_distance_Mpc(z, theta)
    return dist / (1.0 + z)

def DV_Mpc(z, theta):
    DA = angular_diameter_distance_Mpc(z, theta)
    Hz = H_total_of_z(z, theta)
    return ((1+z)**2 * DA**2 * (c_km_s*z/Hz))**(1/3)

def distance_modulus(z, theta, M_abs):
    DL = luminosity_distance_Mpc(z, theta)
    if np.ndim(DL) > 0:
        DL = np.maximum(DL, 1e-5)
    else:
        DL = max(DL, 1e-5)
    return 5 * np.log10(DL * 1e5) + M_abs

# ------------------------------------------------------------
# 4. LIKELIHOODS
# ------------------------------------------------------------

def log_likelihood_SN(theta):
    # theta = [H0, Omega_m]
    mu_th = distance_modulus(SN_DATA["z"], theta, M_ABS_FIXED)
    chi2 = np.sum(((SN_DATA["mu"] - mu_th) / SN_DATA["sigma_mu"])**2)
    return -0.5 * chi2

def log_likelihood_BAO(theta):
    DV_th = np.array([DV_Mpc(zi, theta) for zi in BAO_DATA["z"]])
    DV_over_rd_th = DV_th / BAO_DATA["rd"]
    chi2 = np.sum(((BAO_DATA["DV_over_rd"] - DV_over_rd_th) / BAO_DATA["sigma"])**2)
    return -0.5 * chi2

def log_likelihood_CMB(theta):
    H0_km_s_Mpc, Omega_m = theta
    z_star = CMB_PRIOR["z_star"]

    DA_star = angular_diameter_distance_Mpc(z_star, theta)
    
    # Calculate R shift parameter
    H0_si = H0_km_s_Mpc * 1000.0 / 3.085677581e22
    c_si = 2.99792458e8
    
    # R = sqrt(Omega_m) * H0 * DA / c
    R_th = np.sqrt(Omega_m) * H0_si * (DA_star * 3.085677581e22) / c_si
    
    chi2 = ((R_th - CMB_PRIOR["R_obs"]) / CMB_PRIOR["sigma_R"])**2
    return -0.5 * chi2

def log_likelihood_total(params):
    lp = log_likelihood_SN(params)
    lp += log_likelihood_BAO(params)
    lp += log_likelihood_CMB(params)
    return lp

# ------------------------------------------------------------
# 5. PRIORS + POSTERIOR
# ------------------------------------------------------------

def log_prior(params):
    H0, Om = params

    if not (50.0 < H0 < 100.0): return -np.inf
    if not (0.1 < Om < 1.0):   return -np.inf # Allow higher Om since no Omega_L
    return 0.0

def log_posterior(params):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_total(params)

# ------------------------------------------------------------
# 6. MCMC RUNNER
# ------------------------------------------------------------

def run_mcmc(n_walkers=20, n_steps=600):
    ndim = 2  # [H0, Omega_m]

    # Initial guess around typical values
    # Note: Without Lambda, Omega_m might need to be different, 
    # but the suppression mimics Lambda, so start near LCDM values.
    initial = np.array([72.0, 0.3]) 
    
    p0 = initial + 1e-2 * initial * np.random.randn(n_walkers, ndim)
    
    # Validation loop for priors
    for i in range(n_walkers):
        while not np.isfinite(log_prior(p0[i])):
             p0[i] = initial + 2e-2 * initial * np.random.randn(ndim)

    print(f"Starting MTDC MCMC (Nested Integral Model) | {n_walkers} walkers | {n_steps} steps")
    print(f"Fixed Lambda_g: {LAMBDA_G_MPC:.1f} Mpc")
    
    # NOTE: Multiprocessing recommended for production due to nested quad, 
    # but kept serial here for simple script stability.
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior)
    sampler.run_mcmc(p0, n_steps, progress=True)
    return sampler

if __name__ == "__main__":
    sampler = run_mcmc()

    burn = 150
    flat = sampler.get_chain(discard=burn, thin=5, flat=True)

    mean = np.mean(flat, axis=0)
    std  = np.std(flat, axis=0)

    print("\n--- RESULTS (MTDC) ---")
    print(f"H0:      {mean[0]:.4f} +/- {std[0]:.4f}")
    print(f"Omega_m: {mean[1]:.4f} +/- {std[1]:.4f}")
 
''