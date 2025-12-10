"""
Massive Graviton Cosmology: Background-only MCMC (SN + BAO + CMB).

- Uses your H_mg_phenomenological(a, m_g) (dynamic H0 version).
- Parameter vector: [H0_km_s_Mpc, Omega_m, f_mg, M_abs]
  with m_g = f_mg * M_G_REF_global.
- Data:
  * SN: loads Pantheon+-style file if present (handles string IDs).
  * BAO: realistic 6dFGS + SDSS MGS + BOSS LOWZ/CMASS DV(z)/r_d points.
  * CMB: Planck-like distance prior using the shift parameter R.
"""

import os
import math
import numpy as np
from scipy.integrate import quad
import emcee

# ------------------------------------------------------------
# 0. Massive graviton H_mg_phenomenological
# ------------------------------------------------------------

# --- GLOBAL CONSTANTS ---
c_global = 2.99792458e8
hbar_global = 1.0545718e-34
M_G_REF_global = 8.1e-69  # reference graviton mass [kg]

# H0^2 magnitude (from scaffold)
H0_SQ_MAG = 4.84e-36      # ~ (2.2e-18 s^-1)^2
OMEGA_MG_MAG = 0.7        # target DE fraction from MG

# Aliases
c = c_global
hbar = hbar_global
M_G_REF = M_G_REF_global


def H_mg_phenomenological(a, m_g):
    """
    Phenomenological massive graviton contribution to H^2(a) in SI units [s^-2].
    """
    H0_SQ = H0_SQ_MAG
    OMEGA_MG = OMEGA_MG_MAG

    # ensure a > 0
    a = float(a)
    if a <= 0.0:
        a = 1e-8

    # mass scaling
    mass_factor = (m_g / M_G_REF) ** 2

    # early/late H0 values in s^-2
    H0_early_sq = (67e3 / 3.086e22) ** 2
    H0_late_sq = (73e3 / 3.086e22) ** 2

    # smooth transition
    transition_width = 0.252
    transition_midpoint = 0.555
    transition_factor = 1.0 / (1.0 + math.exp(-((a - transition_midpoint) / transition_width)))

    H0_ratio = H0_late_sq / H0_early_sq
    dynamical_factor = 1.0 + (H0_ratio - 1.0) * transition_factor

    # small power-law correction
    epsilon = -0.085
    power_factor = a ** epsilon

    a_factor = dynamical_factor * power_factor

    return H0_SQ * OMEGA_MG * mass_factor * a_factor  # [s^-2]


# ------------------------------------------------------------
# 1. Paths and data loading
# ------------------------------------------------------------

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_DIR, "data")

c_km_s = 2.99792458e5  # speed of light [km/s]


def load_pantheon_style(path):
    """
    Load Pantheon+SH0ES distance file. 
    Robustly handles string columns (SN names) by using genfromtxt with headers.
    """
    try:
        # names=True reads the header row and allows column access by name
        # dtype=None automatically handles mixed types (strings vs floats)
        data = np.genfromtxt(path, names=True, dtype=None, encoding=None)
    except Exception as e:
        print(f"Error reading file structure: {e}")
        # Fallback for simple space-separated files with no string columns
        arr = np.loadtxt(path, skiprows=1)
        return {"z": arr[:, 0], "mu": arr[:, 1], "sigma_mu": arr[:, 2]}

    # Helper to find the correct column name from common variations
    def get_col(candidates):
        for c in candidates:
            if c in data.dtype.names:
                return data[c]
        # If we can't find standard headers, list available ones for debugging
        raise ValueError(f"Could not find a column matching {candidates}. Found headers: {data.dtype.names}")

    # Extract columns (Pantheon+ standard: zHD, MU_SH0ES, MU_SH0ES_ERR_DIAG)
    z = get_col(['zHD', 'z', 'z_cmb'])
    mu = get_col(['MU_SH0ES', 'mu', 'MU'])
    sigma_mu = get_col(['MU_SH0ES_ERR_DIAG', 'sigma_mu', 'dmu', 'err'])

    return {"z": z, "mu": mu, "sigma_mu": sigma_mu}


PANTHEON_PATH = os.path.join(DATA_DIR, "PantheonSH0ES.dat")

if os.path.exists(PANTHEON_PATH):
    print(f"Loading SN data from {PANTHEON_PATH}...")
    SN_DATA = load_pantheon_style(PANTHEON_PATH)
else:
    print("SN file not found, using mock data...")
    # Mock sample
    SN_DATA = {
        "z": np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.00]),
        "mu": np.array([
            33.257, 36.816, 38.395, 40.034,
            41.031, 42.332, 43.568, 44.163
        ]),
        "sigma_mu": np.array([0.15, 0.15, 0.15, 0.17, 0.18, 0.20, 0.22, 0.25]),
    }

# ------------------------------------------------------------
# 1b. BAO Data
# ------------------------------------------------------------

BAO_DATA = {
    "z": np.array([0.106, 0.15, 0.32, 0.57]),
    "DV_over_rd": np.array([3.11, 4.51, 8.59, 13.98]),
    "sigma": np.array([0.18, 0.17, 0.17, 0.14]),
    "rd": 147.09,
}

# ------------------------------------------------------------
# 1c. CMB Prior
# ------------------------------------------------------------

CMB_PRIOR = {
    "R_obs": 1.7502,
    "sigma_R": 0.0046,
    "z_star": 1089.92,
}


# ------------------------------------------------------------
# 2. Cosmology helpers
# ------------------------------------------------------------

def H_LCDM_squared(a, H0_si, Omega_m, Omega_r, Omega_L):
    """Flat LCDM H^2(a) in SI units [s^-2]."""
    # Guard against a=0 division
    if a <= 1e-9: a = 1e-9
    return H0_si**2 * (Omega_r / a**4 + Omega_m / a**3 + Omega_L)


def H_total_of_z(z, theta):
    """
    Total H(z) in km/s/Mpc for parameter vector:
      theta = [H0_km_s_Mpc, Omega_m, f_mg]
    """
    if np.isscalar(theta) or len(theta) != 3:
        raise ValueError(f"H_total_of_z expects 3 params, got: {theta}")

    H0_km_s_Mpc, Omega_m, f_mg = theta

    Omega_r = 9e-5
    Omega_L = 1.0 - Omega_m - Omega_r

    H0_si = H0_km_s_Mpc * 1000.0 / 3.085677581e22
    a = 1.0 / (1.0 + z)
    m_g = f_mg * M_G_REF

    H2_LCDM = H_LCDM_squared(a, H0_si, Omega_m, Omega_r, Omega_L)
    H2_mg = H_mg_phenomenological(a, m_g)

    H2_total_si = H2_LCDM + H2_mg

    if H2_total_si <= 0.0:
        return 1e-10

    H_si = math.sqrt(H2_total_si)
    return H_si * (3.085677581e22 / 1000.0)


def comoving_distance_Mpc(z, theta):
    def integrate_one(zi):
        integrand = lambda zp: c_km_s / H_total_of_z(zp, theta)
        chi, _ = quad(integrand, 0.0, zi, epsabs=1e-5, epsrel=1e-5)
        return chi

    if np.ndim(z) > 0:
        return np.array([integrate_one(zi) for zi in z])
    else:
        return integrate_one(z)


def luminosity_distance_Mpc(z, theta):
    chi = comoving_distance_Mpc(z, theta)
    return (1.0 + z) * chi


def angular_diameter_distance_Mpc(z, theta):
    chi = comoving_distance_Mpc(z, theta)
    return chi / (1.0 + z)


def DV_Mpc(z, theta):
    DA = angular_diameter_distance_Mpc(z, theta)
    Hz = H_total_of_z(z, theta)
    return ((1 + z)**2 * DA**2 * (c_km_s * z / Hz))**(1.0 / 3.0)


def distance_modulus(z, theta, M_abs):
    DL = luminosity_distance_Mpc(z, theta)
    DL = np.where(DL <= 0, 1e-9, DL)
    return 5.0 * np.log10(DL * 1e5) + M_abs


# ------------------------------------------------------------
# 3. Likelihoods
# ------------------------------------------------------------

def log_likelihood_SN(theta, M_abs):
    z = SN_DATA["z"]
    mu_obs = SN_DATA["mu"]
    sigma_mu = SN_DATA["sigma_mu"]

    mu_th = distance_modulus(z, theta, M_abs)
    chi2 = np.sum(((mu_obs - mu_th) / sigma_mu) ** 2)
    return -0.5 * chi2


def log_likelihood_BAO(theta):
    z = BAO_DATA["z"]
    DV_over_rd_obs = BAO_DATA["DV_over_rd"]
    sigma = BAO_DATA["sigma"]
    rd = BAO_DATA["rd"]

    DV_th = np.array([DV_Mpc(zi, theta) for zi in z])
    DV_over_rd_th = DV_th / rd

    chi2 = np.sum(((DV_over_rd_obs - DV_over_rd_th) / sigma) ** 2)
    return -0.5 * chi2


def log_likelihood_CMB(theta):
    H0_km_s_Mpc, Omega_m, f_mg = theta
    z_star = CMB_PRIOR["z_star"]
    
    # Comoving distance to last scattering
    DM_star = comoving_distance_Mpc(z_star, theta)

    # Convert H0 to SI for R calculation
    H0_si = H0_km_s_Mpc * 1000.0 / 3.085677581e22
    c_si = 2.99792458e8
    DM_star_si = DM_star * 3.085677581e22

    # Shift parameter R
    R_th = math.sqrt(Omega_m) * H0_si * DM_star_si / c_si

    R_obs = CMB_PRIOR["R_obs"]
    sigma_R = CMB_PRIOR["sigma_R"]

    chi2 = ((R_th - R_obs) / sigma_R) ** 2
    return -0.5 * chi2


def log_likelihood_total(params):
    # params = [H0, Omega_m, f_mg, M_abs]
    if len(params) != 4:
        return -np.inf

    theta = params[:-1]
    M_abs = params[-1]

    if np.any(np.isnan(params)):
        return -np.inf

    # Check for physics violations before costly integrals
    if theta[0] <= 0 or theta[1] <= 0:
        return -np.inf

    ll_sn = log_likelihood_SN(theta, M_abs)
    ll_bao = log_likelihood_BAO(theta)
    ll_cmb = log_likelihood_CMB(theta)

    return ll_sn + ll_bao + ll_cmb


# ------------------------------------------------------------
# 4. Priors + posterior
# ------------------------------------------------------------

def log_prior(params):
    H0, Om, f_mg, M_abs = params

    if not (50.0 < H0 < 90.0): return -np.inf
    if not (0.1 < Om < 0.6): return -np.inf
    if not (0.0 < f_mg < 20.0): return -np.inf
    if not (-20.5 < M_abs < -18.0): return -np.inf

    return 0.0


def log_posterior(params):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    
    return lp + log_likelihood_total(params)


# ------------------------------------------------------------
# 5. Run MCMC
# ------------------------------------------------------------

def run_mcmc(n_walkers=24, n_steps=200):
    print("--- Starting MCMC ---")
    ndim = 4

    initial = np.array([70.0, 0.3, 1.0, -19.3])
    
    # Initialize walkers in a small ball
    p0 = initial + 1e-2 * initial * np.random.randn(n_walkers, ndim)

    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior)

    print(f"Running {n_steps} steps with {n_walkers} walkers...")
    sampler.run_mcmc(p0, n_steps, progress=True)

    return sampler


if __name__ == "__main__":
    # Settings
    WALKERS = 32
    STEPS = 1000
    DISCARD = 200

    sampler = run_mcmc(n_walkers=WALKERS, n_steps=STEPS)

    flat = sampler.get_chain(discard=DISCARD, thin=5, flat=True)
    mean = np.mean(flat, axis=0)
    std = np.std(flat, axis=0)

    print("\n--- Results (post-burn-in) ---")
    labels = ["H0", "Omega_m", "f_mg", "M_abs"]
    for i, lbl in enumerate(labels):
        print(f"{lbl}: {mean[i]:.4f} +/- {std[i]:.4f}")

    print("\nAcceptance fraction (avg):", np.mean(sampler.acceptance_fraction))
