"""
Massive Graviton Cosmology: Background-only MCMC (SN + BAO + CMB)

Step 1: Fix the MG dark-energy shape parameters to the final evolved values:

    transition_midpoint = 0.597
    transition_width    = 0.252
    H0_early            = 67.0
    H0_late             = 73.0
    epsilon             = -0.047

Sample ONLY:
    [H0_km_s_Mpc, Omega_m, f_mg, M_abs]

This isolates what SN+BAO+CMB say about cosmology given your final MG model.
"""

import os
import math
import numpy as np

try:
    from scipy.integrate import quad
except Exception:
    quad = None

try:
    import emcee
except Exception:
    emcee = None

# -----------------------------------------------------------------------------
# Massive Graviton Phenomenology (Fixed MG Shape)
# -----------------------------------------------------------------------------

c_global = 2.99792458e8
hbar_global = 1.0545718e-34
M_G_REF_global = 8.1e-69

H0_SQ_MAG = 4.84e-36
OMEGA_MG_MAG = 0.7

c = c_global
hbar = hbar_global
M_G_REF = M_G_REF_global

# FIXED MG SHAPE (from your 0.9998 evolved model)
TRANSITION_MIDPOINT_FIXED = 0.597
TRANSITION_WIDTH_FIXED    = 0.252
H0_EARLY_FIXED            = 67.0
H0_LATE_FIXED             = 73.0
EPSILON_FIXED             = -0.047


def H_mg_phenomenological(
    a,
    m_g,
    *,
    transition_midpoint=None,
    transition_width=None,
    H0_early_km_s_Mpc=None,
    H0_late_km_s_Mpc=None,
    epsilon=None,
):
    """
    Uses final evolved MG dark-energy shape if parameters are None.
    """

    H0_SQ = H0_SQ_MAG
    OMEGA_MG = OMEGA_MG_MAG

    a = float(a)
    if a <= 0.0:
        a = 1e-8

    # Use fixed MG shape
    if transition_midpoint is None:
        transition_midpoint = TRANSITION_MIDPOINT_FIXED
    if transition_width is None:
        transition_width = TRANSITION_WIDTH_FIXED
    if H0_early_km_s_Mpc is None:
        H0_early_km_s_Mpc = H0_EARLY_FIXED
    if H0_late_km_s_Mpc is None:
        H0_late_km_s_Mpc = H0_LATE_FIXED
    if epsilon is None:
        epsilon = EPSILON_FIXED

    mass_factor = (m_g / M_G_REF) ** 2

    H0_early_sq = (H0_early_km_s_Mpc * 1e3 / 3.086e22)**2
    H0_late_sq  = (H0_late_km_s_Mpc * 1e3 / 3.086e22)**2

    x = (a - transition_midpoint) / transition_width
    transition_factor = 1.0 / (1.0 + math.exp(-x))

    H0_ratio = H0_late_sq / H0_early_sq
    dynamical_factor = 1.0 + (H0_ratio - 1.0) * transition_factor

    power_factor = a ** epsilon

    a_factor = dynamical_factor * power_factor

    return H0_SQ * OMEGA_MG * mass_factor * a_factor


# -----------------------------------------------------------------------------
# Paths and Data Loading
# -----------------------------------------------------------------------------

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_DIR, "data")

c_km_s = 2.99792458e5

def load_pantheon_style(path: str) -> dict:
    try:
        data = np.genfromtxt(path, names=True, dtype=None, encoding=None)
    except Exception:
        arr = np.loadtxt(path, skiprows=1)
        return {"z": arr[:, 0], "mu": arr[:, 1], "sigma_mu": arr[:, 2]}

    def get_col(names):
        for n in names:
            if n in data.dtype.names:
                return data[n]
        raise ValueError("Missing required SN column.")

    z = get_col(["zHD", "z", "z_cmb"])
    mu = get_col(["MU_SH0ES", "mu", "MU"])
    sigma_mu = get_col(["MU_SH0ES_ERR_DIAG", "sigma_mu", "dmu", "err"])
    return {"z": z, "mu": mu, "sigma_mu": sigma_mu}

PANTHEON_PATH = os.path.join(DATA_DIR, "PantheonSH0ES.dat")
if os.path.exists(PANTHEON_PATH):
    SN_DATA = load_pantheon_style(PANTHEON_PATH)
else:
    SN_DATA = {
        "z": np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.00]),
        "mu": np.array([33.257, 36.816, 38.395, 40.034, 41.031, 42.332, 43.568, 44.163]),
        "sigma_mu": np.array([0.15, 0.15, 0.15, 0.17, 0.18, 0.20, 0.22, 0.25]),
    }

BAO_DATA = {
    "z": np.array([0.106, 0.15, 0.32, 0.57]),
    "DV_over_rd": np.array([3.11, 4.51, 8.59, 13.98]),
    "sigma": np.array([0.18, 0.17, 0.17, 0.14]),
    "rd": 147.09,
}

CMB_PRIOR = {
    "R_obs": 1.7502,
    "sigma_R": 0.0046,
    "z_star": 1089.92,
}

# -----------------------------------------------------------------------------
# Cosmology Helpers
# -----------------------------------------------------------------------------

def H_LCDM_squared(a, H0_si, Omega_m, Omega_r, Omega_L):
    if a <= 1e-9:
        a = 1e-9
    return H0_si**2 * (Omega_r/a**4 + Omega_m/a**3 + Omega_L)

def H_total_of_z(z, theta):
    """
    theta = [H0, Omega_m, f_mg, M_abs]
    """
    if len(theta) != 4:
        raise ValueError("H_total_of_z expects 4 parameters.")

    H0_km_s_Mpc, Omega_m, f_mg, M_abs = theta

    H0_si = H0_km_s_Mpc * 1000.0 / 3.085677581e22
    Omega_r = 9e-5
    Omega_L = 1.0 - Omega_m - Omega_r

    a = 1.0 / (1.0 + z)
    m_g = f_mg * M_G_REF

    H2_LCDM = H_LCDM_squared(a, H0_si, Omega_m, Omega_r, Omega_L)

    # MG term with fixed shape
    H2_mg = H_mg_phenomenological(a, m_g)

    H2_total = H2_LCDM + H2_mg
    if H2_total <= 0.0:
        return 1e-10

    return math.sqrt(H2_total) * (3.085677581e22 / 1000.0)  # convert to km/s/Mpc

def comoving_distance_Mpc(z, theta):
    if quad is None:
        raise RuntimeError("Scipy is required.")
    def integrate_one(zi):
        integrand = lambda zp: c_km_s / H_total_of_z(zp, theta)
        chi, _ = quad(integrand, 0.0, zi, epsabs=1e-5, epsrel=1e-5)
        return chi
    if np.ndim(z) > 0:
        return np.array([integrate_one(zi) for zi in z])
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
    return ((1+z)**2 * DA**2 * (c_km_s * z / Hz))**(1/3)

def distance_modulus(z, theta, M_abs):
    DL = luminosity_distance_Mpc(z, theta)
    DL = np.where(DL <= 0, 1e-9, DL)
    return 5 * np.log10(DL*1e5) + M_abs

# -----------------------------------------------------------------------------
# Likelihoods
# -----------------------------------------------------------------------------

def log_likelihood_SN(theta):
    z = SN_DATA["z"]
    mu_obs = SN_DATA["mu"]
    sigma_mu = SN_DATA["sigma_mu"]
    H0, Om, f_mg, M_abs = theta
    mu_th = distance_modulus(z, theta, M_abs)
    chi2 = np.sum(((mu_obs - mu_th)/sigma_mu)**2)
    return -0.5 * chi2

def log_likelihood_BAO(theta):
    z = BAO_DATA["z"]
    DV_obs = BAO_DATA["DV_over_rd"]
    sigma = BAO_DATA["sigma"]
    rd = BAO_DATA["rd"]
    DV_th = np.array([DV_Mpc(zi, theta) for zi in z])
    DV_over_rd_th = DV_th / rd
    chi2 = np.sum(((DV_obs - DV_over_rd_th)/sigma)**2)
    return -0.5 * chi2

def log_likelihood_CMB(theta):
    H0, Om, f_mg, M_abs = theta
    z_star = CMB_PRIOR["z_star"]
    DM_star = comoving_distance_Mpc(z_star, theta)
    H0_si = H0 * 1000.0 / 3.085677581e22
    DM_si = DM_star * 3.085677581e22
    R_th = math.sqrt(Om) * H0_si * DM_si / c_global
    R_obs = CMB_PRIOR["R_obs"]
    sigma_R = CMB_PRIOR["sigma_R"]
    chi2 = ((R_th - R_obs)/sigma_R)**2
    return -0.5 * chi2

def log_likelihood_total(params):
    if len(params) != 4:
        return -np.inf
    if np.any(np.isnan(params)):
        return -np.inf

    H0, Om, f_mg, M_abs = params
    if H0 <= 0 or Om <= 0:
        return -np.inf

    return (
        log_likelihood_SN(params)
        + log_likelihood_BAO(params)
        + log_likelihood_CMB(params)
    )

# -----------------------------------------------------------------------------
# Priors
# -----------------------------------------------------------------------------

def log_prior(params):
    if len(params) != 4:
        return -np.inf

    H0, Om, f_mg, M_abs = params

    if not (40.0 < H0 < 90.0): return -np.inf
    if not (0.05 < Om < 0.6): return -np.inf
    if not (0.0 < f_mg < 10.0): return -np.inf
    if not (-20.5 < M_abs < -17.0): return -np.inf

    return 0.0

def log_posterior(params):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_total(params)

# -----------------------------------------------------------------------------
# Run MCMC
# -----------------------------------------------------------------------------

def run_mcmc(n_walkers=32, n_steps=500):
    if emcee is None:
        raise RuntimeError("emcee not installed.")

    ndim = 4
    initial_center = np.array([70.0, 0.3, 0.05, -19.1])
    p0 = initial_center + 1e-2 * np.random.randn(n_walkers, ndim)

    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior)
    sampler.run_mcmc(p0, n_steps, progress=True)
    return sampler

if __name__ == "__main__":
    try:
        sampler = run_mcmc(n_walkers=32, n_steps=1000)
        flat = sampler.get_chain(discard=200, thin=10, flat=True)
        mean = np.mean(flat, axis=0)
        std = np.std(flat, axis=0)

        labels = ["H0_km_s_Mpc", "Omega_m", "f_mg", "M_abs"]

        print("\n--- Results (post-burn-in) ---")
        for i, lbl in enumerate(labels):
            print(f"{lbl}: {mean[i]:.4f} +/- {std[i]:.4f}")

        print("\nAcceptance fraction (avg):", np.mean(sampler.acceptance_fraction))
    except Exception as e:
        print("MCMC run failed:", e)
