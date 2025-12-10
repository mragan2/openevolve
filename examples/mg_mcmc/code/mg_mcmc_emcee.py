"""
Massive Graviton Cosmology: Background‑only MCMC (SN + BAO + CMB) with
additional dynamical dark‑energy parameters.

This version extends the original background‐only sampler to fit not only
the standard cosmological parameters (H0, Ω_m) and the graviton
mass fraction (f_mg) but also the parameters controlling the
phenomenological time evolution of the massive‑graviton dark‑energy term.

The parameter vector is now:

    [H0_km_s_Mpc, Omega_m, f_mg,
     transition_midpoint, transition_width,
     H0_early_km_s_Mpc, H0_late_km_s_Mpc, epsilon,
     M_abs]

where m_g = f_mg * M_G_REF_global and the extra parameters control the
sigmoid interpolation and tilt in the dark‑energy contribution.  See
`mg_cosmology.py` for details on these parameters.
"""

import os
import math
import numpy as np

# We cannot depend on SciPy or emcee in this environment.  If you wish to
# run the MCMC sampler, ensure these packages are installed in your
# environment.  Here we import them with try/except so that the module
# still loads for inspection even when the dependencies are missing.
try:
    from scipy.integrate import quad
except Exception:
    quad = None

try:
    import emcee
except Exception:
    emcee = None

# -----------------------------------------------------------------------------
# 0. Massive graviton phenomenology with dynamical parameters
# -----------------------------------------------------------------------------

# --- GLOBAL CONSTANTS ---
c_global = 2.99792458e8       # speed of light [m/s]
hbar_global = 1.0545718e-34   # Planck's constant [J·s]
M_G_REF_global = 8.1e-69      # reference graviton mass [kg]

# H0^2 magnitude (from scaffold)
H0_SQ_MAG = 4.84e-36  # s^-2
# Required fraction of dark energy from the massive graviton
OMEGA_MG_MAG = 0.7

# Aliases for convenience
c = c_global
hbar = hbar_global
M_G_REF = M_G_REF_global

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
    Phenomenological massive graviton contribution to H^2(a) in SI units [s^-2].

    This version is compatible with the MCMC parameter vector, which may pass
    transition_midpoint, transition_width, H0_early_km_s_Mpc, H0_late_km_s_Mpc
    and epsilon as sampled parameters.

    If any of these are None, final evolved defaults are used:
      transition_midpoint = 0.597
      transition_width    = 0.252
      H0_early_km_s_Mpc   = 67.0
      H0_late_km_s_Mpc    = 73.0
      epsilon             = -0.047
    """
    H0_SQ = H0_SQ_MAG
    OMEGA_MG = OMEGA_MG_MAG

    # ensure a > 0
    a = float(a)
    if a <= 0.0:
        a = 1e-8

    # ----------------------------
    # Use passed values or defaults
    # ----------------------------
    if transition_midpoint is None:
        transition_midpoint = 0.597
    if transition_width is None:
        transition_width = 0.252
    if H0_early_km_s_Mpc is None:
        H0_early_km_s_Mpc = 67.0
    if H0_late_km_s_Mpc is None:
        H0_late_km_s_Mpc = 73.0
    if epsilon is None:
        epsilon = -0.047

    # mass scaling
    mass_factor = (m_g / M_G_REF) ** 2

    # early/late H0 values in s^-2
    H0_early_sq = (H0_early_km_s_Mpc * 1e3 / 3.086e22) ** 2
    H0_late_sq  = (H0_late_km_s_Mpc * 1e3 / 3.086e22) ** 2

    # smooth transition
    x = (a - transition_midpoint) / transition_width
    transition_factor = 1.0 / (1.0 + math.exp(-x))

    H0_ratio = H0_late_sq / H0_early_sq
    dynamical_factor = 1.0 + (H0_ratio - 1.0) * transition_factor

    # small power-law correction
    power_factor = a ** epsilon

    a_factor = dynamical_factor * power_factor

    return H0_SQ * OMEGA_MG * mass_factor * a_factor  # [s^-2]


# -----------------------------------------------------------------------------
# 1. Paths and data loading
# -----------------------------------------------------------------------------

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_DIR, "data")

c_km_s = 2.99792458e5  # speed of light [km/s]

def load_pantheon_style(path: str) -> dict:
    """
    Load a Pantheon+SH0ES distance modulus file.  If the file cannot be
    read using numpy.genfromtxt it falls back to simple loadtxt.

    The return dictionary contains arrays for 'z', 'mu', and 'sigma_mu'.
    """
    try:
        data = np.genfromtxt(path, names=True, dtype=None, encoding=None)
    except Exception:
        arr = np.loadtxt(path, skiprows=1)
        return {"z": arr[:, 0], "mu": arr[:, 1], "sigma_mu": arr[:, 2]}

    def get_col(candidates):
        for c in candidates:
            if c in data.dtype.names:
                return data[c]
        raise ValueError(f"Could not find a column matching {candidates}; found {data.dtype.names}")

    z = get_col(["zHD", "z", "z_cmb"])
    mu = get_col(["MU_SH0ES", "mu", "MU"])
    sigma_mu = get_col(["MU_SH0ES_ERR_DIAG", "sigma_mu", "dmu", "err"])
    return {"z": z, "mu": mu, "sigma_mu": sigma_mu}

# Attempt to load a Pantheon file; fall back to a small mock sample if not found.
PANTHEON_PATH = os.path.join(DATA_DIR, "PantheonSH0ES.dat")
if os.path.exists(PANTHEON_PATH):
    SN_DATA = load_pantheon_style(PANTHEON_PATH)
else:
    SN_DATA = {
        "z": np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.00]),
        "mu": np.array([33.257, 36.816, 38.395, 40.034, 41.031, 42.332, 43.568, 44.163]),
        "sigma_mu": np.array([0.15, 0.15, 0.15, 0.17, 0.18, 0.20, 0.22, 0.25]),
    }

# BAO data: DV(z)/r_d measurements and uncertainties
BAO_DATA = {
    "z": np.array([0.106, 0.15, 0.32, 0.57]),
    "DV_over_rd": np.array([3.11, 4.51, 8.59, 13.98]),
    "sigma": np.array([0.18, 0.17, 0.17, 0.14]),
    "rd": 147.09,
}

# CMB shift parameter prior
CMB_PRIOR = {
    "R_obs": 1.7502,
    "sigma_R": 0.0046,
    "z_star": 1089.92,
}

# -----------------------------------------------------------------------------
# 2. Cosmology helpers
# -----------------------------------------------------------------------------

def H_LCDM_squared(a: float, H0_si: float, Omega_m: float, Omega_r: float, Omega_L: float) -> float:
    """Flat ΛCDM H^2(a) in SI units [s^-2]."""
    if a <= 1e-9:
        a = 1e-9
    return H0_si**2 * (Omega_r / a**4 + Omega_m / a**3 + Omega_L)

def H_total_of_z(z: float, theta: np.ndarray) -> float:
    """
    Total H(z) in km/s/Mpc for a parameter vector theta.

    Parameters
    ----------
    z : float
        Redshift.
    theta : sequence of floats
        Cosmological parameter vector:
          [H0_km_s_Mpc, Omega_m, f_mg,
           transition_midpoint, transition_width,
           H0_early_km_s_Mpc, H0_late_km_s_Mpc, epsilon]

    Returns
    -------
    float
        Hubble parameter H(z) in km/s/Mpc.
    """
    if np.isscalar(theta) or len(theta) != 8:
        raise ValueError(f"H_total_of_z expects 8 params, got: {theta}")
    H0_km_s_Mpc, Omega_m, f_mg, transition_midpoint, transition_width, H0_early, H0_late, epsilon = theta
    Omega_r = 9e-5
    Omega_L = 1.0 - Omega_m - Omega_r
    H0_si = H0_km_s_Mpc * 1000.0 / 3.085677581e22
    a = 1.0 / (1.0 + z)
    m_g = f_mg * M_G_REF
    H2_LCDM = H_LCDM_squared(a, H0_si, Omega_m, Omega_r, Omega_L)
    H2_mg = H_mg_phenomenological(
        a,
        m_g,
        transition_midpoint=transition_midpoint,
        transition_width=transition_width,
        H0_early_km_s_Mpc=H0_early,
        H0_late_km_s_Mpc=H0_late,
        epsilon=epsilon,
    )
    H2_total_si = H2_LCDM + H2_mg
    if H2_total_si <= 0.0:
        return 1e-10
    H_si = math.sqrt(H2_total_si)
    return H_si * (3.085677581e22 / 1000.0)

def comoving_distance_Mpc(z: float | np.ndarray, theta: np.ndarray) -> float | np.ndarray:
    """Compute the line‑of‑sight comoving distance in Mpc by integrating 1/H(z)."""
    if quad is None:
        raise RuntimeError("scipy.integrate.quad is required for comoving_distance_Mpc")
    def integrate_one(zi):
        integrand = lambda zp: c_km_s / H_total_of_z(zp, theta)
        chi, _ = quad(integrand, 0.0, zi, epsabs=1e-5, epsrel=1e-5)
        return chi
    if np.ndim(z) > 0:
        return np.array([integrate_one(zi) for zi in z])
    else:
        return integrate_one(z)

def luminosity_distance_Mpc(z: float | np.ndarray, theta: np.ndarray) -> float | np.ndarray:
    chi = comoving_distance_Mpc(z, theta)
    return (1.0 + z) * chi

def angular_diameter_distance_Mpc(z: float | np.ndarray, theta: np.ndarray) -> float | np.ndarray:
    chi = comoving_distance_Mpc(z, theta)
    return chi / (1.0 + z)

def DV_Mpc(z: float, theta: np.ndarray) -> float:
    DA = angular_diameter_distance_Mpc(z, theta)
    Hz = H_total_of_z(z, theta)
    return ((1 + z)**2 * DA**2 * (c_km_s * z / Hz))**(1.0 / 3.0)

def distance_modulus(z: np.ndarray, theta: np.ndarray, M_abs: float) -> np.ndarray:
    DL = luminosity_distance_Mpc(z, theta)
    DL = np.where(np.asarray(DL) <= 0, 1e-9, DL)
    return 5.0 * np.log10(DL * 1e5) + M_abs

# -----------------------------------------------------------------------------
# 3. Likelihoods
# -----------------------------------------------------------------------------

def log_likelihood_SN(theta: np.ndarray, M_abs: float) -> float:
    z = SN_DATA["z"]
    mu_obs = SN_DATA["mu"]
    sigma_mu = SN_DATA["sigma_mu"]
    mu_th = distance_modulus(z, theta, M_abs)
    chi2 = np.sum(((mu_obs - mu_th) / sigma_mu) ** 2)
    return -0.5 * chi2

def log_likelihood_BAO(theta: np.ndarray) -> float:
    z = BAO_DATA["z"]
    DV_over_rd_obs = BAO_DATA["DV_over_rd"]
    sigma = BAO_DATA["sigma"]
    rd = BAO_DATA["rd"]
    DV_th = np.array([DV_Mpc(zi, theta) for zi in z])
    DV_over_rd_th = DV_th / rd
    chi2 = np.sum(((DV_over_rd_obs - DV_over_rd_th) / sigma) ** 2)
    return -0.5 * chi2

def log_likelihood_CMB(theta: np.ndarray) -> float:
    H0_km_s_Mpc, Omega_m, f_mg, transition_midpoint, transition_width, H0_early, H0_late, epsilon = theta
    z_star = CMB_PRIOR["z_star"]
    DM_star = comoving_distance_Mpc(z_star, theta)
    H0_si = H0_km_s_Mpc * 1000.0 / 3.085677581e22
    c_si = 2.99792458e8
    DM_star_si = DM_star * 3.085677581e22
    R_th = math.sqrt(Omega_m) * H0_si * DM_star_si / c_si
    R_obs = CMB_PRIOR["R_obs"]
    sigma_R = CMB_PRIOR["sigma_R"]
    chi2 = ((R_th - R_obs) / sigma_R) ** 2
    return -0.5 * chi2

def log_likelihood_total(params: np.ndarray) -> float:
    if len(params) != 9:
        return -np.inf
    theta = params[:-1]
    M_abs = params[-1]
    if np.any(np.isnan(params)):
        return -np.inf
    # Basic physics cut: positive H0 and Ω_m
    if theta[0] <= 0 or theta[1] <= 0:
        return -np.inf
    ll_sn = log_likelihood_SN(theta, M_abs)
    ll_bao = log_likelihood_BAO(theta)
    ll_cmb = log_likelihood_CMB(theta)
    return ll_sn + ll_bao + ll_cmb

# -----------------------------------------------------------------------------
# 4. Priors + posterior
# -----------------------------------------------------------------------------

def log_prior(params: np.ndarray) -> float:
    if len(params) != 9:
        return -np.inf
    H0, Om, f_mg, t_mid, t_width, H0_early, H0_late, eps, M_abs = params
    if not (50.0 < H0 < 90.0): return -np.inf
    if not (0.1 < Om < 0.6): return -np.inf
    if not (0.0 < f_mg < 20.0): return -np.inf
    if not (0.1 < t_mid < 1.5): return -np.inf
    if not (0.01 < t_width < 1.0): return -np.inf
    if not (60.0 < H0_early < 75.0): return -np.inf
    if not (70.0 < H0_late < 80.0): return -np.inf
    if not (-0.5 < eps < 0.5): return -np.inf
    if not (-20.5 < M_abs < -18.0): return -np.inf
    return 0.0

def log_posterior(params: np.ndarray) -> float:
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_total(params)

# -----------------------------------------------------------------------------
# 5. Run MCMC
# -----------------------------------------------------------------------------

def run_mcmc(n_walkers: int = 32, n_steps: int = 500):
    """Run the affine‑invariant MCMC sampler if emcee is available."""
    if emcee is None:
        raise RuntimeError("The emcee package is not available in this environment.")
    ndim = 9
    initial = np.array([
        70.0,
        0.3,
        1.0,
        0.559,
        0.252,
        67.0,
        73.0,
        -0.081,
        -19.3,
    ])
    p0 = initial + 1e-2 * initial * np.random.randn(n_walkers, ndim)
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior)
    sampler.run_mcmc(p0, n_steps, progress=True)
    return sampler

if __name__ == "__main__":
    try:
        sampler = run_mcmc(n_walkers=32, n_steps=1000)
        flat = sampler.get_chain(discard=200, thin=10, flat=True)
        mean = np.mean(flat, axis=0)
        std = np.std(flat, axis=0)
        labels = [
            "H0_km_s_Mpc", "Omega_m", "f_mg",
            "transition_midpoint", "transition_width",
            "H0_early_km_s_Mpc", "H0_late_km_s_Mpc", "epsilon",
            "M_abs",
        ]
        print("\n--- Results (post‑burn‑in) ---")
        for i, lbl in enumerate(labels):
            print(f"{lbl}: {mean[i]:.4f} +/- {std[i]:.4f}")
        print("\nAcceptance fraction (avg):", np.mean(sampler.acceptance_fraction))
    except Exception as e:
        print("MCMC run failed:", e)