"""
Self-contained MCMC skeleton for your Massive Graviton background model.

- Uses your H_mg_phenomenological(a, m_g) from the scaffold.
- Parameter vector: [H0_km_s_Mpc, Omega_m, f_mg, M_abs]
  where m_g = f_mg * M_G_REF_global.
- Uses toy SN / BAO / CMB-like data so it runs immediately.
"""

import numpy as np
from scipy.integrate import quad
import emcee
import math

# ------------------------------------------------------------
# 0. Your Massive Graviton H_mg_phenomenological
# ------------------------------------------------------------

# --- GLOBAL CONSTANTS ---
c_global = 2.99792458e8
hbar_global = 1.0545718e-34
M_G_REF_global = 8.1e-69  # reference graviton mass [kg]

# H0^2 magnitude (you used this earlier)
H0_SQ_MAG = 4.84e-36      # ~ (2.2e-18 s^-1)^2
OMEGA_MG_MAG = 0.7        # target DE fraction from MG

# Aliases
c = c_global
hbar = hbar_global
M_G_REF = M_G_REF_global


def H_mg_phenomenological(a, m_g):
    """
    Phenomenological massive graviton contribution to H^2(a) in SI units [s^-2].

    For m_g = M_G_REF and a = 1:
        H_mg_phenomenological(1, M_G_REF) ~ OMEGA_MG_MAG * H0_SQ_MAG.

    Includes a smooth transition between early and late H0 values and
    a small power-law correction in a.
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


import numpy as np
import os

# ------------------------------------------------------------
# Real-ish / realistic DATA SECTION
# ------------------------------------------------------------

# ---------- Supernova: Pantheon+ loader (real data) ----------

def load_pantheon_plus(path):
    """
    Load Pantheon+ (or similar) SN data from a text/CSV file.

    Expected columns (or adapt as needed):
        z, mu, sigma_mu

    You control the file format; this is a simple example assuming
    a whitespace- or comma-delimited file with those three columns.
    """
    arr = np.loadtxt(path, usecols=(0, 1, 2))
    z = arr[:, 0]
    mu = arr[:, 1]
    sigma_mu = arr[:, 2]
    return {"z": z, "mu": mu, "sigma_mu": sigma_mu}


# Point this to your actual Pantheon+ file once you download it.
# For now, fall back to a tiny mock subset if the file is missing.
PANTHEON_PATH = os.path.join("data", "pantheon_plus_sn.txt")

if os.path.exists(PANTHEON_PATH):
    SN_DATA = load_pantheon_plus(PANTHEON_PATH)
else:
    # Minimal mock sample in roughly the right redshift range
    # to keep the script runnable even without the file.
    SN_DATA = {
        "z": np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]),
        "mu": np.array([33.2, 36.3, 37.8, 39.5, 40.6, 42.2, 43.3, 44.0]),
        "sigma_mu": np.array([0.15, 0.15, 0.15, 0.17, 0.18, 0.2, 0.22, 0.25]),
    }


# ---------- BAO: BOSS DR12 DV/rd (Sánchez / Alam DR12) ----------

# From BOSS DR12 combined sample (BAO-only): at z_eff = 0.38, 0.51, 0.61
# DV(z)/rd = 9.89 ± 0.15, 12.86 ± 0.18, 14.51 ± 0.21. :contentReference[oaicite:3]{index=3}

BAO_DATA = {
    "z": np.array([0.38, 0.51, 0.61]),
    "DV_over_rd": np.array([9.89, 12.86, 14.51]),
    "sigma": np.array([0.15, 0.18, 0.21]),
    # You can treat rd as a free parameter or fix it near Planck:
    "rd": 147.09,   # Mpc, Planck 2018 best-fit sound horizon at drag epoch
}


# ---------- CMB: Planck 2018 distance prior (shift parameter R) ----------

# Planck 2018 shift parameter for flat ΛCDM: R_obs = 1.7502 ± 0.0046 at z_* = 1089.92. :contentReference[oaicite:4]{index=4}
CMB_PRIOR = {
    "R_obs": 1.7502,
    "sigma_R": 0.0046,
    "z_star": 1089.92,
}



# ------------------------------------------------------------
# 2. Cosmology helpers: H_LCDM, H_total, distances
# ------------------------------------------------------------

def H_LCDM_squared(a, H0_si, Omega_m, Omega_r, Omega_L):
    """
    Flat LCDM H^2(a) in SI units [s^-2].
    """
    return H0_si**2 * (Omega_r / a**4 + Omega_m / a**3 + Omega_L)


def H_total_of_z(z, theta):
    """
    Total H(z) in km/s/Mpc for parameter vector:
      theta = [H0_km_s_Mpc, Omega_m, f_mg]

    with m_g = f_mg * M_G_REF.
    """
    H0_km_s_Mpc, Omega_m, f_mg = theta

    # Radiation density parameter (approx fixed)
    Omega_r = 9e-5
    Omega_L = 1.0 - Omega_m - Omega_r

    # Convert H0 to SI [s^-1]
    H0_si = H0_km_s_Mpc * 1000.0 / 3.085677581e22

    a = 1.0 / (1.0 + z)
    m_g = f_mg * M_G_REF

    H2_LCDM = H_LCDM_squared(a, H0_si, Omega_m, Omega_r, Omega_L)
    H2_mg = H_mg_phenomenological(a, m_g)

    H2_total_si = H2_LCDM + H2_mg
    if H2_total_si <= 0.0:
        # guard against pathological parameter sets
        return 1e9

    H_si = math.sqrt(H2_total_si)

    # convert back to km/s/Mpc
    H_km_s_Mpc = H_si * (3.085677581e22 / 1000.0)
    return H_km_s_Mpc


def comoving_distance_Mpc(z, theta):
    """
    Comoving distance chi(z) in Mpc.
    """
    integrand = lambda zp: c_km_s / H_total_of_z(zp, theta)
    chi, _ = quad(integrand, 0.0, z, epsabs=1e-6, epsrel=1e-6)
    return chi  # Mpc


def luminosity_distance_Mpc(z, theta):
    chi = comoving_distance_Mpc(z, theta)
    return (1.0 + z) * chi


def angular_diameter_distance_Mpc(z, theta):
    chi = comoving_distance_Mpc(z, theta)
    return chi / (1.0 + z)


def DV_Mpc(z, theta):
    """
    Volume-averaged BAO distance:
      D_V = [ (1+z)^2 * D_A^2 * c*z / H(z) ]^(1/3)
    """
    DA = angular_diameter_distance_Mpc(z, theta)
    Hz = H_total_of_z(z, theta)
    return ((1 + z)**2 * DA**2 * (c_km_s * z / Hz))**(1.0 / 3.0)


def distance_modulus(z, theta, M_abs):
    """
    Distance modulus mu(z) = 5 log10(D_L / 10 pc),
    with D_L in Mpc and 10 pc = 1e-5 Mpc.
    """
    DL = luminosity_distance_Mpc(z, theta)
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
    """
    CMB shift-parameter prior:

      R = sqrt(Ω_m) * H0 * D_A(z_*) / c

    (flat universe, compressed Planck 2018 prior).
    """
    H0_km_s_Mpc, Omega_m, f_mg = theta

    z_star = CMB_PRIOR["z_star"]
    DA_star = angular_diameter_distance_Mpc(z_star, theta)  # Mpc

    # H0 in SI
    H0_si = H0_km_s_Mpc * 1000.0 / 3.085677581e22
    c_si = 2.99792458e8

    R_th = np.sqrt(Omega_m) * H0_si * (DA_star * 3.085677581e22) / c_si

    R_obs = CMB_PRIOR["R_obs"]
    sigma_R = CMB_PRIOR["sigma_R"]

    chi2 = ((R_th - R_obs) / sigma_R) ** 2
    return -0.5 * chi2



def log_likelihood_total(params):
    """
    params = [H0_km_s_Mpc, Omega_m, f_mg, M_abs]
    """
    theta = params[:-1]
    M_abs = params[-1]

    ll_sn = log_likelihood_SN(theta, M_abs)
    ll_bao = log_likelihood_BAO(theta)
    ll_cmb = log_likelihood_CMB(theta)

    return ll_sn + ll_bao + ll_cmb


# ------------------------------------------------------------
# 4. Priors + posterior
# ------------------------------------------------------------

def log_prior(params):
    H0, Om, f_mg, M_abs = params

    # Very simple box priors
    if not (60.0 < H0 < 80.0):
        return -np.inf
    if not (0.2 < Om < 0.4):
        return -np.inf
    if not (0.1 < f_mg < 10.0):  # allow m_g from 0.1–10 × M_G_REF
        return -np.inf
    if not (-20.5 < M_abs < -18.0):
        return -np.inf

    return 0.0  # flat within bounds


def log_posterior(params):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood_total(params)
    return lp + ll


# ------------------------------------------------------------
# 5. Run MCMC
# ------------------------------------------------------------

def run_mcmc(n_walkers=24, n_steps=2000):
    ndim = 4  # [H0, Omega_m, f_mg, M_abs]

    # crude initial guess near LCDM and your MG mass
    initial = np.array([
        70.0,      # H0
        0.3,       # Omega_m
        1.0,       # f_mg = 1 → m_g = M_G_REF
        -19.3      # SN absolute magnitude
    ])

    p0 = initial + 1e-2 * initial * np.random.randn(n_walkers, ndim)

    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior)
    sampler.run_mcmc(p0, n_steps, progress=True)

    return sampler


if __name__ == "__main__":
    sampler = run_mcmc()
    burn = 500
    flat = sampler.get_chain(discard=burn, thin=10, flat=True)

    mean = np.mean(flat, axis=0)
    std = np.std(flat, axis=0)

    print("Posterior mean  [H0, Omega_m, f_mg, M_abs]:")
    print(mean)
    print("Posterior std:")
    print(std)
