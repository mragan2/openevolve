import os
import textwrap

# Base: location of THIS script (should be repo root: openevolve/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_DIR = os.path.join(BASE_DIR, "examples", "mg_mcmc")
CODE_DIR = os.path.join(PROJECT_DIR, "code")
DATA_DIR = os.path.join(PROJECT_DIR, "data")

SCRIPT_PATH = os.path.join(CODE_DIR, "mg_mcmc_emcee.py")
README_PATH = os.path.join(PROJECT_DIR, "README.txt")
PANTHEON_TEMPLATE_PATH = os.path.join(DATA_DIR, "pantheon_plus_sn_template.txt")
RUN_BAT_PATH = os.path.join(BASE_DIR, "run_mg_mcmc.bat")

MG_MCMC_SCRIPT = r"""\"\"\" 
Massive Graviton Cosmology: Background-only MCMC (SN + BAO + CMB).

- Uses your H_mg_phenomenological(a, m_g) (dynamic H0 version).
- Parameter vector: [H0_km_s_Mpc, Omega_m, f_mg, M_abs]
  with m_g = f_mg * M_G_REF_global.
- Data:
  * SN: loads Pantheon+ style file if present, else uses a small mock sample.
  * BAO: BOSS DR12-like DV(z)/rd points.
  * CMB: Planck-like distance prior using the shift parameter R.

Run from repo root (with venv active):
  python examples/mg_mcmc/code/mg_mcmc_emcee.py
\"\"\" 

import os
import math
import numpy as np
from scipy.integrate import quad
import emcee

# ------------------------------------------------------------
# 0. Massive graviton H_mg_phenomenological (your version)
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
    \"\"\" 
    Phenomenological massive graviton contribution to H^2(a) in SI units [s^-2].

    For m_g = M_G_REF and a = 1:
        H_mg_phenomenological(1, M_G_REF) ~ OMEGA_MG_MAG * H0_SQ_MAG.

    Includes a smooth transition between early and late H0 values and
    a small power-law correction in a.
    \"\"\"
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

# This script lives in examples/mg_mcmc/code/
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_DIR, "data")

c_km_s = 2.99792458e5  # speed of light [km/s]


def load_pantheon_style(path):
    \"\"\" 
    Load a simple Pantheon-like SN file with columns:
        z   mu   sigma_mu
    separated by whitespace or commas.
    \"\"\"
    arr = np.loadtxt(path, usecols=(0, 1, 2))
    z = arr[:, 0]
    mu = arr[:, 1]
    sigma_mu = arr[:, 2]
    return {\"z\": z, \"mu\": mu, \"sigma_mu\": sigma_mu}


# Try to load a real file; otherwise use a small mock
PANTHEON_PATH = os.path.join(DATA_DIR, \"pantheon_plus_sn.txt\")

if os.path.exists(PANTHEON_PATH):
    SN_DATA = load_pantheon_style(PANTHEON_PATH)
else:
    # Minimal mock sample in roughly the right redshift range
    SN_DATA = {
        \"z\": np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]),
        \"mu\": np.array([33.2, 36.3, 37.8, 39.5, 40.6, 42.2, 43.3, 44.0]),
        \"sigma_mu\": np.array([0.15, 0.15, 0.15, 0.17, 0.18, 0.20, 0.22, 0.25]),
    }

# BOSS DR12-like BAO DV/rd data (z = 0.38, 0.51, 0.61)
BAO_DATA = {
    \"z\": np.array([0.38, 0.51, 0.61]),
    \"DV_over_rd\": np.array([9.89, 12.86, 14.51]),
    \"sigma\": np.array([0.15, 0.18, 0.21]),
    \"rd\": 147.09,  # Mpc
}

# Planck 2018-like distance prior for the shift parameter R
CMB_PRIOR = {
    \"R_obs\": 1.7502,
    \"sigma_R\": 0.0046,
    \"z_star\": 1089.92,
}


# ------------------------------------------------------------
# 2. Cosmology helpers: H_LCDM, H_total, distances
# ------------------------------------------------------------

def H_LCDM_squared(a, H0_si, Omega_m, Omega_r, Omega_L):
    \"\"\" Flat LCDM H^2(a) in SI units [s^-2]. \"\"\"
    return H0_si**2 * (Omega_r / a**4 + Omega_m / a**3 + Omega_L)


def H_total_of_z(z, theta):
    \"\"\" 
    Total H(z) in km/s/Mpc for parameter vector:
      theta = [H0_km_s_Mpc, Omega_m, f_mg]

    with m_g = f_mg * M_G_REF.
    \"\"\"
    H0_km_s_Mpc, Omega_m, f_mg = theta

    Omega_r = 9e-5
    Omega_L = 1.0 - Omega_m - Omega_r

    # H0 in SI [s^-1]
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
    # back to km/s/Mpc
    H_km_s_Mpc = H_si * (3.085677581e22 / 1000.0)
    return H_km_s_Mpc


def comoving_distance_Mpc(z, theta):
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
    \"\"\" 
    Volume-averaged BAO distance:
      D_V = [ (1+z)^2 * D_A^2 * c*z / H(z) ]^(1/3)
    \"\"\"
    DA = angular_diameter_distance_Mpc(z, theta)
    Hz = H_total_of_z(z, theta)
    return ((1 + z)**2 * DA**2 * (c_km_s * z / Hz))**(1.0 / 3.0)


def distance_modulus(z, theta, M_abs):
    \"\"\" 
    Distance modulus mu(z) = 5 log10(D_L / 10 pc),
    with D_L in Mpc and 10 pc = 1e-5 Mpc.
    \"\"\"
    DL = luminosity_distance_Mpc(z, theta)
    return 5.0 * np.log10(DL * 1e5) + M_abs


# ------------------------------------------------------------
# 3. Likelihoods
# ------------------------------------------------------------

def log_likelihood_SN(theta, M_abs):
    z = SN_DATA[\"z\"]
    mu_obs = SN_DATA[\"mu\"]
    sigma_mu = SN_DATA[\"sigma_mu\"]

    mu_th = distance_modulus(z, theta, M_abs)
    chi2 = np.sum(((mu_obs - mu_th) / sigma_mu) ** 2)
    return -0.5 * chi2


def log_likelihood_BAO(theta):
    z = BAO_DATA[\"z\"]
    DV_over_rd_obs = BAO_DATA[\"DV_over_rd\"]
    sigma = BAO_DATA[\"sigma\"]
    rd = BAO_DATA[\"rd\"]

    DV_th = np.array([DV_Mpc(zi, theta) for zi in z])
    DV_over_rd_th = DV_th / rd

    chi2 = np.sum(((DV_over_rd_obs - DV_over_rd_th) / sigma) ** 2)
    return -0.5 * chi2


def log_likelihood_CMB(theta):
    \"\"\" 
    CMB shift-parameter prior:

      R = sqrt(Ω_m) * H0 * D_A(z_*) / c

    compressed Planck-like prior.
    \"\"\"
    H0_km_s_Mpc, Omega_m, f_mg = theta

    z_star = CMB_PRIOR[\"z_star\"]
    DA_star = angular_diameter_distance_Mpc(z_star, theta)  # Mpc

    H0_si = H0_km_s_Mpc * 1000.0 / 3.085677581e22
    c_si = 2.99792458e8

    R_th = math.sqrt(Omega_m) * H0_si * (DA_star * 3.085677581e22) / c_si

    R_obs = CMB_PRIOR[\"R_obs\"]
    sigma_R = CMB_PRIOR[\"sigma_R\"]

    chi2 = ((R_th - R_obs) / sigma_R) ** 2
    return -0.5 * chi2


def log_likelihood_total(params):
    \"\"\" params = [H0_km_s_Mpc, Omega_m, f_mg, M_abs] \"\"\"
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

    # Simple box priors
    if not (60.0 < H0 < 80.0):
        return -np.inf
    if not (0.2 < Om < 0.4):
        return -np.inf
    if not (0.1 < f_mg < 10.0):  # m_g in [0.1, 10] × M_G_REF
        return -np.inf
    if not (-20.5 < M_abs < -18.0):
        return -np.inf

    return 0.0


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

    initial = np.array([
        70.0,   # H0
        0.3,    # Omega_m
        1.0,    # f_mg
        -19.3,  # M_abs
    ])

    p0 = initial + 1e-2 * initial * np.random.randn(n_walkers, ndim)

    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior)
    sampler.run_mcmc(p0, n_steps, progress=True)
    return sampler


if __name__ == \"__main__\":
    sampler = run_mcmc()
    burn = 500
    flat = sampler.get_chain(discard=burn, thin=10, flat=True)

    mean = np.mean(flat, axis=0)
    std = np.std(flat, axis=0)

    print(\"Posterior mean [H0, Omega_m, f_mg, M_abs]:\")
    print(mean)
    print(\"Posterior std:\")
    print(std)
"""

README_TEXT = """Massive Graviton MCMC Project (examples/mg_mcmc)
==================================================

Structure:
  - code/
      mg_mcmc_emcee.py   : main MCMC driver using H_mg_phenomenological
  - data/
      pantheon_plus_sn.txt          : optional real SN data file you provide
      pantheon_plus_sn_template.txt : template / documentation for columns

Usage:
  1. (Optional) Put your real SN file here:
        examples/mg_mcmc/data/pantheon_plus_sn.txt
     with columns:
        z   mu   sigma_mu
     separated by whitespace or commas.

  2. From repo root (C:\\Users\\Michal\\Documents\\GitHub\\openevolve):
        - activate your .venv
        - run the MCMC script:
            python examples/mg_mcmc/code/mg_mcmc_emcee.py

  3. Or simply double-click run_mg_mcmc.bat in the repo root, which:
        - activates .venv
        - runs the script above.

The MCMC samples the parameters:
    [H0_km_s_Mpc, Omega_m, f_mg, M_abs]
with m_g = f_mg * M_G_REF_global.

"""

PANTHEON_TEMPLATE = """# Pantheon-like SN data template
# Save your real file as: pantheon_plus_sn.txt
# Columns:
#   z           redshift
#   mu          distance modulus
#   sigma_mu    uncertainty on mu
#
# Example (toy numbers):
# 0.01   33.2   0.15
# 0.05   36.3   0.15
# ...
"""

RUN_BAT = r"""@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
python examples\mg_mcmc\code\mg_mcmc_emcee.py
pause
"""


def main() -> None:
    print(f"--- Setting up Massive Graviton MCMC project at: {PROJECT_DIR} ---")

    # 1. Create directory structure
    os.makedirs(CODE_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # 2. Write MCMC script
    with open(SCRIPT_PATH, "w", encoding="utf-8") as f:
        f.write(MG_MCMC_SCRIPT)
    print(f"  - Wrote MCMC script: {SCRIPT_PATH}")

    # 3. Write README
    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(README_TEXT)
    print(f"  - Wrote README: {README_PATH}")

    # 4. Write Pantheon template
    with open(PANTHEON_TEMPLATE_PATH, "w", encoding="utf-8") as f:
        f.write(PANTHEON_TEMPLATE)
    print(f"  - Wrote SN template: {PANTHEON_TEMPLATE_PATH}")

    # 5. Write launcher BAT in repo root
    with open(RUN_BAT_PATH, "w", encoding="utf-8") as f:
        f.write(RUN_BAT)
    print(f"  - Wrote launcher: {RUN_BAT_PATH}")

    print("\n✅ Done.")
    print("You can now:")
    print("  - Put real SN data into: examples/mg_mcmc/data/pantheon_plus_sn.txt")
    print("  - Run MCMC via:")
    print("      python examples/mg_mcmc/code/mg_mcmc_emcee.py")
    print("    or just double-click run_mg_mcmc.bat")


if __name__ == "__main__":
    main()
