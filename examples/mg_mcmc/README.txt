Massive Graviton MCMC Project (examples/mg_mcmc)
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

  2. From repo root (C:\Users\Michal\Documents\GitHub\openevolve):
        - activate your .venv
        - run the MCMC script:
            python examples/mg_mcmc/code/mg_mcmc_emcee.py

  3. Or simply double-click run_mg_mcmc.bat in the repo root, which:
        - activates .venv
        - runs the script above.

The MCMC samples the parameters:
    [H0_km_s_Mpc, Omega_m, f_mg, M_abs]
with m_g = f_mg * M_G_REF_global.

