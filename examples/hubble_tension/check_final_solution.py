# save as examples/hubble_tension/check_final_solution.py

import numpy as np
from final_solution import H_mg_phenomenological, lambda_eff_from_mg, rho_quantum, M_G_REF_global

H0_LATE_SI = 2.365e-18

def main():
    a_vals = np.logspace(-3, 0, 10)  # a from 1e-3 to 1
    m_g = M_G_REF_global

    print("a, H_mg^2(a) [s^-2], rho_q(a)/rho_crit_late, lambda_eff [m^-2]")
    for a in a_vals:
        H2 = H_mg_phenomenological(a, m_g)
        rho_q = rho_quantum(a, H0_LATE_SI, m_g)
        # rho_crit evaluated at H0_LATE_SI:
        G = 6.67430e-11
        rho_crit_late = (3 * H0_LATE_SI**2) / (8 * np.pi * G)
        frac = rho_q / rho_crit_late
        lam = lambda_eff_from_mg(m_g)
        print(f"{a:.4e}, {H2:.3e}, {frac:.3f}, {lam:.3e}")

if __name__ == "__main__":
    main()
