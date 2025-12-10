import math
import numpy as np
import matplotlib.pyplot as plt

from final_physics_model import calculate_rotation_velocity, G, KPC_TO_M

# --- Synthetic NGC 6503–like data (exactly as in evaluator) ---

DATA_R_KPC = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20], dtype=float)
DATA_V_OBS = np.array([110, 115, 118, 120, 121, 121, 120, 119, 118, 117], dtype=float)  # flat curve
DATA_V_BAR = np.array([108, 105, 95, 85, 75, 68, 62, 58, 54, 50], dtype=float)          # Keplerian-like falloff

def enclosed_mass_from_baryonic_velocity(r_kpc: float, v_bary_kms: float) -> float:
    """
    Approximate enclosed baryonic mass using v^2 = G M / r.

    Parameters
    ----------
    r_kpc : float
        Radius in kiloparsecs.
    v_bary_kms : float
        Baryonic circular velocity in km/s.

    Returns
    -------
    float
        Enclosed mass in kg.
    """
    r_m = r_kpc * KPC_TO_M
    v_ms = v_bary_kms * 1000.0
    return (v_ms ** 2) * r_m / G

def main():
    v_total = []
    rel_errors = []

    print(" r_kpc | v_obs | v_bary | v_total | rel_error")
    print("----------------------------------------------")

    for r, v_obs, v_bar in zip(DATA_R_KPC, DATA_V_OBS, DATA_V_BAR):
        M_enc = enclosed_mass_from_baryonic_velocity(r, v_bar)
        v_pred = float(calculate_rotation_velocity(r, v_bar, M_enc))
        v_total.append(v_pred)

        rel_err = abs(v_pred - v_obs) / v_obs
        rel_errors.append(rel_err)

        print(f"{r:5.1f} | {v_obs:5.1f} | {v_bar:6.1f} | {v_pred:7.2f} | {rel_err*100:8.3f}%")

    v_total = np.array(v_total, dtype=float)
    rel_errors = np.array(rel_errors, dtype=float)

    mean_rel_err = float(np.mean(rel_errors))
    max_rel_err = float(np.max(rel_errors))

    print("\nMean relative error: {:.3f}%".format(mean_rel_err * 100.0))
    print("Max  relative error: {:.3f}%".format(max_rel_err * 100.0))

    # --- Plot: v_bary, v_total, v_obs vs radius ---

    plt.figure()
    plt.plot(DATA_R_KPC, DATA_V_OBS, marker="o", label="Observed (synthetic)")
    plt.plot(DATA_R_KPC, DATA_V_BAR, marker="s", label="Baryonic only")
    plt.plot(DATA_R_KPC, v_total, marker="^", label="Baryonic + MG (model)")

    plt.xlabel("Radius r (kpc)")
    plt.ylabel("Rotation velocity (km/s)")
    plt.title("Galaxy Rotation Curve Fit – MOG/Yukawa + Vainshtein")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
