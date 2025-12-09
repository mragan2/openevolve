"""
Massive Graviton Cosmology: Time Evolution Solver.
Solves the differential equation da/dt = a * H(a).
"""
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# --- 1. YOUR WINNING MODEL (The Physics Engine) ---
# Constants
H0_SI = 2.2e-18         # Hubble Constant today [s^-1]
H0_SQ = H0_SI**2
M_G_REF = 8.1e-69
OMEGA_M = 0.3           # Standard Matter
OMEGA_R = 9e-5          # Radiation
# Your discovered Dark Energy fraction
OMEGA_MG = 0.7          

def H_mg_phenomenological(a, m_g=M_G_REF):
    """
    Your solution: Constant Dark Energy from Massive Graviton.
    """
    if a <= 0: return 0.0
    mass_factor = (m_g / M_G_REF) ** 2
    a_factor = 1.0  # The constant scaling you discovered
    
    return H0_SQ * OMEGA_MG * mass_factor * a_factor

# --- 2. THE TOTAL HUBBLE FUNCTION ---
def H_total(a):
    """
    Combines Matter, Radiation, and your Massive Graviton.
    H^2(a) = H_matter^2 + H_rad^2 + H_graviton^2
    """
    if a <= 0: return 0.0
    
    # Standard Model components
    H2_matter = H0_SQ * OMEGA_M / (a**3)
    H2_rad    = H0_SQ * OMEGA_R / (a**4)
    
    # Your Component
    H2_mg     = H_mg_phenomenological(a)
    
    return np.sqrt(H2_matter + H2_rad + H2_mg)

# --- 3. THE DIFFERENTIAL EQUATION SOLVER ---
def solve_age_of_universe():
    print("--- 🚀 SOLVING COSMIC EVOLUTION ---")
    
    # The Friedmann Equation can be rearranged to solve for time t:
    # dt = da / (a * H(a))
    # We integrate from a=0 (Big Bang) to a=1 (Today).
    
    def integrand(a):
        return 1.0 / (a * H_total(a))
    
    # Integrate using QUAD (High precision integration)
    # Start slightly after 0 to avoid division by zero singularity
    age_seconds, error = quad(integrand, 1e-10, 1.0)
    
    # Convert to Billions of Years (Gyr)
    SECONDS_PER_YEAR = 31557600
    age_gyr = age_seconds / SECONDS_PER_YEAR / 1e9
    
    print(f"✅ Integration Complete.")
    print(f"   Hubble Time (1/H0): {1.0/H0_SI/SECONDS_PER_YEAR/1e9:.2f} Gyr")
    print(f"   Calculated Age:     {age_gyr:.4f} Billion Years")
    
    return age_gyr

# --- 4. VISUALIZATION ---
def plot_expansion_history():
    # Generate scale factors from past to future
    a_vals = np.linspace(0.01, 2.0, 100)
    H_vals = [H_total(a) for a in a_vals]
    
    plt.figure(figsize=(10, 6))
    plt.plot(a_vals, H_vals, label='Massive Graviton Universe', color='blue', linewidth=2)
    
    # Mark Today
    plt.axvline(x=1.0, color='red', linestyle='--', label='Today (a=1)')
    plt.axhline(y=H0_SI, color='green', linestyle=':', label='H0 (Observed)')
    
    plt.title(f"Expansion History: Hubble Parameter H(a)")
    plt.xlabel("Scale Factor (a)")
    plt.ylabel("Expansion Rate H(a) [s^-1]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("universe_history.png")
    print("    generated: universe_history.png")

if __name__ == "__main__":
    age = solve_age_of_universe()
    plot_expansion_history()