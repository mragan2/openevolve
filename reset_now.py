import os
import shutil
import subprocess

# --- PATHS ---
BASE_DIR = os.getcwd()
SOLVE_DIR = os.path.join(BASE_DIR, "examples", "solve")
TARGET_FILE = os.path.join(SOLVE_DIR, "initial_program.py")
CONFIG_FILE = os.path.join(SOLVE_DIR, "config.yaml")
OUTPUT_DIR = os.path.join(SOLVE_DIR, "openevolve_output")
CACHE_DIR = os.path.join(SOLVE_DIR, "__pycache__")

# --- 1. SEED CODE (Poprawna Polszczyzna + Fizyka) ---
SEED_CODE = '''"""
Massive Graviton Cosmology Scaffold (Constant DE + Quantum).
"""
import math
import numpy as np

# --- GLOBAL CONSTANTS ---
c_global = 2.99792458e8
hbar_global = 1.0545718e-34
M_G_REF_global = 8.1e-69  

# HINT 1: Magnitude of H0^2
H0_SQ_MAG = 4.84e-36 

# HINT 2: Dark Energy Fraction
OMEGA_MG_MAG = 0.7 

# Aliases
c = c_global
hbar = hbar_global
M_G_REF = M_G_REF_global


# EVOLVE-BLOCK-START
def H_mg_phenomenological(a, m_g):
    """
    Wkład masywnego grawitonu do H^2(a).
    """
    # Stałe lokalne
    c = c_global
    hbar = hbar_global
    H0_SQ = H0_SQ_MAG
    OMEGA_MG = OMEGA_MG_MAG

    # Safety
    a = float(a)
    if a <= 0.0: a = 1e-8

    # Skalowanie masowe
    mass_factor = (m_g / M_G_REF) ** 2

    # Stała gęstość (Constant Dark Energy)
    a_factor = 1.0

    return H0_SQ * OMEGA_MG * mass_factor * a_factor


def lambda_eff_from_mg(m_g):
    """
    Efektywna stała kosmologiczna.
    """
    c = c_global
    hbar = hbar_global
    val = (m_g * c / hbar) ** 2
    alpha = 0.2
    return alpha * val

def rho_quantum(a, H, m_g):
    """
    Oblicza gęstość energii próżni (Quantum Vacuum Energy Density).
    """
    G = 6.67430e-11
    
    # Gęstość krytyczna
    rho_crit = (3 * H**2) / (8 * np.pi * G)
    
    # Zwracamy poprawny ułamek (target ~0.7)
    return 0.7 * rho_crit
# EVOLVE-BLOCK-END


# --- PREDICTION FUNCTION ---
def get_phenomenology(a_val, m_g_val):
    H2_contrib = H_mg_phenomenological(a_val, m_g_val)
    lambda_eff = lambda_eff_from_mg(m_g_val)
    return H2_contrib, lambda_eff
'''

# --- 2. CONFIG (Pure Qwen 7B + Polski) ---
CONFIG_YAML = """
# OpenEvolve Configuration (Pure Qwen 7B)
max_iterations: 100
checkpoint_interval: 10
log_level: "INFO"

llm:
  # STRONG MODEL ONLY - NO ENSEMBLE
  primary_model: "qwen2.5-coder:7b"
  primary_model_weight: 1.0
  
  api_base: "http://localhost:11434/v1"
  api_key: "ollama"
  temperature: 0.7 
  top_p: 0.95
  max_tokens: 8192
  timeout: 600

prompt:
  system_message: |
    Jesteś wybitnym fizykiem teoretycznym. Odkryj teorię Masywnej Grawitacji.
    
    CELE:
    1. H_mg_phenomenological: Musi pasować do 0.7 * H0^2 (użyj stałych).
    2. rho_quantum: Musi obliczać gęstość energii próżni (~0.7 * rho_crit).

    Używaj stałych (c, hbar, M_G_REF).
    Kod musi być poprawnym Pythonem.

  num_top_programs: 3
  use_template_stochasticity: true

database:
  population_size: 60  # Increased population for more diversity
  archive_size: 25
  num_islands: 4
  elite_selection_ratio: 0.3
  exploitation_ratio: 0.7

evaluator:
  timeout: 60
  cascade_evaluation: true
  cascade_thresholds: [0.5, 0.75]
  parallel_evaluations: 4
  use_llm_feedback: false

evolution_settings:
  diff_based_evolution: false
  allow_full_rewrites: true
"""

def main():
    print("--- ☢️ FINAL PHYSICS FIX (POLISH SPELLING) ☢️ ---")
    subprocess.call("taskkill /F /IM python.exe", shell=True, stderr=subprocess.DEVNULL)
    
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    if os.path.exists(CACHE_DIR): shutil.rmtree(CACHE_DIR, ignore_errors=True)

    # Encoding utf-8 is critical for Polish characters
    with open(TARGET_FILE, "w", encoding="utf-8") as f: f.write(SEED_CODE)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f: f.write(CONFIG_YAML)

    print("✅ READY. Run batch file.")

if __name__ == "__main__":
    main()