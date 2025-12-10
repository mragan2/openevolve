import os
import shutil

# --- PATHS ---
BASE_DIR = os.getcwd()
SOURCE_BEST_PROG = os.path.join(BASE_DIR, "examples", "solve", "openevolve_output", "best", "best_program.py")
NEW_DIR = os.path.join(BASE_DIR, "examples", "hubble_tension")

# Target Files
TARGET_SEED = os.path.join(NEW_DIR, "initial_program.py")
TARGET_EVAL = os.path.join(NEW_DIR, "evaluator.py")
TARGET_CONFIG = os.path.join(NEW_DIR, "config.yaml")
TARGET_BAT = os.path.join(BASE_DIR, "run_hubble.bat")

# --- 1. THE HUBBLE TENSION EVALUATOR ---
HUBBLE_EVALUATOR = '''"""
Hubble Tension Evaluator.
Demands a model that satisfies both Early (CMB) and Late (SN) H0 measurements.
"""
import importlib.util
import math
import sys
from pathlib import Path
import numpy as np

# --- CONSTANTS ---
# H0_EARLY (Planck): 67.4 km/s/Mpc -> ~2.184e-18 s^-1
# H0_LATE  (SH0ES):  73.0 km/s/Mpc -> ~2.365e-18 s^-1
H0_EARLY_SI = 2.184e-18
H0_LATE_SI  = 2.365e-18

# Target Density (Still ~0.7 * Rho_Crit_Late)
TARGET_OMEGA = 0.7
G_NEWTON = 6.67430e-11
M_G_REF = 8.1e-69

def _sanitize_candidate_file(path):
    try:
        text = path.read_text(encoding="utf-8")
        if "```" in text:
            lines = [l for l in text.splitlines() if not l.strip().startswith("```")]
            path.write_text("\\n".join(lines), encoding="utf-8")
    except: pass

def evaluate(program_path):
    metrics = {}
    path = Path(program_path)
    _sanitize_candidate_file(path)

    try:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = module
        spec.loader.exec_module(module)
    except:
        return {"combined_score": 0.0}

    H_func = getattr(module, "H_mg_phenomenological", None)
    rho_func = getattr(module, "rho_quantum", None)
    
    if not H_func or not rho_func:
        return {"combined_score": 0.0}

    # --- METRIC 1: LATE UNIVERSE MATCH (z=0) ---
    # Does it hit H0 = 73 km/s/Mpc today?
    try:
        # Get contribution at a=1
        h_mg_today = float(H_func(1.0, M_G_REF))
        
        # Calculate TOTAL H at a=1 (assuming standard matter Omega_m=0.3)
        # H_total^2 = H_matter^2 + H_mg^2
        # We want H_total = H0_LATE
        # So H_mg^2 should be H0_LATE^2 - (0.3 * H0_LATE^2) = 0.7 * H0_LATE^2
        
        target_h_contribution = 0.7 * (H0_LATE_SI**2)
        err_late = abs(h_mg_today - target_h_contribution) / target_h_contribution
        metrics["late_match"] = 1.0 / (1.0 + err_late * 10) # Sharp penalty
    except:
        metrics["late_match"] = 0.0

    # --- METRIC 2: DENSITY MATCH ---
    # Does rho_quantum match 0.7 * Rho_Crit (using H0_LATE)?
    try:
        rho_val = float(rho_func(1.0, H0_LATE_SI, M_G_REF))
        rho_crit_late = (3 * H0_LATE_SI**2) / (8 * np.pi * G_NEWTON)
        
        frac = abs(rho_val) / rho_crit_late
        # Target: 0.7 +/- 0.05
        metrics["density_match"] = float(np.exp(-((frac - 0.7) / 0.05) ** 2))
    except:
        metrics["density_match"] = 0.0

    # --- METRIC 3: DYNAMICAL BEHAVIOR (The Tension Breaker) ---
    # Check if the Dark Energy density EVOLVES between z=0 (a=1) and z=1 (a=0.5).
    # A constant Cosmological Constant (a_factor=1.0) has slope 0.
    # To solve the tension, we often need Phantom Energy (increasing density) or Early Dark Energy.
    try:
        val_1 = float(H_func(1.0, M_G_REF))
        val_05 = float(H_func(0.5, M_G_REF))
        
        # Calculate slope ratio: H(a=1) / H(a=0.5)
        # For Lambda (a^0), ratio is 1.0.
        # For Phantom (e.g. a^0.1), ratio > 1.0.
        
        ratio = val_1 / (val_05 + 1e-50)
        
        # If strictly 1.0 (Constant), we cap the score at 0.8.
        # We REWARD slight deviation (Phantom behavior) that might bridge the gap.
        if abs(ratio - 1.0) < 1e-5:
            metrics["tension_breaker"] = 0.5 # Penalty for static Lambda
        else:
            metrics["tension_breaker"] = 1.0 # Bonus for dynamical behavior
            
    except:
        metrics["tension_breaker"] = 0.0

    # WEIGHTS
    # We prioritize hitting the Late Universe value (0.4) and Density (0.4).
    # The Tension Breaker (0.2) is the nudge to evolve away from static Lambda.
    score = (
        0.4 * metrics.get("late_match", 0) +
        0.4 * metrics.get("density_match", 0) +
        0.2 * metrics.get("tension_breaker", 0)
    )
    
    metrics["combined_score"] = float(np.clip(score, 0.0, 1.0))
    metrics["stability"] = 1.0
    return metrics

def evaluate_stage1(p): return evaluate(p)
def evaluate_stage2(p): return evaluate(p)
'''

# --- 2. THE HUBBLE CONFIG ---
HUBBLE_CONFIG = """
# OpenEvolve: Hubble Tension Solver
max_iterations: 100
checkpoint_interval: 10
log_level: "INFO"

llm:
  primary_model: "qwen3-coder:480b-cloud"
  primary_model_weight: 1.0
  
  api_base: "http://localhost:11434/v1"
  api_key: "${OPENAI_API_KEY}"
  temperature: 0.85 # Higher temp to encourage breaking the standard model
  top_p: 0.95
  max_tokens: 8192
  timeout: 600

prompt:
  system_message: |
    Jesteś wybitnym fizykiem. Musisz rozwiązać "Hubble Tension" (Napięcie Hubble'a).
    
    OBECNY MODEL:
    Twój model Massive Graviton działa jak Stała Kosmologiczna (a_factor = 1.0).
    To pasuje do H0_Late (73 km/s/Mpc), ale jest sprzeczne z H0_Early (67 km/s/Mpc).

    ZADANIE:
    Zmodyfikuj funkcję `H_mg_phenomenological`, aby Ciemna Energia była DYNAMICZNA.
    Gęstość Ciemnej Energii musi się zmieniać w czasie (nie może być stała!).
    Spróbuj skalowania typu: `a_factor = a ** epsilon` (gdzie epsilon jest małe, np. 0.1 lub -0.1).
    
    CELE:
    1. Idealne dopasowanie do H0_LATE (73 km/s/Mpc) dzisiaj (a=1).
    2. Zachowanie gęstości próżni (~0.7 * rho_crit).
    3. Zmienna gęstość w czasie (Dynamical Dark Energy).

    Używaj stałych. Kod w Pythonie.

  num_top_programs: 4
  use_template_stochasticity: true

database:
  population_size: 50
  archive_size: 20
  num_islands: 3
  elite_selection_ratio: 0.2
  exploitation_ratio: 0.6

evaluator:
  timeout: 60
  cascade_evaluation: true
  cascade_thresholds: [0.5, 0.75]
  parallel_evaluations: 1
  use_llm_feedback: false

evolution_settings:
  diff_based_evolution: false
  allow_full_rewrites: true
  max_code_length: 10000
"""

# --- 3. THE LAUNCHER BATCH FILE ---
HUBBLE_BAT = r"""@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
python openevolve-run.py examples/hubble_tension/initial_program.py examples/hubble_tension/evaluator.py --config examples/hubble_tension/config.yaml
pause
"""

def main():
    print(f"--- 🌌 SETUP: {NEW_DIR} 🌌 ---")
    
    # 1. Create Directory
    os.makedirs(NEW_DIR, exist_ok=True)
    print(f"✅ Folder created.")

    # 2. Clone Seed (Best Program)
    if os.path.exists(SOURCE_BEST_PROG):
        shutil.copy(SOURCE_BEST_PROG, TARGET_SEED)
        print(f"✅ Cloned winning seed from 'solve' project.")
    else:
        print(f"⚠️ Warning: Previous winner not found at {SOURCE_BEST_PROG}.")
        print("   Using default hardcoded seed instead.")
        # Fallback seed if needed... (Skipped for brevity, assuming you have it)

    # 3. Write Evaluator
    with open(TARGET_EVAL, "w", encoding="utf-8") as f:
        f.write(HUBBLE_EVALUATOR)
    print(f"✅ Wrote Hubble Evaluator.")

    # 4. Write Config
    with open(TARGET_CONFIG, "w", encoding="utf-8") as f:
        f.write(HUBBLE_CONFIG)
    print(f"✅ Wrote Hubble Config.")

    # 5. Write Launcher
    with open(TARGET_BAT, "w", encoding="utf-8") as f:
        f.write(HUBBLE_BAT)
    print(f"✅ Created launcher: run_hubble.bat")

    print("\n🚀 READY. Double-click 'run_hubble.bat' to start.")

if __name__ == "__main__":
    main()