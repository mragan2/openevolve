import os
import shutil

# --- PATHS ---
BASE_DIR = os.getcwd()
NEW_DIR = os.path.join(BASE_DIR, "examples", "galaxy_rotation")
SOLVE_DIR = os.path.join(BASE_DIR, "examples", "solve")

# Source of your winning Massive Gravity code
SOURCE_BEST_PROG = os.path.join(SOLVE_DIR, "openevolve_output", "best", "best_program.py")

# Target Files
TARGET_SEED = os.path.join(NEW_DIR, "initial_program.py")
TARGET_EVAL = os.path.join(NEW_DIR, "evaluator.py")
TARGET_CONFIG = os.path.join(NEW_DIR, "config.yaml")
TARGET_BAT = os.path.join(BASE_DIR, "run_galaxy.bat")

# --- 1. THE ROTATION CURVE EVALUATOR ---
GALAXY_EVALUATOR = '''"""
Galaxy Rotation Evaluator.
Tests if Massive Gravity can explain rotation curves WITHOUT Dark Matter.
"""
import importlib.util
import math
import sys
from pathlib import Path
import numpy as np

# --- PHYSICAL CONSTANTS ---
G_NEWTON = 6.67430e-11
M_SUN = 1.989e30       # kg
KPC_TO_M = 3.086e19    # meters

# --- REAL GALAXY DATA (Synthetic NGC 6503 Approximation) ---
# Radius (kpc), Velocity_Observed (km/s), Velocity_Baryonic_Only (km/s)
# Baryonic = Stars + Gas (Visible Matter)
DATA_R_KPC = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
DATA_V_OBS = np.array([110, 115, 118, 120, 121, 121, 120, 119, 118, 117]) # Flat curve
DATA_V_BAR = np.array([108, 105, 95, 85, 75, 68, 62, 58, 54, 50])         # Keplerian fall-off

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

    # Get the user's function
    calc_func = getattr(module, "calculate_rotation_velocity", None)
    if not calc_func:
        return {"combined_score": 0.0}

    # --- EVALUATION LOOP ---
    try:
        # We define a fixed Mass of the Graviton based on previous discovery
        # m_g ~ 8.1e-69 kg. The AI can tune a 'coupling strength' or 'Yukawa scale'.
        
        errors = []
        for r_kpc, v_obs, v_bar in zip(DATA_R_KPC, DATA_V_OBS, DATA_V_BAR):
            
            # The AI function takes:
            # r (kpc), v_baryonic (km/s), M_enclosed (approx)
            
            # Approx enclosed mass from baryonic velocity: V^2 = GM/r
            # M = V^2 * r / G
            r_m = r_kpc * KPC_TO_M
            v_bar_ms = v_bar * 1000
            M_enc = (v_bar_ms**2 * r_m) / G_NEWTON
            
            # Call AI function
            v_pred_kms = float(calc_func(r_kpc, v_bar, M_enc))
            
            # Calculate Error
            err = abs(v_pred_kms - v_obs) / v_obs
            errors.append(err)
        
        mean_error = np.mean(errors)
        
        # Score: 1.0 if error is 0. 0.0 if error is large.
        score = 1.0 / (1.0 + mean_error * 10)
        
        metrics["rotation_match"] = score
        metrics["combined_score"] = score
        metrics["stability"] = 1.0
        
    except Exception as e:
        # print(e)
        metrics["combined_score"] = 0.0

    return metrics

def evaluate_stage1(p): return evaluate(p)
def evaluate_stage2(p): return evaluate(p)
'''

# --- 2. THE GALAXY CONFIG ---
GALAXY_CONFIG = """
# OpenEvolve: The Dark Matter Killer
max_iterations: 100
checkpoint_interval: 10
log_level: "INFO"

llm:
  # Using your Cloud Setup for max intelligence
  primary_model: "qwen3-coder:480b-cloud"
  primary_model_weight: 1.0
  api_base: "http://localhost:11434/v1"
  api_key: "aa249496fa974637a67ebe8f05be1e21.bfs5CdlZ_ocSK0O__Guty9w0"
  
  temperature: 0.8
  top_p: 0.95
  max_tokens: 8192
  timeout: 600

prompt:
  system_message: |
    Jesteś fizykiem pracującym nad Modyfikowaną Grawitacją (MOG/Yukawa).
    
    PROBLEM:
    Galaktyki wirują za szybko. Standardowa grawitacja (Newton) daje `v_baryonic`.
    Obserwacje (`v_obs`) są znacznie wyższe na krawędziach.
    
    ZADANIE:
    Napisz funkcję `calculate_rotation_velocity(r_kpc, v_baryonic, M_enclosed)`.
    Musisz dodać "siłę Yukawy" pochodzącą od masywnego grawitonu, aby podbić prędkość.
    
    Wzór ogólny: V_total = sqrt( v_baryonic^2 + V_yukawa^2 )
    
    Gdzie V_yukawa zależy od masy M_enclosed i stałej sprzężenia alpha.
    Użyj masy grawitonu (m_g ~ 10^-69) lub skali długości (lambda ~ 4.6 Gly = 1.4e26 m).
    UWAGA: Skala galaktyczna (kpc) jest DUŻO mniejsza niż lambda. Musisz znaleźć efekt "screening" lub "Vainshtein".

  num_top_programs: 3
  use_template_stochasticity: true

database:
  population_size: 40
  archive_size: 15
  num_islands: 2
  elite_selection_ratio: 0.2
  exploitation_ratio: 0.6

evaluator:
  timeout: 60
  cascade_evaluation: true
  parallel_evaluations: 1
  use_llm_feedback: false

evolution_settings:
  diff_based_evolution: false
  allow_full_rewrites: true
  max_code_length: 10000
"""

# --- 3. THE GALAXY SEED (Blank Slate) ---
GALAXY_SEED = '''"""
Galaxy Rotation Seed.
"""
import math
import numpy as np

# Constants
G = 6.67430e-11
KPC_TO_M = 3.086e19
M_G_REF = 8.1e-69  # Your discovered mass

def calculate_rotation_velocity(r_kpc, v_baryonic, M_enclosed):
    """
    Calculates total rotation velocity.
    Currently just returns Newtonian velocity (Standard Physics).
    AI must modify this to include Massive Gravity effects.
    """
    # Placeholder: Newton only (Fails to explain rotation curves)
    return v_baryonic
'''

# --- 4. LAUNCHER ---
GALAXY_BAT = r"""@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
python openevolve-run.py examples/galaxy_rotation/initial_program.py examples/galaxy_rotation/evaluator.py --config examples/galaxy_rotation/config.yaml
pause
"""

def main():
    print(f"--- 🌌 SETUP: {NEW_DIR} 🌌 ---")
    
    # 1. Create Directory
    os.makedirs(NEW_DIR, exist_ok=True)
    
    # 2. Write Evaluator (Data inside)
    with open(TARGET_EVAL, "w", encoding="utf-8") as f:
        f.write(GALAXY_EVALUATOR)

    # 3. Write Config (Prompt updated)
    with open(TARGET_CONFIG, "w", encoding="utf-8") as f:
        f.write(GALAXY_CONFIG)

    # 4. Write Seed (Blank)
    with open(TARGET_SEED, "w", encoding="utf-8") as f:
        f.write(GALAXY_SEED)

    # 5. Write Launcher
    with open(TARGET_BAT, "w", encoding="utf-8") as f:
        f.write(GALAXY_BAT)

    print(f"✅ Created launcher: run_galaxy.bat")
    print("\n🚀 READY. Double-click 'run_galaxy.bat' to kill Dark Matter.")

if __name__ == "__main__":
    main()