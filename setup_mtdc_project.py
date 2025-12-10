# -*- coding: utf-8 -*-
import os

# --- PATHS ---
# Define the root directory explicitly for clarity, matching the MTDC style
BASE_DIR = r"C:\Users\Michal\Documents\GitHub\openevolve" # Assuming this is your project root
NEW_DIR_NAME = "examples/galaxy_rotation"
NEW_DIR = os.path.join(BASE_DIR, NEW_DIR_NAME)

# Target file paths
TARGET_SEED = os.path.join(NEW_DIR, "initial_program.py")
TARGET_EVAL = os.path.join(NEW_DIR, "evaluator.py")
TARGET_CONFIG = os.path.join(NEW_DIR, "config.yaml")
TARGET_BAT = os.path.join(BASE_DIR, "run_galaxy.bat")

os.makedirs(NEW_DIR, exist_ok=True)
print(f"Directory created: {NEW_DIR}")


# -----------------------------------------------------
# 1. initial_program.py (GALAXY_SEED)
# -----------------------------------------------------
initial_program = """
"""
Galaxy rotation with unit-consistent Yukawa correction and Vainshtein screening.

This is a v2 seed based on the previous best program, modified to:
  - Preserve the same radial behaviour (screening and Yukawa structure).
  - Make units explicit and consistent:
        * Yukawa term is built in SI units [m^2/s^2]
        * Then converted to [km^2/s^2] before combining with v_baryonic^2
  - Expose a small set of parameters for OpenEvolve to tune.
"""

import math

# ----------------------------------------------------------------------
# Physical constants
# ----------------------------------------------------------------------
G = 6.67430e-11        # Gravitational constant [m^3 kg^-1 s^-2]
KPC_TO_M = 3.086e19    # 1 kpc in meters

# ----------------------------------------------------------------------
# Massive graviton / screening scales (tunable)
# ----------------------------------------------------------------------
LAMBDA_G = 1.4e26        # Graviton Compton wavelength (~4.6 Gly) [m]
R_VAIN_KPC = 25.0        # Vainshtein radius in kpc (screening scale)
SCREENING_POWER = 2.0    # Power in the screening factor (controls steepness)

# Coupling strength:
# This is rescaled so that, after the explicit m^2->km^2 conversion,
# the numerical contribution matches the previous best program with
# YUKAWA_ALPHA ≈ 0.6.
ALPHA_YUKAWA_DIMLESS = 6.0e5

# Threshold for the "r << lambda" regime
RATIO_THRESHOLD = 1.0e-3


def calculate_rotation_velocity(r_kpc, v_baryonic, M_enclosed):
    """
    Compute total rotation velocity (km/s) including a Yukawa + Vainshtein
    massive-gravity correction.

    Parameters
    ----------
    r_kpc : float
        Radius in kiloparsecs.
    v_baryonic : float
        Baryonic-only circular velocity in km/s.
    M_enclosed : float
        Enclosed baryonic mass within r (kg).

    Returns
    -------
    float
        Total circular velocity in km/s.
    """
    # Basic safety checks
    if r_kpc <= 0.0 or M_enclosed <= 0.0:
        return float(v_baryonic)

    # Radius in meters
    r_m = r_kpc * KPC_TO_M
    v_bary = float(v_baryonic)

    # --------------------------------------------------------------
    # 1. Vainshtein-like screening (same structure as best program)
    #    screening_factor = 1 + (r / R_VAIN)^SCREENING_POWER
    # --------------------------------------------------------------
    r_vain_m = R_VAIN_KPC * KPC_TO_M
    screening_factor = 1.0 + (r_m / r_vain_m) ** SCREENING_POWER

    # --------------------------------------------------------------
    # 2. Yukawa "core" term in SI units [m^2/s^2]
    #    Core structure matches the previous best:
    #      ~ G * M_enclosed / LAMBDA_G with a possible exponential
    #      suppression for r ≳ LAMBDA_G.
    # --------------------------------------------------------------
    ratio = r_m / LAMBDA_G

    if ratio < RATIO_THRESHOLD:
        # r << lambda: exponential ~ 1, scale-independent boost
        yukawa_core_SI = G * M_enclosed / LAMBDA_G
    else:
        # r ≳ lambda: include exponential Yukawa suppression
        exponential_factor = math.exp(-ratio)
        yukawa_core_SI = G * M_enclosed / LAMBDA_G * exponential_factor

    # Full Yukawa contribution in SI:
    #    v_yukawa_sq_SI ~ α * (G M / LAMBDA_G) * screening_factor
    v_yukawa_sq_SI = ALPHA_YUKAWA_DIMLESS * yukawa_core_SI * screening_factor

    # --------------------------------------------------------------
    # 3. Convert Yukawa term to (km/s)^2 and combine with baryonic v^2
    # --------------------------------------------------------------
    # 1 km^2/s^2 = 1e6 m^2/s^2  ->  divide by 1e6
    v_yukawa_sq_km2 = v_yukawa_sq_SI / 1.0e6

    if v_yukawa_sq_km2 < 0.0:
        v_yukawa_sq_km2 = 0.0

    v_total_sq = v_bary ** 2 + v_yukawa_sq_km2

    if v_total_sq <= 0.0:
        # Fallback: numerical guard
        return v_bary

    return math.sqrt(v_total_sq)
"""

with open(TARGET_SEED, "w", encoding="utf-8") as f:
    f.write(initial_program)
print(f"✅ Created initial_program.py")


# -----------------------------------------------------
# 2. evaluator.py (GALAXY_EVALUATOR)
# -----------------------------------------------------
evaluator = """
"""
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

def _sanitize_candidate_file(path: Path) -> None:
    """Strip Markdown fences if a candidate file was pasted with ``` blocks."""
    try:
        text = path.read_text(encoding="utf-8")
        if "```" in text:
            lines = [l for l in text.splitlines() if not l.strip().startswith("```")]
            path.write_text("\n".join(lines), encoding="utf-8")
    except Exception:
        # Non-fatal; just skip sanitization errors
        pass

def evaluate(program_path: str) -> dict:
    metrics = {}
    path = Path(program_path)
    _sanitize_candidate_file(path)

    # --- Dynamic import of candidate program ---
    try:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
    except Exception:
        return {"combined_score": 0.0}

    # Get the user's function
    calc_func = getattr(module, "calculate_rotation_velocity", None)
    if calc_func is None:
        return {"combined_score": 0.0}

    # --- EVALUATION LOOP ---
    try:
        # Fixed graviton mass scale (from your massive-gravity work)
        # m_g ~ 8.1e-69 kg. The AI can encode coupling / Yukawa scale internally.

        errors = []
        for r_kpc, v_obs, v_bar in zip(DATA_R_KPC, DATA_V_OBS, DATA_V_BAR):
            # r (kpc), v_baryonic (km/s), M_enclosed (kg)

            # Approx enclosed mass from baryonic velocity: v^2 = G M / r  ->  M = v^2 r / G
            r_m = r_kpc * KPC_TO_M
            v_bar_ms = v_bar * 1000.0
            M_enc = (v_bar_ms**2 * r_m) / G_NEWTON

            # Call candidate function (must return km/s)
            v_pred_kms = float(calc_func(r_kpc, v_bar, M_enc))

            # Relative error vs observed flat curve
            err = abs(v_pred_kms - v_obs) / v_obs
            errors.append(err)

        mean_error = float(np.mean(errors))

        # Score: 1.0 if error is 0; decreases as mean_error grows
        score = 1.0 / (1.0 + mean_error * 10.0)

        metrics["rotation_match"] = score
        metrics["combined_score"] = score
        metrics["stability"] = 1.0

    except Exception:
        metrics["combined_score"] = 0.0

    return metrics

def evaluate_stage1(p: str) -> dict:
    return evaluate(p)

def evaluate_stage2(p: str) -> dict:
    return evaluate(p)
"""

with open(TARGET_EVAL, "w", encoding="utf-8") as f:
    f.write(evaluator)
print(f"✅ Created evaluator.py")


# -----------------------------------------------------
# 3. config.yaml (GALAXY_CONFIG)
# -----------------------------------------------------
config_yaml = """
# OpenEvolve: The Dark Matter Killer
max_iterations: 100
checkpoint_interval: 10
log_level: "INFO"

llm:
  # Using your Cloud Setup for max intelligence
  primary_model: "qwen3-coder:480b-cloud"
  primary_model_weight: 1.0
  api_base: "http://localhost:11434/v1"
  api_key: "${OPENAI_API_KEY}"

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

  num_top_programs: 4
  use_template_stochasticity: true

database:
  # Moderate-sized population, enough diversity without huge cost
  population_size: 120
  archive_size: 40

  # Island model: balanced diversity vs convergence
  num_islands: 5
  migration_interval: 40      # migrate every ~40 iterations
  migration_rate: 0.12        # ~12% of each island's population migrates

  # Selection / exploitation as you already had
  elite_selection_ratio: 0.2  # top 20% as elites
  exploitation_ratio: 0.5     # half of new programs from exploiting elites

evaluator:
  timeout: 60
  cascade_evaluation: true
  parallel_evaluations: 1
  use_llm_feedback: false

evolution_settings:
  diff_based_evolution: true
  allow_full_rewrites: true
  max_code_length: 10000
"""

with open(TARGET_CONFIG, "w", encoding="utf-8") as f:
    f.write(config_yaml)
print(f"✅ Created config.yaml")


# -----------------------------------------------------
# 4. run_galaxy.bat (GALAXY_BAT)
# -----------------------------------------------------
batch = r"""@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
python -m openevolve.run --config examples\galaxy_rotation\config.yaml
pause
"""

with open(TARGET_BAT, "w", encoding="utf-8") as f:
    f.write(batch)
print(f"✅ Created run_galaxy.bat")

print("\n🚀 SETUP COMPLETE. Run the batch file 'run_galaxy.bat' to start the evolution.")