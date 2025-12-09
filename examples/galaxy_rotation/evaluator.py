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

def _sanitize_candidate_file(path):
    try:
        text = path.read_text(encoding="utf-8")
        if "```" in text:
            lines = [l for l in text.splitlines() if not l.strip().startswith("```")]
            path.write_text("\n".join(lines), encoding="utf-8")
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
