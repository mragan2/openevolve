"""
Galaxy Rotation Evaluator.
Tests if Massive Gravity can explain rotation curves WITHOUT Dark Matter.
"""
import importlib.util
import math
import sys
from pathlib import Path
import numpy as np
import traceback

# --- PHYSICAL CONSTANTS ---
G_NEWTON = 6.67430e-11
M_SUN = 1.989e30       # kg
KPC_TO_M = 3.086e19    # meters

# --- REAL GALAXY DATA (Synthetic NGC 6503 Approximation) ---
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
    except Exception as e:
        print("[EVALUATOR] Import error:", e)
        traceback.print_exc()
        return {"combined_score": 0.0}

    # Get the user's function
    calc_func = getattr(module, "calculate_rotation_velocity", None)
    if calc_func is None:
        print("[EVALUATOR] Missing function: calculate_rotation_velocity")
        return {"combined_score": 0.0}

    # --- EVALUATION LOOP ---
    try:
        errors = []
        for r_kpc, v_obs, v_bar in zip(DATA_R_KPC, DATA_V_OBS, DATA_V_BAR):
            r_m = r_kpc * KPC_TO_M
            v_bar_ms = v_bar * 1000.0
            M_enc = (v_bar_ms**2 * r_m) / G_NEWTON

            v_pred_kms = float(calc_func(r_kpc, v_bar, M_enc))
            err = abs(v_pred_kms - v_obs) / v_obs
            errors.append(err)

        mean_error = float(np.mean(errors))
        if not np.isfinite(mean_error):
            raise ValueError(f"Non-finite mean_error: {mean_error}")

        score = 1.0 / (1.0 + mean_error * 10.0)

        metrics["rotation_match"] = score
        metrics["combined_score"] = score
        metrics["stability"] = 1.0

    except Exception as e:
        print("[EVALUATOR] Runtime error:", e)
        traceback.print_exc()
        metrics["combined_score"] = 0.0

    return metrics

def evaluate_stage1(p: str) -> dict:
    return evaluate(p)

def evaluate_stage2(p: str) -> dict:
    return evaluate(p)
