"""
Physics Evaluator for Massive Graviton Cosmology.
"""
import importlib.util
import math
import sys
from pathlib import Path
import numpy as np

# --- TARGET CONSTANTS ---
TARGET_OMEGA_MG = 0.7
TARGET_LAMBDA = 1.1e-52
H0_SI = 2.2e-18
H0_SQ = H0_SI ** 2
M_G_REF = 8.1e-69 

# --- LCDM REFERENCE ---
OMEGA_M_DEFAULT = 0.3
OMEGA_R_DEFAULT = 9e-5
OMEGA_L_DEFAULT = 1.0 - OMEGA_M_DEFAULT - OMEGA_R_DEFAULT
G_NEWTON = 6.67430e-11

def H_LCDM_array(a):
    a = np.asarray(a, dtype=float)
    return H0_SI * np.sqrt(OMEGA_R_DEFAULT / a**4 + OMEGA_M_DEFAULT / a**3 + OMEGA_L_DEFAULT)

# --- METRICS ---
def compute_rho_q_today_score(rho_q_today, rho_crit_today):
    """
    Calculates score for Quantum Vacuum Density.
    Target: 0.7 (70% of critical density).
    """
    if not np.isfinite(rho_q_today) or rho_crit_today <= 0:
        return 0.0
    
    frac = abs(rho_q_today) / rho_crit_today
    target = 0.7
    width = 0.1
    
    return float(np.exp(-((frac - target) / width) ** 2))

def _sanitize_candidate_file(path):
    try:
        text = path.read_text(encoding="utf-8")
        if "```" in text:
            lines = [l for l in text.splitlines() if not l.strip().startswith("```")]
            path.write_text("\n".join(lines), encoding="utf-8")
    except: pass

# --- MAIN EVALUATOR ---
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

    # 1. Basic Metrics
    try:
        H_func = module.H_mg_phenomenological
        val_today = float(H_func(1.0, M_G_REF))
        target_val = 0.7 * H0_SQ
        err = abs(val_today - target_val) / target_val
        metrics["dark_energy_match"] = 1.0 / (1.0 + err)
    except: metrics["dark_energy_match"] = 0.0

    try:
        l_func = module.lambda_eff_from_mg
        val_l = float(l_func(M_G_REF))
        metrics["lambda_match"] = 1.0 / (1.0 + abs(math.log10(val_l/TARGET_LAMBDA)))
    except: metrics["lambda_match"] = 0.0

    metrics["stability"] = 1.0 

    # 2. Advanced Metrics (Rho Quantum)
    try:
        a_grid = np.logspace(-4, 0, 100)
        H_vals = H_LCDM_array(a_grid)
        rho_crit_vals = 3 * H_vals**2 / (8 * np.pi * G_NEWTON)
        
        rho_q_fn = getattr(module, "rho_quantum", None)
        if rho_q_fn:
            # Calculate for today (last element)
            rho_q_today = float(rho_q_fn(1.0, H_vals[-1], M_G_REF))
            metrics["rho_q_today_score"] = compute_rho_q_today_score(rho_q_today, rho_crit_vals[-1])
        else:
            metrics["rho_q_today_score"] = 0.0
    except:
        metrics["rho_q_today_score"] = 0.0

    # 3. Combined Score
    score = (
        0.18 * metrics.get("dark_energy_match", 0) +
        0.12 * metrics.get("lambda_match", 0) +
        0.10 * metrics.get("stability", 0) +
        0.40 * metrics.get("rho_q_today_score", 0) +
        0.20 * 1.0 # Bonus
    )
    metrics["combined_score"] = float(np.clip(score, 0.0, 1.0))
    return metrics

def evaluate_stage1(p): return evaluate(p)
def evaluate_stage2(p): return evaluate(p)