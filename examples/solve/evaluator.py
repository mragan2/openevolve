"""
Physics Evaluator for Massive Graviton Theory.
Checks consistency with Dark Energy observations and dimensional analysis.
"""

import importlib.util
import math
import sys
from pathlib import Path
import numpy as np

# Targets
TARGET_OMEGA_MG = 0.7
TARGET_LAMBDA = 1.1e-52
H0_SI = 2.2e-18
H0_SQ = H0_SI**2

# CRITICAL UPDATE: Match the scaffold
M_G_REF = 8.1e-69


def evaluate(program_path):
    """
    Imports the AI's code and scores it based on physical consistency.
    """
    metrics = {
        "combined_score": 0.0,
        "dark_energy_match": 0.0,
        "lambda_match": 0.0,
        "stability": 0.0,
    }

    # 1. Load the Candidate Module
    try:
        path = Path(program_path)
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = module
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"Import Failed: {e}")
        return metrics  # Score 0

    # 2. Extract Functions
    try:
        H_mg_func = module.H_mg_phenomenological
        lambda_func = module.lambda_eff_from_mg
    except AttributeError:
        print("Missing required functions in EVOLVE block.")
        return metrics

    # 3. Test 1: Dark Energy Match at a=1 (Present Day)
    # The graviton contribution H_mg should be approx 0.7 * H0^2
    try:
        val_at_today = float(H_mg_func(1.0, M_G_REF, H0_SI))

        target_val = TARGET_OMEGA_MG * H0_SQ

        # Avoid division by zero or massive overflow
        if math.isnan(val_at_today) or math.isinf(val_at_today):
            raise ValueError("Infinity/NaN")

        # Score based on fractional error
        error_H = abs(val_at_today - target_val) / (target_val + 1e-30)
        score_H = 1.0 / (1.0 + error_H)
        metrics["dark_energy_match"] = score_H

    except Exception as e:
        metrics["dark_energy_match"] = 0.0

    # 4. Test 2: Lambda Value Match
    # Should be approx 1.1e-52 m^-2
    try:
        val_lambda = float(lambda_func(M_G_REF))

        # Score based on log-scale error
        if val_lambda <= 0:
            score_L = 0.0
        else:
            log_diff = abs(math.log10(val_lambda) - math.log10(TARGET_LAMBDA))
            score_L = 1.0 / (1.0 + log_diff)

        metrics["lambda_match"] = score_L

    except Exception:
        metrics["lambda_match"] = 0.0

    # 5. Test 3: Stability / Smoothness (Evolution check)
    try:
        val_past = float(H_mg_func(0.5, M_G_REF, H0_SI))
        ratio = abs(val_past) / H0_SQ
        if ratio > 1000.0:
            metrics["stability"] = 0.1
        elif math.isnan(val_past):
            metrics["stability"] = 0.0
        else:
            metrics["stability"] = 1.0
    except:
        metrics["stability"] = 0.0

    # Final Weighted Score
    metrics["combined_score"] = (
        0.5 * metrics["dark_energy_match"]
        + 0.4 * metrics["lambda_match"]
        + 0.1 * metrics["stability"]
    )

    return metrics
