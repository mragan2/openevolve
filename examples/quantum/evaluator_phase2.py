"""
Standalone evaluator for the second phase of semiclassical cosmology evolution.
This version includes all missing helpers so the evaluator is completely functional.
"""

from __future__ import annotations

import ast
import logging
import math
import os
import runpy
import tempfile
import textwrap
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List


# ============================================================
# Required user-implemented functions in evolved program
# ============================================================

REQUIRED_FUNCTIONS = [
    "rho_quantum",
    "H_squared_with_quantum",
    "run_sanity_checks",
]


# ============================================================
# Environment helpers
# ============================================================

def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _bool_env(name: str) -> bool:
    return os.environ.get(name, "0").lower() in {"1", "true", "yes", "on"}


LOGGER = logging.getLogger(__name__)
LOG_LOW_SCORE_DETAILS = _bool_env("OPENEVOLVE_LOG_LOW_SCORES")
LOW_SCORE_THRESHOLD = _float_env("OPENEVOLVE_LOW_SCORE_THRESHOLD", 0.4)

H0_RATIO_TOLERANCE = max(_float_env("OPENEVOLVE_H0_RATIO_TOLERANCE", 1.5), 1e-6)
RHO_Q_TODAY_TOLERANCE = max(_float_env("OPENEVOLVE_RHO_Q_TOLERANCE", 2.0), 1e-6)

TARGET_RHO_Q_FRACTION = _float_env("OPENEVOLVE_TARGET_RHO_Q", 0.03)
TARGET_RHO_Q_TOLERANCE = max(_float_env("OPENEVOLVE_TARGET_RHO_Q_TOLERANCE", 0.02), 1e-6)


# ============================================================
# Core helper: function presence score
# ============================================================

def _functions_present_score(ns: Dict[str, Any]) -> float:
    """Score fraction of required functions present in namespace."""
    present = 0
    for fn in REQUIRED_FUNCTIONS:
        if fn in ns and callable(ns[fn]):
            present += 1
    return present / len(REQUIRED_FUNCTIONS)


# ============================================================
# Namespace loader (safe)
# ============================================================

def _load_candidate_namespace(program_path: str) -> Dict[str, Any]:
    """
    Execute candidate program in an isolated namespace and return globals.
    """
    full_path = str(Path(program_path).resolve())
    try:
        return runpy.run_path(full_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to import candidate program: {exc}")


# ============================================================
# Run sanity metrics from candidate's run_sanity_checks()
# ============================================================

def _run_sanity_metrics(ns: Dict[str, Any]) -> Dict[str, float]:
    if "run_sanity_checks" not in ns:
        raise RuntimeError("Function run_sanity_checks not found in namespace")
    try:
        result = ns["run_sanity_checks"]()
        if not isinstance(result, dict):
            raise ValueError("run_sanity_checks must return a dict")
        return result
    except Exception as exc:
        raise RuntimeError(f"Error running sanity checks: {exc}")


# ============================================================
# Monotonic H(a) score
# ============================================================

def _monotonic_H_score(ns: Dict[str, Any]) -> float:
    """Reward monotonic decrease of H(a) from a=0.05 to a=1."""

    if "H_squared_with_quantum" not in ns or "CosmologyParams" not in ns:
        return 0.0

    H2_fn = ns["H_squared_with_quantum"]
    Params = ns["CosmologyParams"]
    params = Params()

    a_grid = [0.05 + i * (0.95 / 20) for i in range(21)]
    H_vals = []

    for a in a_grid:
        try:
            H2 = H2_fn(a, params)
            H_vals.append(math.sqrt(max(H2, 0)))
        except Exception:
            return 0.0

    # Check monotonic decreasing: H[i] >= H[i+1]
    score = 1.0
    drops = 0
    for i in range(len(H_vals) - 1):
        if H_vals[i] < H_vals[i + 1]:
            drops += 1

    if drops == 0:
        return 1.0
    return max(0.0, 1.0 - drops / len(H_vals))


# ============================================================
# Variation score: encourage non-flat rho_q(a)
# ============================================================

def _rho_profile_variation_score(ns: Dict[str, Any]) -> float:
    """Reward non-trivial variation in rho_quantum(a)."""

    if "rho_quantum" not in ns:
        return 0.0

    rho_fn = ns["rho_quantum"]
    if "classical_H_squared" not in ns or "CosmologyParams" not in ns:
        return 0.0

    H2_classical = ns["classical_H_squared"]
    Params = ns["CosmologyParams"]
    params = Params()

    a_grid = [0.05 + i * (0.95 / 20) for i in range(21)]
    values = []

    for a in a_grid:
        try:
            H2 = H2_classical(a, params)
            H = math.sqrt(max(H2, 0))
            rho = rho_fn(a, H, params.m_g)
            values.append(rho)
        except Exception:
            return 0.0

    if len(values) < 2:
        return 0.0

    variation = max(values) - min(values)
    if variation <= 0:
        return 0.0

    # Normalize by mean
    mean_val = sum(values) / len(values)
    if mean_val <= 0:
        return 1.0

    return min(1.0, variation / (mean_val * 5))


# ============================================================
# Low-score logging
# ============================================================

def _log_low_score_details(checks: Dict[str, float], metrics: Dict[str, float]):
    if not LOG_LOW_SCORE_DETAILS:
        return
    if metrics["combined_score"] >= LOW_SCORE_THRESHOLD:
        return

    LOGGER.warning("LOW SCORE METRICS:")
    LOGGER.warning("Sanity Checks: %s", checks)
    LOGGER.warning("Metrics: %s", metrics)


# ============================================================
# Full metric bundle (Phase 2 scoring)
# ============================================================

def _full_metric_bundle(ns: Dict[str, Any], functions_present_score: float) -> Dict[str, float]:
    checks = _run_sanity_metrics(ns)

    # H0 ratio close to unity
    H0_ratio_delta = abs(checks["ratio_H0"] - 1.0)
    H0_ratio_score = max(0.0, 1.0 - H0_ratio_delta / H0_RATIO_TOLERANCE)

    # Original small-quantum penalty
    rho_today = max(checks["rho_q_today_over_crit0"], 0.0)
    rho_q_today_penalty = min(rho_today / RHO_Q_TODAY_TOLERANCE, 1.0)
    rho_q_today_score = max(0.0, 1.0 - rho_q_today_penalty)

    # Early domination: H(a) > 0
    early_domination_score = 1.0 if checks["H_at_early_a"] > 0.0 else 0.0

    # Reuse as small-early score
    quantum_small_early_score = rho_q_today_score

    monotonic_H_score = _monotonic_H_score(ns)
    rho_profile_variation_score = _rho_profile_variation_score(ns)

    # New Phase 2 target reward
    rho_q_target_diff = abs(rho_today - TARGET_RHO_Q_FRACTION)
    rho_q_target_score = max(0.0, 1.0 - rho_q_target_diff / TARGET_RHO_Q_TOLERANCE)

    combined_score = (
        0.10 * functions_present_score
        + 0.15 * H0_ratio_score
        + 0.10 * rho_q_today_score
        + 0.10 * early_domination_score
        + 0.20 * monotonic_H_score
        + 0.10 * rho_profile_variation_score
        + 0.25 * rho_q_target_score
    )
    combined_score = float(max(0.0, min(1.0, combined_score)))

    metrics = {
        "functions_present_score": float(functions_present_score),
        "H0_ratio_score": float(H0_ratio_score),
        "rho_q_today_score": float(rho_q_today_score),
        "early_domination_score": float(early_domination_score),
        "quantum_small_early_score": float(quantum_small_early_score),
        "monotonic_H_score": float(monotonic_H_score),
        "rho_profile_variation_score": float(rho_profile_variation_score),
        "rho_q_target_score": float(rho_q_target_score),
        "combined_score": float(combined_score),
    }

    _log_low_score_details(checks, metrics)
    return metrics


# ============================================================
# Stage 1 Evaluation
# ============================================================

def evaluate_stage1(program_path: str) -> Dict[str, float]:
    t0 = time.time()
    try:
        ns = _load_candidate_namespace(program_path)
        fps = _functions_present_score(ns)
        t1 = time.time()
        return {
            "syntax_valid": 1.0,
            "module_loaded": 1.0,
            "functions_present_score": float(fps),
            "combined_score": float(fps),
            "eval_time": float(t1 - t0),
        }
    except Exception as exc:
        t1 = time.time()
        return {
            "syntax_valid": 0.0,
            "module_loaded": 0.0,
            "functions_present_score": 0.0,
            "combined_score": 0.0,
            "eval_time": float(t1 - t0),
            "error": f"{exc}\n{traceback.format_exc()}",
        }


# ============================================================
# Stage 2 Evaluation (full physics)
# ============================================================

def evaluate_stage2(program_path: str) -> Dict[str, float]:
    t0 = time.time()
    try:
        ns = _load_candidate_namespace(program_path)
        fps = _functions_present_score(ns)
        metrics = _full_metric_bundle(ns, fps)
        t1 = time.time()
        metrics.update(
            {
                "syntax_valid": 1.0,
                "module_loaded": 1.0,
                "eval_time": float(t1 - t0),
            }
        )
        return metrics
    except Exception as exc:
        t1 = time.time()
        return {
            "syntax_valid": 0.0,
            "module_loaded": 0.0,
            "functions_present_score": 0.0,
            "H0_ratio_score": 0.0,
            "rho_q_today_score": 0.0,
            "early_domination_score": 0.0,
            "quantum_small_early_score": 0.0,
            "monotonic_H_score": 0.0,
            "rho_profile_variation_score": 0.0,
            "combined_score": 0.0,
            "eval_time": float(t1 - t0),
            "error": f"{exc}\n{traceback.format_exc()}",
        }


# ============================================================
# Default entry point for OpenEvolve
# ============================================================

def evaluate(program_path: str) -> Dict[str, float]:
    return evaluate_stage2(program_path)


__all__ = [
    "evaluate",
    "evaluate_stage1",
    "evaluate_stage2",
]
