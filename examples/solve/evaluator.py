"""
Evaluator for missing-term reconstruction in H(a).

This version assumes each candidate is a FULL Python module
defining at least:

    - H_LCDM(a, H0, Omega_m, Omega_r, Omega_L)
    - correction_term(a)
    - prediction(a, H0, Omega_m, Omega_r, Omega_L)

It does NOT do any EVOLVE-block stitching.

It compares prediction(a) to a synthetic "true" H(a) built from
ΛCDM plus a hidden correction term δ_true(a), and computes:

    - fit_score
    - early_suppression_score
    - smoothness_score
    - curvature_score
    - delta_today_score
    - combined_score

All metrics are in [0, 1].
"""

from __future__ import annotations

import importlib.util
import math
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np

# ----------------------------------------------------------------------
# Required functions
# ----------------------------------------------------------------------

REQUIRED_FUNCTIONS = ["H_LCDM", "correction_term", "prediction"]

# Default cosmological parameters (should match initial_program.py)
H0_DEFAULT = 2.2e-18
OMEGA_M_DEFAULT = 0.3
OMEGA_R_DEFAULT = 9e-5
OMEGA_L_DEFAULT = 1.0 - OMEGA_M_DEFAULT - OMEGA_R_DEFAULT

# Metric scales
FIT_TOL = 0.02
EARLY_SUPPRESSION_SCALE = 0.02
SMOOTHNESS_SCALE = 0.05
CURVATURE_SCALE = 0.05
DELTA_TODAY_TARGET = 0.03
DELTA_TODAY_TOL = 0.02


# ----------------------------------------------------------------------
# Candidate loading
# ----------------------------------------------------------------------

def _load_candidate_namespace(program_path: str) -> SimpleNamespace:
    """
    Import candidate module as-is (no EVOLVE stitching).
    """
    path = Path(program_path)
    if not path.is_file():
        raise FileNotFoundError(f"Candidate program not found: {path}")

    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create spec for {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return SimpleNamespace(**module.__dict__)


def _functions_present_score(ns: Any) -> float:
    have = 0
    for name in REQUIRED_FUNCTIONS:
        if hasattr(ns, name) and callable(getattr(ns, name)):
            have += 1
    return have / float(len(REQUIRED_FUNCTIONS))


# ----------------------------------------------------------------------
# Helpers for calling candidate functions
# ----------------------------------------------------------------------

def _call_H_LCDM(ns: Any, a: np.ndarray) -> np.ndarray:
    H_LCDM = getattr(ns, "H_LCDM")
    return np.asarray(
        H_LCDM(a, H0_DEFAULT, OMEGA_M_DEFAULT, OMEGA_R_DEFAULT, OMEGA_L_DEFAULT),
        dtype=float,
    )


def _call_prediction(ns: Any, a: np.ndarray) -> np.ndarray:
    prediction = getattr(ns, "prediction")
    try:
        return np.asarray(
            prediction(a, H0_DEFAULT, OMEGA_M_DEFAULT, OMEGA_R_DEFAULT, OMEGA_L_DEFAULT),
            dtype=float,
        )
    except TypeError:
        # If candidate defined 1-arg prediction(a)
        return np.asarray(prediction(a), dtype=float)


def _call_correction(ns: Any, a: np.ndarray,
                     base: np.ndarray, pred: np.ndarray) -> np.ndarray:
    if hasattr(ns, "correction_term") and callable(getattr(ns, "correction_term")):
        try:
            corr = ns.correction_term(a)
            return np.asarray(corr, dtype=float)
        except Exception:
            pass
    # Fallback: infer from ratio
    base_safe = np.where(base == 0.0, 1e-30, base)
    return pred / base_safe - 1.0


# ----------------------------------------------------------------------
# Synthetic "truth" curve
# ----------------------------------------------------------------------

def _delta_true(a: np.ndarray) -> np.ndarray:
    """
    Hidden correction term that defines the synthetic truth model.
    Small, smooth, monotonic, a few percent today.
    """
    a = np.asarray(a, dtype=float)
    term1 = 0.02 * (a ** 2) / (1.0 + a ** 2)
    term2 = 0.01 * (a ** 3) / (1.0 + a ** 3)
    term3 = 0.002 * np.log1p(a)
    return term1 + term2 + term3


def _build_truth(ns: Any, a: np.ndarray) -> np.ndarray:
    base = _call_H_LCDM(ns, a)
    delta = _delta_true(a)
    return base * (1.0 + delta)


# ----------------------------------------------------------------------
# Metric bundle
# ----------------------------------------------------------------------

def _full_metric_bundle(ns: Any, fps: float) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "functions_present_score": float(fps),
        "fit_score": 0.0,
        "early_suppression_score": 0.0,
        "smoothness_score": 0.0,
        "curvature_score": 0.0,
        "delta_today_score": 0.0,
        "combined_score": 0.0,
    }

    if fps < 1.0:
        return metrics

    # Grid in a
    a = np.linspace(0.05, 1.0, 100)

    try:
        base = _call_H_LCDM(ns, a)
        pred = _call_prediction(ns, a)
        delta = _call_correction(ns, a, base, pred)
    except Exception as exc:
        metrics["error"] = f"Failed to evaluate candidate: {exc!r}"
        return metrics

    try:
        base = base.reshape(a.shape)
        pred = pred.reshape(a.shape)
        delta = delta.reshape(a.shape)
    except Exception as exc:
        metrics["error"] = f"Shape mismatch in candidate outputs: {exc!r}"
        return metrics

    # 1) Fit score: rms fractional error vs truth
    try:
        H_true = _build_truth(ns, a)
        denom = np.maximum(np.abs(H_true), 1e-30)
        frac_err = (pred - H_true) / denom
        rms = math.sqrt(float(np.mean(frac_err ** 2)))
        metrics["fit_score"] = 1.0 / (1.0 + rms / FIT_TOL)
    except Exception:
        metrics["fit_score"] = 0.0

    # 2) Early-time suppression (a <= 0.2)
    try:
        mask = (a >= 0.05) & (a <= 0.2)
        if np.any(mask):
            delta_early = delta[mask]
            max_abs = float(np.max(np.abs(delta_early)))
            metrics["early_suppression_score"] = 1.0 / (
                1.0 + max_abs / EARLY_SUPPRESSION_SCALE
            )
        else:
            metrics["early_suppression_score"] = 0.0
    except Exception:
        metrics["early_suppression_score"] = 0.0

    # 3) Smoothness (first derivative)
    try:
        d_delta = np.diff(delta) / np.diff(a)
        tv = float(np.mean(np.abs(d_delta)))
        metrics["smoothness_score"] = 1.0 / (1.0 + tv / SMOOTHNESS_SCALE)
    except Exception:
        metrics["smoothness_score"] = 0.0

    # 4) Curvature (second derivative)
    try:
        d2 = np.diff(delta, n=2) / (np.diff(a)[:-1] ** 2)
        avg_curv = float(np.mean(np.abs(d2)))
        metrics["curvature_score"] = 1.0 / (1.0 + avg_curv / CURVATURE_SCALE)
    except Exception:
        metrics["curvature_score"] = 0.0

    # 5) Delta_today
    try:
        a1 = 1.0
        base1 = float(_call_H_LCDM(ns, np.array([a1]))[0])
        pred1 = float(_call_prediction(ns, np.array([a1]))[0])
        delta1 = (pred1 / max(base1, 1e-30)) - 1.0
        err = abs(delta1 - DELTA_TODAY_TARGET)
        metrics["delta_today_score"] = 1.0 / (1.0 + err / DELTA_TODAY_TOL)
    except Exception:
        metrics["delta_today_score"] = 0.0

    # Combined score
    weights = dict(
        functions_present_score=0.05,
        fit_score=0.40,
        early_suppression_score=0.15,
        smoothness_score=0.15,
        curvature_score=0.15,
        delta_today_score=0.10,
    )
    num = 0.0
    den = 0.0
    for k, w in weights.items():
        num += metrics.get(k, 0.0) * w
        den += w
    metrics["combined_score"] = float(max(0.0, min(1.0, num / max(den, 1e-12))))

    return metrics


# ----------------------------------------------------------------------
# Stage 1 / Stage 2 / default evaluate
# ----------------------------------------------------------------------

def evaluate_stage1(program_path: str) -> Dict[str, float]:
    """
    Light structural check: can we import and are required functions present?
    """
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


def evaluate_stage2(program_path: str) -> Dict[str, float]:
    """
    Full physics evaluation (normal evolution mode).
    """
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
            "fit_score": 0.0,
            "early_suppression_score": 0.0,
            "smoothness_score": 0.0,
            "curvature_score": 0.0,
            "delta_today_score": 0.0,
            "combined_score": 0.0,
            "eval_time": float(t1 - t0),
            "error": f"{exc}\n{traceback.format_exc()}",
        }


def evaluate(program_path: str) -> Dict[str, float]:
    """
    Default entry for OpenEvolve when cascade_evaluation is off.
    """
    return evaluate_stage2(program_path)


__all__ = ["evaluate", "evaluate_stage1", "evaluate_stage2"]