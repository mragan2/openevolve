"""
Hubble Tension Evaluator with Stage-Based Evaluation.

Demands a model that satisfies both Early (CMB) and Late (SN) H0 measurements.

Stages:
- evaluate_stage1: quick, cheap filter (late H0 + density only)
- evaluate_stage2: full evaluation (includes dynamical "tension_breaker")
- evaluate: alias for full evaluation (for non-cascade runs)
"""

import importlib.util
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from openevolve.evaluation_result import EvaluationResult

# ----------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------

# H0_EARLY (Planck): 67.4 km/s/Mpc -> ~2.184e-18 s^-1
# H0_LATE  (SH0ES):  73.0 km/s/Mpc -> ~2.365e-18 s^-1
H0_EARLY_SI = 2.184e-18
H0_LATE_SI = 2.365e-18

G_NEWTON = 6.67430e-11
M_G_REF = 8.1e-69  # Your reference massive graviton in kg

# Target: MG contribution ~ 0.7 * H0_LATE^2 at a=1
TARGET_OMEGA = 0.7  # conceptually, but we encode it via H^2 and rho
EPS = 1e-50


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _sanitize_candidate_file(path: Path) -> None:
    """
    Strip accidental Markdown ``` fences if a candidate file was pasted
    directly from a chat or notebook.
    """
    try:
        text = path.read_text(encoding="utf-8")
        if "```" in text:
            cleaned = "\n".join(
                line for line in text.splitlines()
                if not line.strip().startswith("```")
            )
            path.write_text(cleaned, encoding="utf-8")
    except Exception:
        # Non-fatal; better to try evaluation than to hard fail here.
        pass


def _load_module(program_path: str):
    """
    Dynamically import the candidate program as a module.
    Returns (module, error_msg_or_None).
    """
    path = Path(program_path)
    _sanitize_candidate_file(path)

    try:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module, None
    except Exception as e:
        return None, f"Import error: {e}"


def _safe_float(x):
    try:
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None


# ----------------------------------------------------------------------
# Core evaluation logic (shared by all stages)
# ----------------------------------------------------------------------

def _evaluate_internal(program_path: str, fast: bool, stage_name: str) -> EvaluationResult:
    """
    Shared internal evaluator.

    fast = True  -> Stage 1: quick filter (late H0 + density only).
    fast = False -> Stage 2 / full: also includes dynamical behavior metric.
    """
    start_time = time.time()

    # Base metrics structure; keeps keys consistent even on failure
    metrics = {
        "late_match": 0.0,
        "density_match": 0.0,
        "tension_breaker": 0.0,
        "combined_score": 0.0,
        "stability": 0.0,
    }
    artifacts = {
        "stage": stage_name,
    }

    # 1) Import candidate module
    module, import_err = _load_module(program_path)
    if module is None:
        eval_time = time.time() - start_time
        artifacts.update(
            {
                "stderr": import_err,
                "failure_stage": f"{stage_name}_import_error",
                "traceback": traceback.format_exc(),
                "execution_time": f"{eval_time:.2f}s",
            }
        )
        return EvaluationResult(metrics=metrics, artifacts=artifacts)

    # 2) Fetch required functions
    H_func = getattr(module, "H_mg_phenomenological", None)
    rho_func = getattr(module, "rho_quantum", None)

    missing = []
    if H_func is None:
        missing.append("H_mg_phenomenological")
    if rho_func is None:
        missing.append("rho_quantum")

    if missing:
        eval_time = time.time() - start_time
        artifacts.update(
            {
                "stderr": f"Missing required functions: {', '.join(missing)}",
                "failure_stage": f"{stage_name}_missing_functions",
                "execution_time": f"{eval_time:.2f}s",
            }
        )
        return EvaluationResult(metrics=metrics, artifacts=artifacts)

    # 3) Compute metrics safely
    try:
        # --- METRIC 1: LATE UNIVERSE MATCH (z = 0, a = 1) ---
        # We want MG contribution ~ 0.7 * H0_LATE^2
        H_today_raw = H_func(1.0, M_G_REF)
        H_today = _safe_float(H_today_raw)

        if H_today is None:
            raise ValueError(f"H_mg_phenomenological(1.0, M_G_REF) returned non-finite: {H_today_raw}")

        target_h_contribution = TARGET_OMEGA * (H0_LATE_SI ** 2)
        err_late = abs(H_today - target_h_contribution) / (abs(target_h_contribution) + EPS)

        # Sharpen penalty: if exact → 1, if off by factor of a few → small
        late_match = 1.0 / (1.0 + 10.0 * err_late)
        late_match = float(np.clip(late_match, 0.0, 1.0))
        metrics["late_match"] = late_match

        # --- METRIC 2: DENSITY MATCH AT a=1 ---
        # rho_quantum(a=1) should be ~ 0.7 * rho_crit(H0_LATE)
        rho_val_raw = rho_func(1.0, H0_LATE_SI, M_G_REF)
        rho_val = _safe_float(rho_val_raw)

        if rho_val is None:
            raise ValueError(f"rho_quantum(1.0, H0_LATE_SI, M_G_REF) returned non-finite: {rho_val_raw}")

        rho_crit_late = (3.0 * H0_LATE_SI ** 2) / (8.0 * np.pi * G_NEWTON)
        frac = abs(rho_val) / (rho_crit_late + EPS)

        # Gaussian around 0.7 with σ ~ 0.05
        density_match = np.exp(-((frac - 0.7) / 0.05) ** 2)
        density_match = float(np.clip(density_match, 0.0, 1.0))
        metrics["density_match"] = density_match

        # --- METRIC 3: DYNAMICAL BEHAVIOR ("Tension Breaker") ---
        # Only for full evaluations (stage2 / evaluate)
        if not fast:
            H_half_raw = H_func(0.5, M_G_REF)
            H_half = _safe_float(H_half_raw)

            if H_half is None:
                raise ValueError(f"H_mg_phenomenological(0.5, M_G_REF) returned non-finite: {H_half_raw}")

            ratio = H_today / (H_half + EPS)

            # If it's effectively constant (ratio ~ 1), penalize.
            if abs(ratio - 1.0) < 1e-5:
                tension_breaker = 0.5  # behaves like Λ
            else:
                tension_breaker = 1.0  # truly dynamical
            metrics["tension_breaker"] = float(np.clip(tension_breaker, 0.0, 1.0))
        else:
            metrics["tension_breaker"] = 0.0

        # --- COMBINED SCORE ---
        if fast:
            # Stage 1: emphasize fast checks (late & density)
            combined = 0.7 * metrics["late_match"] + 0.3 * metrics["density_match"]
        else:
            # Full evaluation: include dynamical behavior
            combined = (
                0.4 * metrics["late_match"]
                + 0.4 * metrics["density_match"]
                + 0.2 * metrics["tension_breaker"]
            )

        metrics["combined_score"] = float(np.clip(combined, 0.0, 1.0))
        metrics["stability"] = 1.0

        # --- Artifacts: diagnostics for you ---
        eval_time = time.time() - start_time
        artifacts.update(
            {
                "execution_time": f"{eval_time:.2f}s",
                "H_today": f"{H_today:.3e}",
                "target_H2_contribution": f"{target_h_contribution:.3e}",
                "rho_today": f"{rho_val:.3e}",
                "rho_crit_late": f"{rho_crit_late:.3e}",
                "rho_fraction": f"{frac:.3f}",
                "late_match_score": f"{metrics['late_match']:.3f}",
                "density_match_score": f"{metrics['density_match']:.3f}",
                "tension_breaker_score": f"{metrics['tension_breaker']:.3f}",
                "combined_score": f"{metrics['combined_score']:.3f}",
            }
        )

        if not fast:
            artifacts["note"] = "Full evaluation including dynamical behavior between a=0.5 and a=1.0"
        else:
            artifacts["note"] = "Fast stage-1 evaluation (no dynamical metric)"

        return EvaluationResult(metrics=metrics, artifacts=artifacts)

    except Exception as e:
        eval_time = time.time() - start_time
        artifacts.update(
            {
                "stderr": f"Metric computation failed in {stage_name}: {e}",
                "traceback": traceback.format_exc(),
                "failure_stage": f"{stage_name}_metric_computation",
                "execution_time": f"{eval_time:.2f}s",
            }
        )
        # metrics already zeroed except stability; keep stability=0.0 on hard failure
        return EvaluationResult(metrics=metrics, artifacts=artifacts)


# ----------------------------------------------------------------------
# Public API used by OpenEvolve
# ----------------------------------------------------------------------

def evaluate(program_path: str) -> EvaluationResult:
    """
    Full evaluation (same semantics as before).
    Called when cascade_evaluation is disabled or when a full score is needed.
    """
    return _evaluate_internal(program_path, fast=False, stage_name="full")


def evaluate_stage1(program_path: str) -> EvaluationResult:
    """
    Stage 1: quick, cheap filter.

    Intended for use with cascade_evaluation + cascade_thresholds in the config.
    Uses:
      - late_match
      - density_match
    to produce `combined_score`, so it's fast but still physically meaningful.
    """
    return _evaluate_internal(program_path, fast=True, stage_name="stage1")


def evaluate_stage2(program_path: str) -> EvaluationResult:
    """
    Stage 2: full evaluation (adds the dynamical behavior metric).
    """
    return _evaluate_internal(program_path, fast=False, stage_name="stage2")
