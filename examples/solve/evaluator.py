"""
Evaluator for discovering missing terms in H(a) using OpenEvolve.

This evaluator expects candidate programs that:
  - Define a baseline ΛCDM H(a)
  - Contain an EVOLVE block with a function:
        correction_term(a)
  - Define a public function:
        prediction(a)

Evolution modifies ONLY correction_term(a).

This evaluator:
  • Stitches EVOLVE blocks into the locked scaffold initial_program.py
  • Loads the resulting module
  • Checks function existence
  • Computes physics-based metrics:
        - Fit to synthetic or real H(a) data
        - Early-time suppression
        - Smoothness (first derivative)
        - Curvature (second derivative)
        - Smallness of correction at a=1
        - Boundedness penalties
  • Combines metrics into a single score
"""

from __future__ import annotations

import importlib.util
import logging
import math
import numpy as np
import os
import tempfile
import traceback
import uuid
import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any

LOGGER = logging.getLogger(__name__)

# ============================================================
# Required function(s)
# ============================================================

REQUIRED_FUNCTIONS = [
    "prediction",      # full H(a) = H_LCDM(a)*(1+δ(a))
]

# ============================================================
# EVOLVE block extraction
# ============================================================

def _extract_evolve_block(source: str) -> str:
    start_tag = "# EVOLVE-BLOCK-START"
    end_tag = "# EVOLVE-BLOCK-END"

    lines = source.splitlines()
    start_idx = None
    end_idx = None

    for i, line in enumerate(lines):
        if line.strip() == start_tag:
            start_idx = i
        if line.strip() == end_tag:
            end_idx = i
            break

    if start_idx is None or end_idx is None or end_idx <= start_idx:
        raise ValueError("Missing EVOLVE block markers")

    block = "\n".join(lines[start_idx + 1 : end_idx]).strip()
    if not block:
        raise ValueError("EVOLVE block is empty")

    return block


def _sanitize_user_block(block: str) -> str:
    out = []
    for line in block.splitlines():
        if line.strip().startswith("from __future__ import"):
            continue
        out.append(line)
    result = "\n".join(out).strip()
    if not result:
        raise ValueError("EVOLVE block empty after sanitization")
    return result


def _build_locked_program(scaffold_source: str, user_block: str) -> str:
    out = []
    in_block = False

    for line in scaffold_source.splitlines():
        stripped = line.strip()

        if stripped == "# EVOLVE-BLOCK-START":
            out.append(line)
            out.append(user_block)
            in_block = True
            continue

        if stripped == "# EVOLVE-BLOCK-END" and in_block:
            out.append(line)
            in_block = False
            continue

        if not in_block:
            out.append(line)

    return "\n".join(out)


# ============================================================
# Candidate loading
# ============================================================

def _execute_stitched(full_source: str) -> SimpleNamespace:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tmp:
        tmp.write(full_source)
        tmp_path = Path(tmp.name)

    try:
        g = runpy.run_path(str(tmp_path))
        return SimpleNamespace(**g)
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def _import_full(program_path: str) -> SimpleNamespace:
    name = f"candidate_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, program_path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Failed spec for {program_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return SimpleNamespace(**module.__dict__)


def _load_candidate_namespace(program_path: str) -> SimpleNamespace:
    full_path = Path(program_path)
    source = full_path.read_text(encoding="utf-8")

    scaffold_path = Path(__file__).parent / "initial_program.py"

    try:
        block = _extract_evolve_block(source)
        block = _sanitize_user_block(block)
        scaffold_source = scaffold_path.read_text(encoding="utf-8")
        full_source = _build_locked_program(scaffold_source, block)
        return _execute_stitched(full_source)
    except ValueError:
        # No EVOLVE block → full module
        return _import_full(str(full_path))


# ============================================================
# Function presence scoring
# ============================================================

def _functions_present_score(ns: Any) -> float:
    have = 0
    for fn in REQUIRED_FUNCTIONS:
        if hasattr(ns, fn) and callable(getattr(ns, fn)):
            have += 1
    return have / len(REQUIRED_FUNCTIONS)


# ============================================================
# Physics-inspired scoring
# ============================================================

H0_REF = 70.0

# synthetic scale factor grid for testing
A_GRID = np.linspace(0.05, 1.0, 30)

# synthetic “true” H(a) assuming ΛCDM only
def H_LCDM(a):
    Omega_m = 0.3
    Omega_r = 9e-5
    Omega_L = 1.0 - Omega_m - Omega_r
    return H0_REF * np.sqrt(Omega_r * a**(-4) +
                            Omega_m * a**(-3) +
                            Omega_L)

H_TRUE = H_LCDM(A_GRID)
H_ERR = 0.05 * H_TRUE  # nominal 5% uncertainty


def _physics_metrics(ns: Any) -> Dict[str, float]:
    metrics = {
        "fit_score": 0.0,
        "early_suppression_score": 0.0,
        "smoothness_score": 0.0,
        "curvature_score": 0.0,
        "delta_today_score": 0.0,
        "combined_score": 0.0,
    }

    try:
        pred = ns.prediction(A_GRID)
    except Exception:
        return metrics

    # ------------------------------------------------------------
    # Fit score: χ²-like
    # ------------------------------------------------------------
    chi2 = np.sum(((pred - H_TRUE) / H_ERR)**2)
    metrics["fit_score"] = 1.0 / (1.0 + chi2 / len(A_GRID))

    # ------------------------------------------------------------
    # Early suppression: correction small at a=0.05
    # ------------------------------------------------------------
    try:
        d0 = (pred[0] - H_TRUE[0]) / H_TRUE[0]
        metrics["early_suppression_score"] = 1.0 / (1.0 + 50.0 * abs(d0))
    except Exception:
        pass

    # ------------------------------------------------------------
    # Smoothness (first derivative)
    # ------------------------------------------------------------
    try:
        slope = np.gradient(pred, A_GRID)
        metrics["smoothness_score"] = 1.0 / (1.0 + np.std(slope) / np.mean(np.abs(slope) + 1e-9))
    except Exception:
        pass

    # ------------------------------------------------------------
    # Curvature (second derivative)
    # ------------------------------------------------------------
    try:
        curv = np.gradient(np.gradient(pred, A_GRID), A_GRID)
        metrics["curvature_score"] = 1.0 / (1.0 + np.mean(np.abs(curv)) / np.mean(pred))
    except Exception:
        pass

    # ------------------------------------------------------------
    # delta(a=1) should be small unless data demands otherwise
    # ------------------------------------------------------------
    try:
        d1 = (pred[-1] - H_TRUE[-1]) / H_TRUE[-1]
        metrics["delta_today_score"] = 1.0 / (1.0 + abs(d1))
    except Exception:
        pass

    # ------------------------------------------------------------
    # Combined score
    # ------------------------------------------------------------

    weights = dict(
        fit_score=0.50,
        early_suppression_score=0.10,
        smoothness_score=0.15,
        curvature_score=0.15,
        delta_today_score=0.10,
    )

    num = sum(metrics[k] * w for k, w in weights.items())
    den = sum(weights.values())
    metrics["combined_score"] = max(0.0, min(1.0, num / den))

    return metrics


# ============================================================
# Stage 1 evaluation
# ============================================================

def evaluate_stage1(program_path: str) -> Dict[str, float]:
    try:
        ns = _load_candidate_namespace(program_path)
        fps = _functions_present_score(ns)

        return {
            "syntax_valid": 1.0,
            "module_loaded": 1.0,
            "functions_present_score": fps,
            "combined_score": fps,
        }

    except Exception as exc:
        return {
            "syntax_valid": 0.0,
            "module_loaded": 0.0,
            "functions_present_score": 0.0,
            "combined_score": 0.0,
            "error": f"{exc}\n{traceback.format_exc()}",
        }


# ============================================================
# Stage 2 evaluation (full physics)
# ============================================================

def evaluate_stage2(program_path: str) -> Dict[str, float]:
    try:
        ns = _load_candidate_namespace(program_path)
        fps = _functions_present_score(ns)

        metrics = _physics_metrics(ns)
        metrics.update({
            "syntax_valid": 1.0,
            "module_loaded": 1.0,
            "functions_present_score": fps,
        })
        return metrics

    except Exception as exc:
        return {
            "syntax_valid": 0.0,
            "module_loaded": 0.0,
            "functions_present_score": 0.0,
            "combined_score": 0.0,
            "error": f"{exc}\n{traceback.format_exc()}",
        }


# ============================================================
# Compatibility entry point
# ============================================================

def evaluate(program_path: str) -> Dict[str, float]:
    return evaluate_stage2(program_path)


__all__ = ["evaluate", "evaluate_stage1", "evaluate_stage2"]
