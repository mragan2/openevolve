"""
Evaluator for missing-term discovery using a massive-graviton H(a) target.

This expects candidate programs (or EVOLVE blocks) that define:

    def prediction(a: ArrayLike) -> ArrayLike:

where:
    prediction(a) = H_LCDM(a) * (1.0 + correction_term(a))

The evaluator:

  * Stitches EVOLVE blocks into the locked scaffold initial_program.py
  * Loads the resulting module as a namespace
  * Checks that prediction(a) exists
  * Computes physics-based metrics by comparing prediction(a)
    to a synthetic H_target(a) generated from a phenomenological
    massive-graviton H(a) with:

        m_g = 8.012922413646382e-70 kg

The goal of evolution is to discover a correction_term(a) that acts
as an effective "missing term" reproducing the massive-graviton H(a)
while remaining small, smooth, and physically plausible.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import numpy as np
import os
import runpy
import tempfile
import traceback
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any

LOGGER = logging.getLogger(__name__)

# ============================================================
# Required function(s)
# ============================================================

REQUIRED_FUNCTIONS = [
    "prediction",
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
# Massive-graviton target H(a)
# ============================================================

M_G_TRUE = 8.012922413646382e-70
H0_TRUE = 2.2e-18

OMEGA_M = 0.3
OMEGA_R = 9.0e-5
OMEGA_L = 1.0 - OMEGA_M - OMEGA_R

A_GRID = np.linspace(0.05, 1.0, 30)


def H_LCDM(a: np.ndarray, H0: float = H0_TRUE) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return H0 * np.sqrt(
        OMEGA_R * a**(-4) +
        OMEGA_M * a**(-3) +
        OMEGA_L
    )


def H_mg_phenomenological(a: np.ndarray,
                          m_g: float = M_G_TRUE,
                          H0: float = H0_TRUE) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    H_lcdm = H_LCDM(a, H0=H0)

    eps_mg = m_g / M_G_TRUE
    bump = 0.05 * eps_mg * np.exp(-((1.0 - a)**2) / 0.1)

    H_sq = H_lcdm**2 * (1.0 + bump)
    return np.sqrt(np.maximum(H_sq, 0.0))


H_TARGET = H_mg_phenomenological(A_GRID, m_g=M_G_TRUE, H0=H0_TRUE)
H_ERR = 0.05 * H_TARGET


# ============================================================
# Physics-inspired scoring
# ============================================================

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
        pred = np.asarray(pred, dtype=float)
    except Exception:
        return metrics

    chi2 = np.sum(((pred - H_TARGET) / H_ERR)**2)
    metrics["fit_score"] = 1.0 / (1.0 + chi2 / len(A_GRID))

    try:
        d0 = (pred[0] - H_LCDM(A_GRID[0])) / H_LCDM(A_GRID[0])
        metrics["early_suppression_score"] = 1.0 / (1.0 + 50.0 * abs(d0))
    except Exception:
        pass

    try:
        slope = np.gradient(pred, A_GRID)
        metrics["smoothness_score"] = 1.0 / (
            1.0 + np.std(slope) / (np.mean(np.abs(slope)) + 1e-9)
        )
    except Exception:
        pass

    try:
        curv = np.gradient(np.gradient(pred, A_GRID), A_GRID)
        metrics["curvature_score"] = 1.0 / (
            1.0 + np.mean(np.abs(curv)) / (np.mean(pred) + 1e-9)
        )
    except Exception:
        pass

    try:
        d1 = (pred[-1] - H_TARGET[-1]) / H_TARGET[-1]
        metrics["delta_today_score"] = 1.0 / (1.0 + abs(d1))
    except Exception:
        pass

    weights = dict(
        fit_score=0.50,
        early_suppression_score=0.10,
        smoothness_score=0.15,
        curvature_score=0.15,
        delta_today_score=0.10,
    )

    num = sum(metrics[k] * w for k, w in weights.items())
    metrics["combined_score"] = max(0.0, min(1.0, num / sum(weights.values())))

    return metrics


# ============================================================
# Stage 1 evaluation
# ============================================================

def evaluate_stage1(program_path: str) -> Dict[str, Any]:
    try:
        ns = _load_candidate_namespace(program_path)
        fps = _functions_present_score(ns)

        metrics = {
            "syntax_valid": 1.0,
            "module_loaded": 1.0,
            "functions_present_score": fps,
            "combined_score": fps,
        }

        return {
            "metrics": metrics,
            "combined_score": metrics["combined_score"],
        }

    except Exception as exc:
        metrics = {
            "syntax_valid": 0.0,
            "module_loaded": 0.0,
            "functions_present_score": 0.0,
            "combined_score": 0.0,
        }
        return {
            "metrics": metrics,
            "combined_score": metrics["combined_score"],
            "error": f"{exc}\n{traceback.format_exc()}",
        }




# ============================================================
# Stage 2 evaluation
# ============================================================

def evaluate_stage2(program_path: str) -> Dict[str, Any]:
    try:
        ns = _load_candidate_namespace(program_path)
        fps = _functions_present_score(ns)

        metrics = _physics_metrics(ns)
        metrics.update({
            "syntax_valid": 1.0,
            "module_loaded": 1.0,
            "functions_present_score": fps,
        })

        return {
            "metrics": metrics,
            "combined_score": metrics.get("combined_score", 0.0),
        }

    except Exception as exc:
        metrics = {
            "syntax_valid": 0.0,
            "module_loaded": 0.0,
            "functions_present_score": 0.0,
            "combined_score": 0.0,
        }
        return {
            "metrics": metrics,
            "combined_score": metrics["combined_score"],
            "error": f"{exc}\n{traceback.format_exc()}",
        }



# ============================================================
# Compatibility entry point
# ============================================================

def evaluate(program_path: str) -> Dict[str, Any]:
    return evaluate_stage2(program_path)


__all__ = ["evaluate", "evaluate_stage1", "evaluate_stage2"]
