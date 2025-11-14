"""Evaluator for massive graviton / dark energy models in OpenEvolve.

This evaluator handles candidate modules that implement modifications to a
massive-graviton cosmological scaffold. Candidates may emit either the
entire scaffold or only the EVOLVE block; in the latter case the evaluator
stitches the user-provided block into the canonical initial_program.py before
execution. It then computes a suite of physics-inspired scores that quantify
how closely the candidate adheres to expected massive-gravity behaviour.

Required functions in the candidate namespace:
  - graviton_mass_from_lambda(lambda_g_m: float) -> float
  - yukawa_potential(r: float, M: float, lambda_g_m: float) -> float
  - gw_group_velocity(omega: float, m_g: float) -> float
  - lambda_eff_from_mg(m_g: float) -> float
  - H_mg_phenomenological(a: float, m_g: float, H0: float) -> float

Optionally:
  - run_sanity_checks() -> dict

The evaluate() function returns a dictionary with keys including
'combined_score'. On failure it returns combined_score=0.0 and an
'error' message.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import os
import tempfile
import time
import traceback
import uuid
from pathlib import Path
from typing import Dict, Any
from types import SimpleNamespace

import runpy

# ============================================================
# Required functions expected in the evolved program
# ============================================================

REQUIRED_FUNCTIONS = [
    "graviton_mass_from_lambda",
    "yukawa_potential",
    "gw_group_velocity",
    "lambda_eff_from_mg",
    "H_mg_phenomenological",
]

# ============================================================
# Environment helpers (for tunable tolerances)
# ============================================================

def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


def _bool_env(name: str) -> bool:
    return os.environ.get(name, "0").lower() in {"1", "true", "yes", "on"}


LOGGER = logging.getLogger(__name__)
LOG_LOW_SCORE_DETAILS = _bool_env("OPENEVOLVE_LOG_LOW_SCORES")
LOW_SCORE_THRESHOLD = _float_env("OPENEVOLVE_LOW_SCORE_THRESHOLD", 0.3)

MG_TOL = _float_env("OPENEVOLVE_MG_TOL", 1e-3)
YUKAWA_TOL = _float_env("OPENEVOLVE_YUKAWA_TOL", 1e-3)
GW_TOL = _float_env("OPENEVOLVE_GW_TOL", 1e-6)
LAMBDA_TOL = _float_env("OPENEVOLVE_LAMBDA_TOL", 3.0)
HMG_TOL = _float_env("OPENEVOLVE_HMG_TOL", 0.2)

# ============================================================
# Reference constants
# ============================================================

C_LIGHT = 299_792_458.0
HBAR = 1.054_571_817e-34
G_NEWTON = 6.674_30e-11

LAMBDA_G_REF_METERS = 4.39e26
M_G_EXPECTED = HBAR / (C_LIGHT * LAMBDA_G_REF_METERS)

LAMBDA_EFF_REF = 1.0e-52
OMEGA_MG_REF = 0.7
H0_REF = 2.2e-18

# ============================================================
# EVOLVE block extraction / stitching
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
    start_tag = "# EVOLVE-BLOCK-START"
    end_tag = "# EVOLVE-BLOCK-END"

    for line in scaffold_source.splitlines():
        stripped = line.strip()
        if stripped == start_tag:
            out.append(line)
            out.append(user_block)
            in_block = True
            continue
        if stripped == end_tag and in_block:
            out.append(line)
            in_block = False
            continue
        if not in_block:
            out.append(line)

    return "\n".join(out)


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
    """Load either stitched EVOLVE block or full module."""
    full_path = Path(program_path)
    try:
        source = full_path.read_text(encoding="utf-8")
    except Exception as exc:
        raise RuntimeError(f"Failed to read candidate program {full_path}: {exc}")

    scaffold_path = Path(__file__).parent / "initial_program.py"

    try:
        block = _extract_evolve_block(source)
        block = _sanitize_user_block(block)
        scaffold_source = scaffold_path.read_text(encoding="utf-8")
        full_source = _build_locked_program(scaffold_source, block)
        return _execute_stitched(full_source)
    except ValueError:
        # No EVOLVE block: assume this is a full module
        return _import_full(str(full_path))

# ============================================================
# Function presence scoring
# ============================================================

def _functions_present_score(ns: Any) -> float:
    have = 0
    for fn in REQUIRED_FUNCTIONS:
        if hasattr(ns, fn) and callable(getattr(ns, fn)):
            have += 1
    return float(have) / float(len(REQUIRED_FUNCTIONS))

# ============================================================
# Low-score logging
# ============================================================

def _log_low(metrics: Dict[str, float]) -> None:
    if LOG_LOW_SCORE_DETAILS and metrics.get("combined_score", 1.0) < LOW_SCORE_THRESHOLD:
        LOGGER.warning("LOW SCORE: %s", metrics)

# ============================================================
# Full metric bundle for massive graviton physics
# ============================================================

def _full_metric_bundle(ns: Any, fps: float) -> Dict[str, float]:
    """
    Full physics evaluation for massive graviton cosmology.
    Includes the original 5 metrics PLUS 5 new metrics that provide
    meaningful selective pressure beyond the baseline implementation:
        - early_suppression_score          (A)
        - slope_score                      (B)
        - curvature_score                  (C)
        - w_eff_score                      (D)
        - lambda_variation_score           (E)
    """

    # Base metrics
    metrics: Dict[str, float] = {
        "functions_present_score": float(fps),
        "mg_score": 0.0,
        "yukawa_score": 0.0,
        "gw_score": 0.0,
        "lambda_eff_score": 0.0,
        "Hmg_score": 0.0,

        # New advanced metrics (A–E)
        "early_suppression_score": 0.0,
        "slope_score": 0.0,
        "curvature_score": 0.0,
        "w_eff_score": 0.0,
        "lambda_variation_score": 0.0,

        "combined_score": 0.0,
    }

    if fps < 1.0:
        return metrics

    # ============================================================
    # 1. Original Metrics
    # ============================================================

    # --- mg reproduction ---
    try:
        mg_calc = ns.graviton_mass_from_lambda(LAMBDA_G_REF_METERS)
        rel_err = abs(mg_calc - M_G_EXPECTED) / max(abs(M_G_EXPECTED), 1e-99)
        metrics["mg_score"] = 1.0 / (1.0 + rel_err / MG_TOL)
    except Exception:
        metrics["mg_score"] = 0.0

    # --- Yukawa behaviour ---
    try:
        M_test = 1.0e30
        r_small = 1.0e20
        r_large = LAMBDA_G_REF_METERS

        V_small = ns.yukawa_potential(r_small, M_test, LAMBDA_G_REF_METERS)
        V_newton_small = -G_NEWTON * M_test / r_small
        err_small = abs((V_small / V_newton_small) - 1.0)

        V_large = ns.yukawa_potential(r_large, M_test, LAMBDA_G_REF_METERS)
        V_newton_large = -G_NEWTON * M_test / r_large
        expected_large = math.exp(-1.0)
        err_large = abs((V_large / V_newton_large) - expected_large)

        metrics["yukawa_score"] = 0.5 * (
            1.0 / (1.0 + err_small / YUKAWA_TOL)
            + 1.0 / (1.0 + err_large / YUKAWA_TOL)
        )
    except Exception:
        metrics["yukawa_score"] = 0.0

    # --- GW velocity ---
    try:
        omega = 1.0e3
        v = ns.gw_group_velocity(omega, M_G_EXPECTED)
        ratio = (M_G_EXPECTED * C_LIGHT * C_LIGHT) / (HBAR * omega)
        expected = C_LIGHT * math.sqrt(max(0.0, 1.0 - ratio * ratio))
        rel = abs(v - expected) / max(abs(expected), 1e-99)
        metrics["gw_score"] = 1.0 / (1.0 + rel / GW_TOL)
    except Exception:
        metrics["gw_score"] = 0.0

    # --- Lambda_eff ---
    try:
        lam = ns.lambda_eff_from_mg(M_G_EXPECTED)
        if lam > 0.0:
            log_ratio = abs(math.log10(lam / LAMBDA_EFF_REF))
            metrics["lambda_eff_score"] = 1.0 / (1.0 + log_ratio / LAMBDA_TOL)
        else:
            metrics["lambda_eff_score"] = 0.0
    except Exception:
        metrics["lambda_eff_score"] = 0.0

    # --- Hmg(a=1) target ---
    try:
        Hmg = ns.H_mg_phenomenological(1.0, M_G_EXPECTED, H0_REF)
        target = OMEGA_MG_REF * (H0_REF ** 2)
        log_ratio = abs(math.log10(max(Hmg, 1e-50) / max(target, 1e-50)))
        metrics["Hmg_score"] = 1.0 / (1.0 + log_ratio / HMG_TOL)
    except Exception:
        metrics["Hmg_score"] = 0.0

    # ============================================================
    # 2. NEW METRIC A — Early-time suppression
    # ============================================================
    try:
        H_early = ns.H_mg_phenomenological(0.05, M_G_EXPECTED, H0_REF)
        H_today = ns.H_mg_phenomenological(1.0, M_G_EXPECTED, H0_REF)
        ratio = H_early / max(H_today, 1e-50)
        metrics["early_suppression_score"] = 1.0 / (1.0 + 100.0 * ratio)
    except Exception:
        metrics["early_suppression_score"] = 0.0

    # ============================================================
    # 3. NEW METRIC B — Late-time slope
    # ============================================================
    try:
        a1, a2 = 0.3, 1.0
        H1 = ns.H_mg_phenomenological(a1, M_G_EXPECTED, H0_REF)
        H2 = ns.H_mg_phenomenological(a2, M_G_EXPECTED, H0_REF)
        slope = abs(H2 - H1) / (a2 - a1)
        target_slope = 1e-35
        metrics["slope_score"] = 1.0 / (1.0 + abs(math.log10((slope + 1e-99) / target_slope)))
    except Exception:
        metrics["slope_score"] = 0.0

    # ============================================================
    # 4. NEW METRIC C — Curvature of the dark-energy term
    # ============================================================
    try:
        alist = [0.3, 0.5, 0.7, 1.0]
        H = [ns.H_mg_phenomenological(a, M_G_EXPECTED, H0_REF) for a in alist]
        curvature = 0.0
        for i in range(1, len(H) - 1):
            curvature += abs(H[i + 1] - 2 * H[i] + H[i - 1])
        curvature /= (len(H) - 2)
        target_curv = 1e-72
        metrics["curvature_score"] = 1.0 / (1.0 + abs(math.log10((curvature + 1e-99) / target_curv)))
    except Exception:
        metrics["curvature_score"] = 0.0

    # ============================================================
    # 5. NEW METRIC D — Effective equation of state w(a)
    # ============================================================
    try:
        def w_eff(a):
            eps = 1e-3
            Hm = ns.H_mg_phenomenological(a, M_G_EXPECTED, H0_REF)
            Hp = ns.H_mg_phenomenological(a + eps, M_G_EXPECTED, H0_REF)
            Hm2 = Hm * Hm
            Hp2 = Hp * Hp
            dlnH2 = math.log(max(Hp2, 1e-99)) - math.log(max(Hm2, 1e-99))
            dlnA = math.log(a + eps) - math.log(a)
            return -1.0 - (1.0 / 3.0) * (dlnH2 / dlnA)

        w1 = w_eff(1.0)
        w03 = w_eff(0.3)
        err = abs(w1 + 1.0) + abs(w03 + 1.0)  # want ≈ -1 but not flat everywhere
        metrics["w_eff_score"] = 1.0 / (1.0 + err)
    except Exception:
        metrics["w_eff_score"] = 0.0

    # ============================================================
    # 6. NEW METRIC E — Lambda_eff(a) variation
    # ============================================================
    try:
        a_grid = [0.3 + i * 0.07 for i in range(10)]
        lam_values = []
        for a in a_grid:
            Hm = ns.H_mg_phenomenological(a, M_G_EXPECTED, H0_REF)
            lam_values.append(Hm / max(H0_REF * H0_REF, 1e-99))

        mean_lam = sum(lam_values) / len(lam_values)
        var = sum((x - mean_lam) ** 2 for x in lam_values) / len(lam_values)
        target_var = 1e-70
        metrics["lambda_variation_score"] = 1.0 / (1.0 + var / target_var)
    except Exception:
        metrics["lambda_variation_score"] = 0.0

    # ============================================================
    # Combined weighted score
    # ============================================================

    weights = dict(
        functions_present_score=0.05,
        mg_score=0.10,
        yukawa_score=0.10,
        gw_score=0.05,
        lambda_eff_score=0.15,
        Hmg_score=0.10,
        early_suppression_score=0.10,
        slope_score=0.10,
        curvature_score=0.10,
        w_eff_score=0.10,
        lambda_variation_score=0.05,
    )

    num = sum(metrics[k] * w for k, w in weights.items())
    den = sum(weights.values())
    metrics["combined_score"] = float(max(0.0, min(1.0, num / den)))

    _log_low(metrics)
    return metrics


# ============================================================
# Stage 1 Evaluation
# ============================================================

def evaluate_stage1(program_path: str) -> Dict[str, float]:
    """
    Fast structural check used for cascade_evaluation stage 1.
    Only tests that the module imports and required functions exist.
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

# ============================================================
# Stage 2 Evaluation (full physics)
# ============================================================

def evaluate_stage2(program_path: str) -> Dict[str, float]:
    """
    Full physics evaluation used for cascade_evaluation stage 2
    and for the default evaluate() entry point.
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
            "mg_score": 0.0,
            "yukawa_score": 0.0,
            "gw_score": 0.0,
            "lambda_eff_score": 0.0,
            "Hmg_score": 0.0,
            "combined_score": 0.0,
            "eval_time": float(t1 - t0),
            "error": f"{exc}\n{traceback.format_exc()}",
        }

# ============================================================
# Compatibility entry point
# ============================================================

def evaluate(program_path: str) -> Dict[str, float]:
    """
    Default entry point expected by OpenEvolve when cascade_evaluation is off.
    When cascade_evaluation is on, the framework calls evaluate_stage1 and
    evaluate_stage2 directly according to cascade_thresholds.
    """
    return evaluate_stage2(program_path)


__all__ = ["evaluate", "evaluate_stage1", "evaluate_stage2"]
