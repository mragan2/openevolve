"""Evaluator for massive graviton / dark energy models in OpenEvolve.

This evaluator handles candidate modules that implement modifications to a
massive-graviton cosmological scaffold.  Candidates may emit either the
entire scaffold or only the EVOLVE block; in the latter case the evaluator
stitches the user-provided block into the canonical initial_program.py before
execution.  It then computes a suite of physics-inspired scores that
quantify how closely the candidate adheres to expected massive-gravity
behaviour.  Higher combined scores indicate more physically plausible
implementations.

Required functions in the candidate namespace:
  - graviton_mass_from_lambda(lambda_g_m: float) -> float
  - yukawa_potential(r: float, M: float, lambda_g_m: float) -> float
  - gw_group_velocity(omega: float, m_g: float) -> float
  - lambda_eff_from_mg(m_g: float) -> float
  - H_mg_phenomenological(a: float, m_g: float, H0: float) -> float

Optionally:
  - run_sanity_checks() -> dict

The evaluate() function returns a dictionary with keys including
'combined_score'.  On failure it returns combined_score=0.0 and an
'error' message.
"""

from __future__ import annotations

import importlib.util
import math
import tempfile
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any

import runpy

# ---------------------------------------------------------------------------
# Reference constants (must match initial_program constants)
# ---------------------------------------------------------------------------

C_LIGHT: float = 299_792_458.0
HBAR: float = 1.054_571_817e-34
G_NEWTON: float = 6.674_30e-11

# Reference graviton Compton wavelength and mass
LAMBDA_G_REF_METERS: float = 4.39e26  # ≈ 4.64 gly
M_G_EXPECTED: float = HBAR / (C_LIGHT * LAMBDA_G_REF_METERS)

# Observational scales
LAMBDA_EFF_REF: float = 1.0e-52   # m^-2, order of observed cosmological constant
OMEGA_MG_REF: float = 0.7         # present-day dark-energy fraction
H0_REF: float = 2.2e-18           # s^-1

# ---------------------------------------------------------------------------
# EVOLVE block splicing helpers
# ---------------------------------------------------------------------------

def _extract_evolve_block(source: str) -> str:
    """Extract the EVOLVE block from *source*.

    Raises ValueError if the markers are missing or the block is empty.
    """
    start_tag = "# EVOLVE-BLOCK-START"
    end_tag = "# EVOLVE-BLOCK-END"
    lines = source.splitlines()
    start_idx = None
    end_idx = None
    for idx, line in enumerate(lines):
        if line.strip() == start_tag:
            start_idx = idx
            continue
        if line.strip() == end_tag:
            end_idx = idx
            break
    if start_idx is None or end_idx is None or end_idx <= start_idx:
        raise ValueError("Missing EVOLVE block markers")
    block = "\n".join(lines[start_idx + 1 : end_idx]).strip()
    if not block:
        raise ValueError("EVOLVE block is empty")
    return block


def _sanitize_user_block(block: str) -> str:
    """Sanitise a user-supplied EVOLVE block.

    Removes directives such as 'from __future__ import' that would be invalid
    inside the EVOLVE block.  Raises ValueError if the result is empty.
    """
    sanitized_lines = []
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith("from __future__ import"):
            continue
        sanitized_lines.append(line)
    result = "\n".join(sanitized_lines).strip()
    if not result:
        raise ValueError("EVOLVE block became empty after sanitization")
    return result


def _build_locked_program(scaffold_source: str, user_block: str) -> str:
    """Inject *user_block* into the scaffold_source between EVOLVE markers."""
    lines = scaffold_source.splitlines()
    out_lines = []
    in_block = False
    start_tag = "# EVOLVE-BLOCK-START"
    end_tag = "# EVOLVE-BLOCK-END"
    for line in lines:
        stripped = line.strip()
        if stripped == start_tag:
            out_lines.append(line)
            out_lines.append(user_block)
            in_block = True
            continue
        if stripped == end_tag and in_block:
            in_block = False
            out_lines.append(line)
            continue
        if not in_block:
            out_lines.append(line)
    return "\n".join(out_lines)


def _execute_stitched_program(full_source: str) -> SimpleNamespace:
    """Execute *full_source* in isolation and return the resulting namespace."""
    # Write the stitched program to a temporary file
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tmp:
        tmp.write(full_source)
        tmp_path = Path(tmp.name)
    try:
        globals_dict = runpy.run_path(str(tmp_path), run_name="__candidate__")
        return SimpleNamespace(**globals_dict)
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def _import_full_module(program_path: str) -> SimpleNamespace:
    """Import a candidate module directly when it already contains the scaffold."""
    module_name = f"candidate_massive_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create spec for {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return SimpleNamespace(**module.__dict__)


def _load_candidate_module(program_path: str) -> SimpleNamespace:
    """Load a candidate file, stitching it into the scaffold when necessary."""
    path = Path(program_path)
    source = path.read_text(encoding="utf-8")
    # Path to the canonical scaffold (initial_program.py) located next to this evaluator
    scaffold_path = Path(__file__).resolve().parent / "initial_program.py"
    try:
        user_block = _extract_evolve_block(source)
        sanitized = _sanitize_user_block(user_block)
        scaffold_source = scaffold_path.read_text(encoding="utf-8")
        full_source = _build_locked_program(scaffold_source, sanitized)
        return _execute_stitched_program(full_source)
    except ValueError:
        # Candidate probably emitted the full module already
        return _import_full_module(str(program_path))


# ---------------------------------------------------------------------------
# Scoring utilities
# ---------------------------------------------------------------------------

def _score_from_relative_error(rel_err: float, scale: float = 1.0) -> float:
    """Map a relative error into the interval [0, 1] with a soft fall-off.

    A relative error equal to `scale` produces a score of about 0.5.  Smaller
    errors yield scores closer to 1, while larger errors rapidly decrease the
    score.  Negative errors are treated as zero.
    """
    rel = max(0.0, rel_err / max(scale, 1e-30))
    return 1.0 / (1.0 + rel)


def _score_bounded_ratio(val: float, target: float) -> float:
    """Score how close `val` is to `target` (> 0) in logarithmic space.

    Returns 1 when val ≈ target and decreases symmetrically for values an
    order of magnitude away.  Non-positive inputs yield a zero score.
    """
    if val <= 0.0 or target <= 0.0:
        return 0.0
    log_ratio = abs(math.log10(val / target))
    return 1.0 / (1.0 + log_ratio)


def _clamp01(x: float) -> float:
    """Clamp a float into the [0, 1] interval."""
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Primary entry point
# ---------------------------------------------------------------------------

def evaluate(program_path: str) -> Dict[str, float]:
    """
    Evaluate a candidate massive graviton cosmology implementation.

    This function loads the candidate program, stitches the evolved block into
    the canonical scaffold if necessary, and computes a set of physics-inspired
    metrics.  It returns a dictionary with keys such as 'combined_score'.
    Higher combined scores indicate better adherence to expected physical
    behaviour.
    """
    t0 = time.time()
    metrics: Dict[str, Any] = {
        "syntax_valid": 0.0,
        "module_loaded": 0.0,
        "functions_present_score": 0.0,
        "mg_score": 0.0,
        "yukawa_score": 0.0,
        "gw_score": 0.0,
        "lambda_eff_score": 0.0,
        "Hmg_score": 0.0,
        "combined_score": 0.0,
        "eval_time": 0.0,
    }
    try:
        ns = _load_candidate_module(program_path)
        metrics["syntax_valid"] = 1.0
        metrics["module_loaded"] = 1.0
        # Verify required functions exist
        required = [
            "graviton_mass_from_lambda",
            "yukawa_potential",
            "gw_group_velocity",
            "lambda_eff_from_mg",
            "H_mg_phenomenological",
        ]
        present_count = 0
        for fn in required:
            if hasattr(ns, fn) and callable(getattr(ns, fn)):
                present_count += 1
        metrics["functions_present_score"] = present_count / float(len(required))
        if metrics["functions_present_score"] < 1.0:
            # Early exit: candidate missing required functions
            metrics["combined_score"] = 0.0
            metrics["eval_time"] = time.time() - t0
            return metrics
        # m_g reproduction from lambda
        try:
            mg_from_lambda = ns.graviton_mass_from_lambda(float(LAMBDA_G_REF_METERS))
        except Exception:
            mg_from_lambda = float("nan")
        mg_ref = float(M_G_EXPECTED)
        if mg_ref != 0.0 and math.isfinite(mg_from_lambda):
            rel_err_mg = abs(mg_from_lambda - mg_ref) / max(abs(mg_ref), 1e-99)
            metrics["mg_score"] = _score_from_relative_error(rel_err_mg, scale=0.1)
        else:
            metrics["mg_score"] = 0.0
        # Yukawa potential behaviour
        try:
            M_test = 1.0e30
            r_small = 1.0e20  # much smaller than lambda_g_ref
            r_large = float(LAMBDA_G_REF_METERS)  # around lambda_g_ref
            V_small = ns.yukawa_potential(r_small, M_test, float(LAMBDA_G_REF_METERS))
            V_newton_small = -G_NEWTON * M_test / r_small
            ratio_small = V_small / V_newton_small if V_newton_small != 0 else 0.0
            error_small = abs(ratio_small - 1.0)
            score_small = _clamp01(1.0 / (1.0 + error_small))
            V_large = ns.yukawa_potential(r_large, M_test, float(LAMBDA_G_REF_METERS))
            V_newton_large = -G_NEWTON * M_test / r_large
            ratio_large = V_large / V_newton_large if V_newton_large != 0 else 0.0
            expected_large = math.exp(-r_large / float(LAMBDA_G_REF_METERS))
            error_large = abs(ratio_large - expected_large)
            score_large = _clamp01(1.0 / (1.0 + error_large))
            metrics["yukawa_score"] = 0.5 * (score_small + score_large)
        except Exception:
            metrics["yukawa_score"] = 0.0
        # Gravitational-wave group velocity
        try:
            omega_high = 1.0e3  # rad/s
            v_g_val = ns.gw_group_velocity(omega_high, float(M_G_EXPECTED))
            ratio = (M_G_EXPECTED * C_LIGHT * C_LIGHT) / (HBAR * omega_high)
            expected_vg = C_LIGHT * math.sqrt(max(0.0, 1.0 - ratio * ratio))
            rel_err_vg = abs(v_g_val - expected_vg) / max(abs(expected_vg), 1e-99)
            metrics["gw_score"] = _score_from_relative_error(rel_err_vg, scale=0.05)
        except Exception:
            metrics["gw_score"] = 0.0
        # Effective cosmological constant mapping
        try:
            lam_val = ns.lambda_eff_from_mg(float(M_G_EXPECTED))
            if lam_val > 0.0 and LAMBDA_EFF_REF > 0.0:
                metrics["lambda_eff_score"] = _score_bounded_ratio(lam_val, float(LAMBDA_EFF_REF))
            else:
                metrics["lambda_eff_score"] = 0.0
        except Exception:
            metrics["lambda_eff_score"] = 0.0
        # Phenomenological graviton contribution at a=1
        try:
            Hmg_val = ns.H_mg_phenomenological(1.0, float(M_G_EXPECTED), float(H0_REF))
            target_val = float(OMEGA_MG_REF) * (float(H0_REF) ** 2)
            if Hmg_val > 0.0 and target_val > 0.0:
                metrics["Hmg_score"] = _score_bounded_ratio(Hmg_val, target_val)
            else:
                metrics["Hmg_score"] = 0.0
        except Exception:
            metrics["Hmg_score"] = 0.0
        # Weighted combined score
        w_funcs = 0.1
        w_mg = 0.15
        w_yuk = 0.15
        w_gw = 0.1
        w_lambda = 0.25
        w_Hmg = 0.25
        metrics["combined_score"] = (
            w_funcs * metrics["functions_present_score"]
            + w_mg * metrics["mg_score"]
            + w_yuk * metrics["yukawa_score"]
            + w_gw * metrics["gw_score"]
            + w_lambda * metrics["lambda_eff_score"]
            + w_Hmg * metrics["Hmg_score"]
        )
        metrics["eval_time"] = time.time() - t0
        metrics["combined_score"] = _clamp01(metrics["combined_score"])
        return metrics
    except Exception as exc:
        # Unexpected failure: capture error and return zero score
        metrics["combined_score"] = 0.0
        metrics["error"] = str(exc)
        metrics["eval_time"] = time.time() - t0
        return metrics


__all__ = ["evaluate"]