"""
Evaluator for massive graviton / dark energy models in OpenEvolve.

Expected functions in the candidate module:
  - graviton_mass_from_lambda(lambda_g_m: float) -> float
  - yukawa_potential(r: float, M: float, lambda_g_m: float) -> float
  - gw_group_velocity(omega: float, m_g: float) -> float
  - lambda_eff_from_mg(m_g: float) -> float
  - H_mg_phenomenological(a: float, m_g: float, H0: float) -> float
  - build_massive_gravity_model(...) -> dict

Optionally:
  - run_sanity_checks() -> dict

CRITICAL (per OpenEvolve README):
  - evaluate(program_path) must return a dictionary, not EvaluationResult.
  - The dict MUST include 'combined_score' as the primary metric.
  - On failure: combined_score = 0.0 and an 'error' key is recommended.
"""

import importlib.util
import math
import time
import uuid
from typing import Any, Dict


# Reference constants (must match initial_program constants)
C_LIGHT = 299_792_458.0
HBAR = 1.054_571_817e-34
G_NEWTON = 6.674_30e-11

# Reference graviton Compton wavelength and mass
LAMBDA_G_REF_METERS = 4.39e26  # ≈ 4.64 gly
M_G_EXPECTED = HBAR / (C_LIGHT * LAMBDA_G_REF_METERS)

# Observational scales
LAMBDA_EFF_REF = 1.0e-52   # m^-2, order of observed cosmological constant
OMEGA_MG_REF = 0.7         # present-day dark-energy fraction
H0_REF = 2.2e-18           # s^-1


def _load_candidate_module(program_path: str):
    """Dynamically import the candidate program as a module."""
    module_name = f"candidate_massive_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create spec for {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _score_from_relative_error(rel_err: float, scale: float = 1.0) -> float:
    """
    Convert a relative error into a [0, 1] score.

    rel_err: |x - x_ref| / |x_ref|
    scale: tolerance scale; rel_err ~ scale => score ~ 0.5.
    """
    rel = max(0.0, rel_err / max(scale, 1e-30))
    return 1.0 / (1.0 + rel)


def _score_bounded_ratio(val: float, target: float) -> float:
    """
    Score how close val is to target > 0, in log space.
    Score is 1.0 when val ~ target and falls off by orders of magnitude.
    """
    if val <= 0.0 or target <= 0.0:
        return 0.0
    log_ratio = abs(math.log10(val / target))
    return 1.0 / (1.0 + log_ratio)


def _clamp01(x: float) -> float:
    """Clamp a float to the [0, 1] interval."""
    return max(0.0, min(1.0, float(x)))


def _evaluate_core(program_path: str) -> Dict[str, Any]:
    """
    Core evaluation logic that returns a metrics dictionary.

    This is used by evaluate, evaluate_stage1, and evaluate_stage2.
    """
    start = time.perf_counter()

    syntax_valid = 0.0
    module_loaded = 0.0
    functions_present_score = 0.0
    mg_score = 0.0
    yukawa_local_score = 0.0
    yukawa_suppression_score = 0.0
    gw_velocity_score = 0.0
    lambda_eff_score = 0.0
    H_mg_score = 0.0
    sanity_checks_score = 0.0

    try:
        module = _load_candidate_module(program_path)
        syntax_valid = 1.0
        module_loaded = 1.0
    except Exception:
        eval_time = time.perf_counter() - start
        return {
            "combined_score": 0.0,
            "syntax_valid": syntax_valid,
            "module_loaded": module_loaded,
            "functions_present_score": 0.0,
            "mg_score": 0.0,
            "yukawa_local_score": 0.0,
            "yukawa_suppression_score": 0.0,
            "gw_velocity_score": 0.0,
            "lambda_eff_score": 0.0,
            "H_mg_score": 0.0,
            "sanity_checks_score": 0.0,
            "eval_time": float(eval_time),
            "error": "import_error",
        }

    # ---------- Check required functions ----------
    required_funcs = [
        "graviton_mass_from_lambda",
        "yukawa_potential",
        "gw_group_velocity",
        "lambda_eff_from_mg",
        "H_mg_phenomenological",
        "build_massive_gravity_model",
    ]
    present = [hasattr(module, name) for name in required_funcs]
    functions_present_score = sum(1.0 for p in present if p) / max(1, len(required_funcs))

    if functions_present_score == 0.0:
        eval_time = time.perf_counter() - start
        return {
            "combined_score": 0.0,
            "syntax_valid": syntax_valid,
            "module_loaded": module_loaded,
            "functions_present_score": functions_present_score,
            "mg_score": mg_score,
            "yukawa_local_score": yukawa_local_score,
            "yukawa_suppression_score": yukawa_suppression_score,
            "gw_velocity_score": gw_velocity_score,
            "lambda_eff_score": lambda_eff_score,
            "H_mg_score": H_mg_score,
            "sanity_checks_score": sanity_checks_score,
            "eval_time": float(eval_time),
            "error": "missing_required_functions",
        }

    # ---------- Physics checks ----------

    # 1) m_g from λ_g
    m_g = M_G_EXPECTED
    try:
        if hasattr(module, "graviton_mass_from_lambda"):
            m_g_val = module.graviton_mass_from_lambda(LAMBDA_G_REF_METERS)
            if m_g_val > 0.0:
                m_g = m_g_val
            rel_err_mg = abs(m_g - M_G_EXPECTED) / max(abs(M_G_EXPECTED), 1e-99)
            mg_score = _score_from_relative_error(rel_err_mg, scale=1.0)
    except Exception:
        pass

    # 2) Yukawa vs Newtonian behavior
    try:
        if hasattr(module, "yukawa_potential"):
            M_test = 1.0e30
            r_small = 1.0e20   # r << λ_g
            r_large = 1.0e28   # ~ λ_g

            V_yuk_small = module.yukawa_potential(r_small, M_test, LAMBDA_G_REF_METERS)
            V_new_small = -G_NEWTON * M_test / r_small
            rel_diff_small = abs(V_yuk_small - V_new_small) / max(abs(V_new_small), 1e-99)
            yukawa_local_score = _score_from_relative_error(rel_diff_small, scale=0.1)

            V_yuk_large = module.yukawa_potential(r_large, M_test, LAMBDA_G_REF_METERS)
            V_new_large = -G_NEWTON * M_test / r_large
            ratio = abs(V_yuk_large) / max(abs(V_new_large), 1e-99)
            if ratio < 1.0:
                yukawa_suppression_score = 1.0 / (1.0 + ratio)
            else:
                yukawa_suppression_score = 1.0 / (1.0 + ratio * ratio)
    except Exception:
        pass

    # 3) GW group velocity close to c and subluminal
    try:
        if hasattr(module, "gw_group_velocity"):
            omega = 1.0e3  # rad·s^-1
            v_g = module.gw_group_velocity(omega, m_g)
            if v_g > 0.0:
                rel_diff_v = abs(v_g - C_LIGHT) / max(C_LIGHT, 1e-99)
                gw_velocity_score = _score_from_relative_error(rel_diff_v, scale=1.0e-2)
                if v_g > C_LIGHT:
                    gw_velocity_score *= 0.1  # heavy penalty for superluminal
    except Exception:
        pass

    # 4) Λ_eff scale
    try:
        if hasattr(module, "lambda_eff_from_mg"):
            lam_eff = module.lambda_eff_from_mg(m_g)
            lambda_eff_score = _score_bounded_ratio(abs(lam_eff), LAMBDA_EFF_REF)
    except Exception:
        pass

    # 5) H_mg phenomenology at a = 1
    try:
        if hasattr(module, "H_mg_phenomenological"):
            H_mg_val = module.H_mg_phenomenological(1.0, m_g, H0_REF)
            if H_mg_val > 0.0:
                target = OMEGA_MG_REF * H0_REF * H0_REF
                H_mg_score = _score_bounded_ratio(H_mg_val, target)
    except Exception:
        pass

    # 6) Optional internal sanity checks
    try:
        if hasattr(module, "run_sanity_checks"):
            res = module.run_sanity_checks()
            if isinstance(res, dict):
                partial_scores = []
                if "rel_error_m_g" in res:
                    partial_scores.append(
                        _score_from_relative_error(res["rel_error_m_g"], scale=1.0)
                    )
                if "rel_diff_potential" in res:
                    partial_scores.append(
                        _score_from_relative_error(res["rel_diff_potential"], scale=0.1)
                    )
                if "rel_diff_vg" in res:
                    partial_scores.append(
                        _score_from_relative_error(res["rel_diff_vg"], scale=1.0e-2)
                    )
                if partial_scores:
                    sanity_checks_score = sum(partial_scores) / len(partial_scores)
    except Exception:
        pass

    # ---------- Clamp and combine ----------

    syntax_valid = _clamp01(syntax_valid)
    module_loaded = _clamp01(module_loaded)
    functions_present_score = _clamp01(functions_present_score)
    mg_score = _clamp01(mg_score)
    yukawa_local_score = _clamp01(yukawa_local_score)
    yukawa_suppression_score = _clamp01(yukawa_suppression_score)
    gw_velocity_score = _clamp01(gw_velocity_score)
    lambda_eff_score = _clamp01(lambda_eff_score)
    H_mg_score = _clamp01(H_mg_score)
    sanity_checks_score = _clamp01(sanity_checks_score)

    combined_score = (
        0.10 * syntax_valid
        + 0.10 * module_loaded
        + 0.10 * functions_present_score
        + 0.15 * mg_score
        + 0.10 * yukawa_local_score
        + 0.05 * yukawa_suppression_score
        + 0.10 * gw_velocity_score
        + 0.10 * lambda_eff_score
        + 0.10 * H_mg_score
        + 0.10 * sanity_checks_score
    )

    combined_score = _clamp01(combined_score)
    eval_time = time.perf_counter() - start

    metrics: Dict[str, Any] = {
        "combined_score": combined_score,
        "syntax_valid": syntax_valid,
        "module_loaded": module_loaded,
        "functions_present_score": functions_present_score,
        "mg_score": mg_score,
        "yukawa_local_score": yukawa_local_score,
        "yukawa_suppression_score": yukawa_suppression_score,
        "gw_velocity_score": gw_velocity_score,
        "lambda_eff_score": lambda_eff_score,
        "H_mg_score": H_mg_score,
        "sanity_checks_score": sanity_checks_score,
        "eval_time": float(eval_time),
    }
    return metrics


def evaluate(program_path: str) -> Dict[str, Any]:
    """
    Main entry point used by OpenEvolve.

    MUST return a dictionary containing at least:
      - 'combined_score' (primary metric).

    On error, returns combined_score = 0.0 and an 'error' field.
    """
    try:
        metrics = _evaluate_core(program_path)
        if "combined_score" not in metrics:
            metrics["combined_score"] = 0.0
        return metrics
    except Exception as e:
        return {
            "combined_score": 0.0,
            "error": str(e),
        }


def evaluate_stage1(program_path: str) -> Dict[str, Any]:
    """
    Optional stage 1 for cascade evaluation.

    Focuses on syntax/module health and presence of required functions.
    """
    try:
        metrics = _evaluate_core(program_path)
        core = (
            0.3 * metrics.get("syntax_valid", 0.0)
            + 0.3 * metrics.get("module_loaded", 0.0)
            + 0.4 * metrics.get("functions_present_score", 0.0)
        )
        metrics["combined_score"] = _clamp01(core)
        return metrics
    except Exception as e:
        return {
            "combined_score": 0.0,
            "error": f"stage1_error: {e}",
        }


def evaluate_stage2(program_path: str) -> Dict[str, Any]:
    """
    Optional stage 2 for cascade evaluation.

    Uses the full physics-aware combined score.
    """
    try:
        metrics = _evaluate_core(program_path)
        if "combined_score" not in metrics:
            metrics["combined_score"] = 0.0
        return metrics
    except Exception as e:
        return {
            "combined_score": 0.0,
            "error": f"stage2_error: {e}",
        }

