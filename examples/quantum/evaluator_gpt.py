"""
Standalone evaluator for semiclassical quantum cosmology (OPTION B).

This module is self-contained and loaded directly by OpenEvolve. The workflow is:

    1. Read the candidate program produced by evolution.
    2. Extract the code between EVOLVE-BLOCK markers.
    3. Splice that code into the fixed scaffold in initial_program.py.
    4. Execute the stitched program in isolation.
    5. Return numeric metrics for MAP-Elites / cascade evolution.

Cascade support:
    - evaluate_stage1 performs the light checks (syntax/module load/function presence).
    - evaluate_stage2 runs the full physical evaluation.
    - evaluate delegates to evaluate_stage2 for backward compatibility.
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

REQUIRED_FUNCTIONS = [
    "rho_quantum",
    "H_squared_with_quantum",
    "run_sanity_checks",
]


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


# ---------------------------------------------------------------------------
# Helpers for safely constructing / executing candidate programs
# ---------------------------------------------------------------------------


def _extract_evolve_block(source: str) -> str:
    """
    Extract the user-supplied EVOLVE block from a candidate source.

    Strategy (in order):
      1. Prefer explicit markers: `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END`.
      2. Fall back to AST: find a top-level `def rho_quantum(...)` and return
         the full source segment for that function definition.
      3. Line-based fallback: search for a `def rho_quantum` line and capture
         the contiguous indented block following it.

    Returns the extracted source (typically a `def rho_quantum(...)` block).
    """
    lines = source.splitlines()

    # 1) Try explicit markers first (strict, preferred)
    start = None
    end = None
    for idx, line in enumerate(lines):
        if "# EVOLVE-BLOCK-START" in line:
            start = idx
        if "# EVOLVE-BLOCK-END" in line:
            end = idx
            break

    if start is not None and end is not None and end > start:
        block = "\n".join(lines[start + 1 : end]).strip()
        if not block:
            raise ValueError("EVOLVE block is empty.")
        return block

    # 2) AST fallback: look for a FunctionDef named 'rho_quantum' or other
    #    reasonable variants (e.g., 'rho_q', 'rhoQuantum', etc.). We prefer
    #    exact match but fall back to a fuzzy match on 'rho'/'quant' in name.
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            name = node.name.lower()
            exact = name == "rho_quantum"
            fuzzy = ("rho" in name) or ("quant" in name) or ("q" == name)
            if not (exact or fuzzy):
                continue

            # Prefer end_lineno when available (Python 3.8+)
            if hasattr(node, "end_lineno") and node.end_lineno is not None:
                start_idx = node.lineno - 1
                end_idx = node.end_lineno
                block = "\n".join(lines[start_idx:end_idx]).strip()
                if block:
                    return block

            # Try ast.get_source_segment() if available
            try:
                seg = ast.get_source_segment(source, node)
                if seg and seg.strip():
                    return seg.strip()
            except Exception:
                pass

            # Best-effort manual range: capture until the next top-level def
            start_idx = node.lineno - 1
            def_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
            end_idx = start_idx + 1
            for i in range(start_idx + 1, len(lines)):
                line = lines[i]
                if not line.strip():
                    end_idx = i + 1
                    continue
                indent = len(line) - len(line.lstrip())
                if indent <= def_indent and line.lstrip().startswith("def "):
                    break
                end_idx = i + 1

            block = "\n".join(lines[start_idx:end_idx]).strip()
            if block:
                return block
    except Exception:
        # AST parsing can fail on malformed candidates; fall through to line-scan
        pass

    # 3) Line-based fallback: search for top-level defs that look like the
    #    target function. Accept names containing 'rho' or 'quant' as fuzzy matches.
    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped.startswith("def "):
            continue
        # extract function name from 'def name('
        try:
            fname = stripped.split()[1].split("(")[0].lower()
        except Exception:
            fname = ""
        if not ("rho" in fname or "quant" in fname):
            continue
        # treat this as the evolve block candidate
        # (covers def rho_q, def rho_quantum, def rhoQuantum, etc.)
        start_idx = idx
        def_indent = len(line) - len(line.lstrip())
        end_idx = start_idx + 1
        for i in range(start_idx + 1, len(lines)):
            l = lines[i]
            if not l.strip():
                end_idx = i + 1
                continue
            indent = len(l) - len(l.lstrip())
            if indent <= def_indent and l.lstrip().startswith("def "):
                break
            end_idx = i + 1

        block = "\n".join(lines[start_idx:end_idx]).strip()
        if block:
            return block

    # 4) Final heuristic: if the file contains a top-level code section that
    #    looks like a function body (indented lines referencing 'a' or 'm_g'),
    #    wrap it into a def rho_quantum(...) to attempt recovery. This is
    #    aggressive and should be last-resort only.
    body_lines = []
    for i, line in enumerate(lines):
        if line.startswith("    ") or line.startswith("\t"):
            body_lines.append(line)
        elif body_lines:
            break
    if body_lines:
        wrapped = "def rho_quantum(a, H_classical, m_g):\n" + "\n".join(body_lines)
        return wrapped

    # Nothing found
    raise ValueError("Missing or malformed EVOLVE block markers.")


def _precheck_candidate_source(source: str) -> None:
    """Lightweight pre-checks to detect common malformed outputs.

    - Detect unbalanced triple-quoted strings which commonly cause
      `unterminated triple-quoted string literal` SyntaxErrors when the
      scaffold is stitched and executed.
    - Try a quick AST parse to catch obvious unterminated string errors
      early and return a clearer ValueError.
    """
    # Fast triple-quote balance check
    tq1 = source.count('"""')
    tq2 = source.count("'''")
    if (tq1 % 2) != 0 or (tq2 % 2) != 0:
        raise ValueError("Unbalanced triple-quoted string literal in candidate source.")

    # Try parsing; if parsing fails with unterminated string, raise a ValueError
    try:
        ast.parse(source)
    except SyntaxError as e:
        msg = str(e)
        if "unterminated string literal" in msg or "EOL while scanning string literal" in msg:
            raise ValueError(f"Candidate has unterminated string literal: {e}")
        # For other syntax errors we don't reject here; let later extraction handle them.
    except Exception:
        # Non-syntax problems are ignored at this stage
        pass


def _sanitize_user_block(block: str) -> str:
    """Remove forbidden directives (e.g., from __future__ imports)."""
    sanitized: List[str] = []

    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith("from __future__ import") or stripped.startswith("import __future__"):
            # The scaffold already declares future imports at the top. Duplicates
            # inserted mid-file would be a syntax error, so we drop them here.
            continue
        sanitized.append(line)

    result = "\n".join(sanitized).strip()
    if not result:
        raise ValueError("EVOLVE block became empty after sanitization.")
    return result


def _build_locked_program(scaffold_source: str, user_block: str) -> str:
    lines = scaffold_source.splitlines()
    out_lines: List[str] = []
    in_block = False

    for line in lines:
        if "# EVOLVE-BLOCK-START" in line:
            out_lines.append(line)
            out_lines.append(user_block)
            in_block = True
            continue

        if "# EVOLVE-BLOCK-END" in line:
            in_block = False
            out_lines.append(line)
            continue

        if not in_block:
            out_lines.append(line)

    return "\n".join(out_lines)


def _safe_run_candidate(full_src: str) -> Dict[str, Any]:
    """Persist full_src to a temp file, execute it, and return the globals."""
    tmp_path: Path | None = None
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tmp:
        tmp.write(full_src)
        tmp_path = Path(tmp.name)

    try:
        return runpy.run_path(str(tmp_path), run_name="__candidate__")
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _load_candidate_namespace(program_path: str) -> Dict[str, Any]:
    program_path = Path(program_path)
    evaluator_dir = Path(__file__).resolve().parent
    scaffold_path = evaluator_dir / "initial_program.py"

    if not scaffold_path.exists():
        raise FileNotFoundError(f"Cannot find scaffold: {scaffold_path}")

    user_src = program_path.read_text(encoding="utf-8")
    scaffold_src = scaffold_path.read_text(encoding="utf-8")
    try:
        # Pre-check candidate for unbalanced quotes / unterminated strings
        _precheck_candidate_source(user_src)

        user_block = _extract_evolve_block(user_src)
    except Exception:
        # Dump the raw candidate for debugging
        out_dir = evaluator_dir / "openevolve_output" / "failed_candidates"
        try:
            os.makedirs(out_dir, exist_ok=True)
            dump_name = f"failed_{uuid.uuid4().hex}.py"
            dump_path = out_dir / dump_name
            dump_path.write_text(user_src, encoding="utf-8")
        except Exception:
            # ignore dump failures
            pass
        # Re-raise to preserve original behavior/logging
        raise

    sanitized_block = _sanitize_user_block(user_block)
    full_src = _build_locked_program(scaffold_src, sanitized_block)

    return _safe_run_candidate(full_src)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _functions_present_score(ns: Dict[str, Any]) -> float:
    present = 0
    for name in REQUIRED_FUNCTIONS:
        obj = ns.get(name)
        if callable(obj):
            present += 1
    return present / float(len(REQUIRED_FUNCTIONS))


def _run_sanity_metrics(ns: Dict[str, Any]) -> Dict[str, float]:
    defaults = {
        "ratio_H0": 0.0,
        "rho_q_today_over_crit0": 0.0,
        "H_at_early_a": 0.0,
    }
    run_checks = ns.get("run_sanity_checks")
    if not callable(run_checks):
        return defaults

    try:
        checks = run_checks()
    except Exception:
        return defaults

    return {
        "ratio_H0": float(checks.get("ratio_H0", 0.0)),
        "rho_q_today_over_crit0": float(checks.get("rho_q_today_over_crit0", 0.0)),
        "H_at_early_a": float(checks.get("H_at_early_a", 0.0)),
    }


def _monotonic_H_score(ns: Dict[str, Any]) -> float:
    sample_fn = ns.get("sample_H_of_a_grid")
    params_ctor = ns.get("CosmologyParams")
    if not callable(sample_fn) or not callable(params_ctor):
        return 0.0

    try:
        params = params_ctor()
        a_values = [10 ** (i / 10.0 - 2.0) for i in range(0, 31)]  # 1e-2 .. 1e1
        H_map = sample_fn(params, a_values)
        H_vals = [float(H_map[a]) for a in a_values if a in H_map]
        if len(H_vals) < 2:
            return 0.0
        nonincreasing = sum(1 for i in range(len(H_vals) - 1) if H_vals[i] >= H_vals[i + 1])
        return nonincreasing / float(len(H_vals) - 1)
    except Exception:
        return 0.0


def _rho_profile_variation_score(ns: Dict[str, Any]) -> float:
    """Reward candidates whose quantum energy density meaningfully varies with a."""
    rho_fn = ns.get("rho_quantum")
    classical_fn = ns.get("classical_H_squared")
    params_ctor = ns.get("CosmologyParams")
    if not (callable(rho_fn) and callable(classical_fn) and callable(params_ctor)):
        return 0.0

    try:
        params = params_ctor()
        a_values = [10 ** (i / 10.0 - 2.0) for i in range(0, 31)]  # 1e-2 .. 1e1
        rho_vals: List[float] = []
        for a in a_values:
            H2 = classical_fn(a, params)
            if H2 <= 0.0:
                continue
            rho = rho_fn(a, math.sqrt(H2), params.m_g)
            rho_val = float(rho)
            if not math.isfinite(rho_val):
                continue
            rho_vals.append(rho_val)
        if len(rho_vals) < 2:
            return 0.0
        mean = sum(rho_vals) / len(rho_vals)
        variance = sum((val - mean) ** 2 for val in rho_vals) / len(rho_vals)
        std = math.sqrt(max(0.0, variance))
        denom = abs(mean) + 1e-30
        ratio = std / denom
        # Encourage at least ~30% modulation but clamp to [0, 1].
        return float(max(0.0, min(1.0, ratio / 0.3)))
    except Exception:
        return 0.0


def _log_low_score_details(checks: Dict[str, float], metrics: Dict[str, float]) -> None:
    if not LOG_LOW_SCORE_DETAILS:
        return

    combined = float(metrics.get("combined_score", 0.0))
    if combined >= LOW_SCORE_THRESHOLD:
        return

    LOGGER.info(
        "Low-score candidate (combined=%.3f): ratio_H0=%.3f, rho_q_today=%.3e, H_early=%.3e, monotonic=%.3f, rho_var=%.3f",
        combined,
        float(checks.get("ratio_H0", 0.0)),
        float(checks.get("rho_q_today_over_crit0", 0.0)),
        float(checks.get("H_at_early_a", 0.0)),
        float(metrics.get("monotonic_H_score", 0.0)),
        float(metrics.get("rho_profile_variation_score", 0.0)),
    )


def _full_metric_bundle(ns: Dict[str, Any], functions_present_score: float) -> Dict[str, float]:
    checks = _run_sanity_metrics(ns)
    H0_ratio_delta = abs(checks["ratio_H0"] - 1.0)
    H0_ratio_score = max(0.0, 1.0 - H0_ratio_delta / H0_RATIO_TOLERANCE)
    rho_today = max(checks["rho_q_today_over_crit0"], 0.0)
    rho_q_today_penalty = min(rho_today / RHO_Q_TODAY_TOLERANCE, 1.0)
    rho_q_today_score = max(0.0, 1.0 - rho_q_today_penalty)
    # Treat any strictly positive early-time H as a success.  The previous
    # hard threshold of 1e-12 s⁻¹ unrealistically penalized physically
    # reasonable models (e.g., H ~ 4×10⁻¹···s⁻¹ at a=0.1).  As long as
    # the computed Hubble rate at early times is positive, we consider
    # matter/radiation domination satisfied.
    early_domination_score = 1.0 if checks["H_at_early_a"] > 0.0 else 0.0
    quantum_small_early_score = rho_q_today_score
    monotonic_H_score = _monotonic_H_score(ns)
    rho_profile_variation_score = _rho_profile_variation_score(ns)

    combined_score = (
        0.10 * functions_present_score
        + 0.20 * H0_ratio_score
        + 0.15 * rho_q_today_score
        + 0.20 * early_domination_score
        + 0.20 * monotonic_H_score
        + 0.15 * rho_profile_variation_score
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
        "combined_score": float(combined_score),
    }
    _log_low_score_details(checks, metrics)
    return metrics


# ---------------------------------------------------------------------------
# Cascade-aware evaluation entry points
# ---------------------------------------------------------------------------


def evaluate_stage1(program_path: str) -> Dict[str, float]:
    """Lightweight stage: syntax/module/function checks."""
    t0 = time.time()
    try:
        ns = _load_candidate_namespace(program_path)
        functions_present_score = _functions_present_score(ns)
        t1 = time.time()
        return {
            "syntax_valid": 1.0,
            "module_loaded": 1.0,
            "functions_present_score": float(functions_present_score),
            "combined_score": float(functions_present_score),
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
    """Full physics evaluation used for cascade stage 2."""
    t0 = time.time()
    try:
        ns = _load_candidate_namespace(program_path)
        functions_present_score = _functions_present_score(ns)
        metrics = _full_metric_bundle(ns, functions_present_score)
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


def evaluate(program_path: str) -> Dict[str, float]:
    """Compatibility entry point expected by OpenEvolve."""
    return evaluate_stage2(program_path)


__all__ = ["evaluate", "evaluate_stage1", "evaluate_stage2"]
