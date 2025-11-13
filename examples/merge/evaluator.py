"""
Evaluator for Black Hole Information & Semiclassical Unification
Conforms to README requirements:
- Returns a DICTIONARY with 'combined_score' (primary metric)
- Always includes 'combined_score' even on error
- Adds 'error' on failure for traceability
- Uses Windows-safe quoting for temp wrapper paths
"""

from __future__ import annotations

import numpy as np
import time
import os
import subprocess
import tempfile
import traceback
import sys
import pickle
from typing import Dict, Any


class TimeoutError(Exception):
    pass


def validate_results(results: Dict[str, Any]) -> bool:
    """
    Validate the output dictionary from the black hole simulation.
    Required keys: 'time', 'S_rad_bits', 'kappa'
    """
    if not isinstance(results, dict):
        print("Output is not a dictionary")
        return False

    required_keys = ["time", "S_rad_bits", "kappa"]
    for key in required_keys:
        if key not in results:
            print(f"Missing required key: {key}")
            return False

    # time and S_rad_bits must be list/array without NaN
    for key in ["time", "S_rad_bits"]:
        arr = results[key]
        if not isinstance(arr, (list, np.ndarray)):
            print(f"{key} is not a list or numpy array")
            return False
        arr = np.asarray(arr, dtype=float)
        if np.isnan(arr).any():
            print(f"NaN values detected in {key}")
            return False

    # kappa must be finite number
    kappa = results["kappa"]
    if not isinstance(kappa, (int, float)) or not np.isfinite(kappa):
        print(f"Invalid kappa: {kappa}")
        return False

    return True


def run_with_timeout(program_path: str, timeout_seconds: int = 30) -> Dict[str, Any]:
    """
    Run the candidate program in a separate Python process with a timeout.
    Writes a short wrapper that imports the program and calls evolve_page_and_unify().
    Uses repr(...) when embedding paths so Windows backslashes are escaped.
    """
    # Make a temp wrapper script
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        wrapper_path = temp_file.name

    results_path = f"{wrapper_path}.results"

    # Windows-safe embedding of paths using !r (repr)
    wrapper_code = f"""
import sys, os, pickle, traceback, importlib.util
print(f"Running in subprocess, Python version: {{sys.version}}")
print("Program path:", {program_path!r})
sys.path.insert(0, os.path.dirname({program_path!r}))
try:
    # Read the candidate program bytes and robustly decode them so
    # Python can import the file even if it's not UTF-8 encoded.
    cleaned_path = {program_path!r} + ".cleaned"
    with open({program_path!r}, "rb") as pf:
        raw = pf.read()
    try:
        text = raw.decode("utf-8")
    except Exception:
        try:
            text = raw.decode("latin-1")
        except Exception:
            text = raw.decode("utf-8", errors="replace")
    with open(cleaned_path, "w", encoding="utf-8") as cf:
        cf.write(text)

    spec = importlib.util.spec_from_file_location("program", cleaned_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)

    # Call main entry
    if hasattr(program, 'EvaporationConfig'):
        config = program.EvaporationConfig(grid_points=200, M0=5.0)
        results = program.evolve_page_and_unify(config)
    else:
        results = program.evolve_page_and_unify()

    # Keep only pickle-safe items (avoid heavy/symbolic objects)
    safe = {{}}
    for k, v in results.items():
        if k == 'unified_equation':
            safe[k] = str(v)
        elif k == 'config':
            # Skip or stringify config
            continue
        else:
            safe[k] = v

    with open({results_path!r}, "wb") as f:
        pickle.dump(safe, f)
    print("Results saved to", {results_path!r})

    except Exception as e:
    print("Error in subprocess:", str(e))
    traceback.print_exc()
    with open({results_path!r}, "wb") as f:
        pickle.dump({{'error': str(e)}}, f)
    print("Error saved to", {results_path!r})
finally:
    try:
        if os.path.exists(cleaned_path):
            os.unlink(cleaned_path)
    except Exception:
        pass
"""

    # Write wrapper code
    with open(wrapper_path, "w", encoding="utf-8") as f:
        f.write(wrapper_code)

    try:
        proc = subprocess.Popen(
            [sys.executable, wrapper_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise TimeoutError(f"Process timed out after {timeout_seconds} seconds")

        # Always show child output for debugging
        if stdout:
            print("Subprocess stdout:", stdout)
        if stderr:
            print("Subprocess stderr:", stderr)

        if proc.returncode != 0:
            raise RuntimeError(f"Process exited with code {proc.returncode}")

        if not os.path.exists(results_path):
            raise RuntimeError("Results file not found")

        with open(results_path, "rb") as f:
            results = pickle.load(f)

        if isinstance(results, dict) and "error" in results:
            raise RuntimeError(f"Program execution failed: {results['error']}")

        return results

    finally:
        # Cleanup
        for p in (wrapper_path, results_path):
            try:
                if p and os.path.exists(p):
                    os.unlink(p)
            except Exception:
                pass


def evaluate(program_path: str) -> Dict[str, Any]:
    """
    Full evaluation used by Stage 2 and also valid for Stage 1.
    Returns a DICT with 'combined_score' as the primary metric.
    """
    try:
        t0 = time.time()
        results = run_with_timeout(program_path, timeout_seconds=30)
        t1 = time.time()
        eval_time = t1 - t0

        if not results:
            return {
                "combined_score": 0.0,
                "validity": 0.0,
                "page_curve_score": 0.0,
                "physics_consistency": 0.0,
                "eval_time": float(eval_time),
                "kappa_value": 0.0,
                "error": "No results returned",
            }

        valid = validate_results(results)
        if not valid:
            return {
                "combined_score": 0.0,
                "validity": 0.0,
                "page_curve_score": 0.0,
                "physics_consistency": 0.0,
                "eval_time": float(eval_time),
                "kappa_value": 0.0,
                "error": "Invalid results structure",
            }

        # ----- Metrics -----
        S_rad = np.asarray(results["S_rad_bits"], dtype=float)
        max_entropy = float(np.max(S_rad))
        final_entropy = float(S_rad[-1])

        # Page-curve recovery score in [0,1]
        if max_entropy > 1e-12:
            recovery_ratio = 1.0 - (final_entropy / max_entropy)
            page_curve_score = float(np.clip(recovery_ratio, 0.0, 1.0))
        else:
            page_curve_score = 0.0

        kappa = float(results["kappa"])
        if np.isfinite(kappa) and 1e-5 < abs(kappa) < 1000.0:
            physics_consistency = 1.0
        else:
            physics_consistency = 0.5 if np.isfinite(kappa) else 0.0

        validity = 1.0  # passed structure checks

        # Combined score (cap at 1.0)
        combined_score = min(
            1.0,
            (validity * 0.3) + (page_curve_score * 0.4) + (physics_consistency * 0.3)
            + (0.1 if "unified_equation" in results and results["unified_equation"] else 0.0)
        )

        return {
            "combined_score": float(combined_score),     # REQUIRED
            # Additional raw metrics (usable as feature_dimensions)
            "validity": float(validity),
            "page_curve_score": float(page_curve_score),
            "physics_consistency": float(physics_consistency),
            "kappa_value": float(kappa),
            "eval_time": float(eval_time),
        }

    except Exception as e:
        print(f"Evaluation failed completely: {e}")
        traceback.print_exc()
        return {
            "combined_score": 0.0,   # REQUIRED even on error
            "page_curve_score": 0.0,
            "physics_consistency": 0.0,
            "eval_time": 0.0,
            "validity": 0.0,
            "kappa_value": 0.0,
            "error": str(e),
        }


def evaluate_stage1(program_path: str) -> Dict[str, Any]:
    """
    Fast validation stage.
    Still returns a DICT with 'combined_score'.
    """
    try:
        results = run_with_timeout(program_path, timeout_seconds=15)
        if not validate_results(results):
            return {"combined_score": 0.0, "page_curve_score": 0.0, "physics_consistency": 0.0, "eval_time": 0.0, "validity": 0.0, "kappa_value": 0.0, "error": "Invalid results structure"}

        S_rad = results.get("S_rad_bits", [])
        if not isinstance(S_rad, (list, np.ndarray)) or len(S_rad) == 0:
            return {"combined_score": 0.0, "page_curve_score": 0.0, "physics_consistency": 0.0, "eval_time": 0.0, "validity": 0.0, "kappa_value": 0.0, "error": "Empty entropy array"}

        # Quick pass: it ran and produced sane structure
        return {"combined_score": 1.0, "validity": 1.0, "page_curve_score": 1.0, "physics_consistency": 1.0, "eval_time": 0.0, "kappa_value": 0.0}

    except TimeoutError as e:
        return {"combined_score": 0.0, "page_curve_score": 0.0, "physics_consistency": 0.0, "eval_time": 0.0, "validity": 0.0, "kappa_value": 0.0, "error": f"Timeout: {e}"}
    except Exception as e:
        return {"combined_score": 0.0, "page_curve_score": 0.0, "physics_consistency": 0.0, "eval_time": 0.0, "validity": 0.0, "kappa_value": 0.0, "error": str(e)}


def evaluate_stage2(program_path: str) -> Dict[str, Any]:
    """Full physics evaluation."""
    return evaluate(program_path)

