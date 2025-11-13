"""
Evaluator for Black Hole Information & Semiclassical Unification
FIXED VERSION — prevents 'NoneType' spec.loader errors
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
import importlib.util
from typing import Dict, Any


class TimeoutError(Exception):
    pass


# ============================================================
# FIX 1: safe, explicit module loader to prevent spec=None
# ============================================================
def safe_load_module(path: str):
    """Load module robustly. Raises clear errors instead of 'spec.loader' crash."""
    if not isinstance(path, str):
        raise TypeError(f"Program path must be string, got: {path!r}")

    path = os.path.abspath(path)

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Candidate program not found: {path}")

    spec = importlib.util.spec_from_file_location("program", path)

    if spec is None:
        raise ImportError(
            f"Could not create import spec for file: {path}\n"
            "This is usually caused by:\n"
            " - invalid filename\n"
            " - hidden characters\n"
            " - missing .py extension\n"
            " - Windows quoting issues\n"
        )

    if spec.loader is None:
        raise ImportError(
            f"Import spec created but loader is None for: {path}\n"
            "This is the exact root cause of the old error."
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


# ============================================================
# VALIDATION LOGIC
# ============================================================
def validate_results(results: Dict[str, Any]) -> bool:
    required = ["time", "S_rad_bits", "kappa"]

    if not isinstance(results, dict):
        print("Results not a dict")
        return False

    for key in required:
        if key not in results:
            print(f"Missing required key: {key}")
            return False

    for key in ["time", "S_rad_bits"]:
        arr = np.asarray(results[key], dtype=float)
        if arr.ndim == 0 or np.isnan(arr).any():
            print(f"Invalid array for {key}")
            return False

    k = results["kappa"]
    if not isinstance(k, (int, float)) or not np.isfinite(k):
        print("Invalid kappa")
        return False

    return True


# ============================================================
# EXECUTION WITH TIMEOUT — FIXED
# ============================================================
def run_with_timeout(program_path: str, timeout_seconds: int = 30) -> Dict[str, Any]:

    program_path = os.path.abspath(program_path)

    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
        wrapper_path = tmp.name

    results_path = wrapper_path + ".results"

    # Windows-safe quoting via repr
    wrapper_code = f"""
import sys, os, pickle, traceback, importlib.util

prog_path = {program_path!r}
prog_path = os.path.abspath(prog_path)

# === FIX 2: check existence before loading ===
if not os.path.isfile(prog_path):
    with open({results_path!r}, "wb") as f:
        pickle.dump({{'error': f'Program path invalid: {{prog_path}}'}}, f)
    raise SystemExit()

spec = importlib.util.spec_from_file_location("program", prog_path)
if spec is None or spec.loader is None:
    with open({results_path!r}, "wb") as f:
        pickle.dump({{'error': 'Failed to construct loader for candidate program'}}, f)
    raise SystemExit()

program = importlib.util.module_from_spec(spec)
spec.loader.exec_module(program)

try:
    if hasattr(program, "EvaporationConfig"):
        cfg = program.EvaporationConfig(grid_points=200, M0=5.0)
        results = program.evolve_page_and_unify(cfg)
    else:
        results = program.evolve_page_and_unify()

    # filter unsafe objects
    safe = {{}}
    for k, v in results.items():
        if k == "unified_equation":
            safe[k] = str(v)
        elif k == "config":
            continue
        else:
            safe[k] = v

    with open({results_path!r}, "wb") as f:
        pickle.dump(safe, f)

except Exception as e:
    with open({results_path!r}, "wb") as f:
        pickle.dump({{'error': str(e)}}, f)
"""

    # write wrapper
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
            raise TimeoutError("Timed out")

        if stdout:
            print("stdout:", stdout)
        if stderr:
            print("stderr:", stderr)

        if not os.path.exists(results_path):
            raise RuntimeError("Subprocess did not write results")

        with open(results_path, "rb") as f:
            out = pickle.load(f)

        if "error" in out:
            raise RuntimeError(f"Program execution failed: {out['error']}")

        return out

    finally:
        for p in (wrapper_path, results_path):
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except:
                pass


# ============================================================
# MAIN EVALUATION
# ============================================================
def evaluate(program_path: str) -> Dict[str, Any]:
    try:
        t0 = time.time()
        results = run_with_timeout(program_path, timeout_seconds=30)
        t1 = time.time()

        if not validate_results(results):
            return {"combined_score": 0.0, "error": "Invalid results structure"}

        S = np.asarray(results["S_rad_bits"], float)
        maxS = float(np.max(S))
        finalS = float(S[-1])
        if maxS > 0:
            page_score = float(np.clip(1 - finalS / maxS, 0, 1))
        else:
            page_score = 0.0

        kappa = float(results["kappa"])
        physics_score = 1.0 if (np.isfinite(kappa) and 1e-5 < abs(kappa) < 1000) else 0.5

        combined = min(1.0, 0.3 + 0.4 * page_score + 0.3 * physics_score)

        return {
            "combined_score": combined,
            "page_curve_score": page_score,
            "physics_consistency": physics_score,
            "kappa_value": kappa,
            "eval_time": t1 - t0,
        }

    except Exception as e:
        traceback.print_exc()
        return {"combined_score": 0.0, "error": str(e)}


def evaluate_stage1(program_path: str) -> Dict[str, Any]:
    return evaluate(program_path)


def evaluate_stage2(program_path: str) -> Dict[str, Any]:
    return evaluate(program_path)


if __name__ == "__main__":
    print(evaluate(sys.argv[1]))
