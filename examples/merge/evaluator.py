"""
Evaluator for Black Hole Information & Semiclassical Unification
Based on the Circle Packing evaluator template.
"""

import importlib.util
import numpy as np
import time
import os
import signal
import subprocess
import tempfile
import traceback
import sys
import pickle


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Function execution timed out")


def validate_results(results):
    """
    Validate the output dictionary from the black hole simulation.

    Args:
        results: Dictionary containing simulation data

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(results, dict):
        print("Output is not a dictionary")
        return False

    required_keys = ["time", "S_rad_bits", "kappa"]
    for key in required_keys:
        if key not in results:
            print(f"Missing required key: {key}")
            return False

    # Check for NaN values in arrays
    for key in ["time", "S_rad_bits"]:
        arr = results[key]
        if isinstance(arr, (list, np.ndarray)):
            if np.isnan(arr).any():
                print(f"NaN values detected in {key}")
                return False
        else:
            print(f"{key} is not a list or numpy array")
            return False

    # Check kappa
    kappa = results["kappa"]
    if not isinstance(kappa, (int, float)):
        print(f"kappa is not a number: {type(kappa)}")
        return False
    
    if np.isnan(kappa):
        print("kappa is NaN")
        return False

    return True


def run_with_timeout(program_path, timeout_seconds=20):
    """
    Run the program in a separate process with timeout
    using a simple subprocess approach.

    Args:
        program_path: Path to the program file
        timeout_seconds: Maximum execution time in seconds

    Returns:
        The results dictionary from evolve_page_and_unify()
    """
    # Create a temporary file to execute
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        # Write a script that executes the program and saves results
        script = f"""
import sys
import numpy as np
import os
import pickle
import traceback

# Add the directory to sys.path
sys.path.insert(0, os.path.dirname('{program_path}'))

# Debugging info
print(f"Running in subprocess, Python version: {{sys.version}}")
print(f"Program path: {program_path}")

try:
    # Import the program
    spec = __import__('importlib.util').util.spec_from_file_location("program", '{program_path}')
    program = __import__('importlib.util').util.module_from_spec(spec)
    spec.loader.exec_module(program)
    
    # Run the main evolution function
    print("Calling evolve_page_and_unify()...")
    
    # Try to construct a config if the class exists, otherwise call with defaults
    if hasattr(program, 'EvaporationConfig'):
        print("Found EvaporationConfig, using test configuration...")
        # Use a slightly smaller grid for faster evaluation
        config = program.EvaporationConfig(grid_points=200, M0=5.0)
        results = program.evolve_page_and_unify(config)
    else:
        print("EvaporationConfig not found, calling with defaults...")
        results = program.evolve_page_and_unify()
        
    print(f"evolve_page_and_unify() returned successfully.")

    # Ensure results are pickle-able (convert sympy objects if necessary or exclude them)
    # We mainly care about the numerical data for evaluation
    safe_results = {{}}
    for k, v in results.items():
        if k == 'unified_equation':
            # Store string representation of symbolic equation to avoid pickle issues with SymPy across processes
            safe_results[k] = str(v)
        elif k == 'config':
            # Config might be a dataclass, skip it or convert to dict if needed, 
            # but for eval we mostly need the arrays. Let's skip complex objects.
            pass
        else:
            safe_results[k] = v

    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump(safe_results, f)
    print(f"Results saved to {temp_file.name}.results")
    
except Exception as e:
    # If an error occurs, save the error instead
    print(f"Error in subprocess: {{str(e)}}")
    traceback.print_exc()
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump({{'error': str(e)}}, f)
    print(f"Error saved to {temp_file.name}.results")
"""
        temp_file.write(script.encode())
        temp_file_path = temp_file.name

    results_path = f"{temp_file_path}.results"

    try:
        # Run the script with timeout
        process = subprocess.Popen(
            [sys.executable, temp_file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            exit_code = process.returncode

            # Always print output for debugging purposes
            print(f"Subprocess stdout: {stdout.decode()}")
            if stderr:
                print(f"Subprocess stderr: {stderr.decode()}")

            # Still raise an error for non-zero exit codes
            if exit_code != 0:
                raise RuntimeError(f"Process exited with code {exit_code}")

            # Load the results
            if os.path.exists(results_path):
                with open(results_path, "rb") as f:
                    results = pickle.load(f)

                # Check if an error was returned
                if "error" in results:
                    raise RuntimeError(f"Program execution failed: {results['error']}")

                return results
            else:
                raise RuntimeError("Results file not found")

        except subprocess.TimeoutExpired:
            # Kill the process if it times out
            process.kill()
            process.wait()
            raise TimeoutError(f"Process timed out after {timeout_seconds} seconds")

    finally:
        # Clean up temporary files
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if os.path.exists(results_path):
            os.unlink(results_path)


def evaluate(program_path):
    """
    Evaluate the black hole simulation program.

    Args:
        program_path: Path to the program file

    Returns:
        Dictionary of metrics
    """
    try:
        start_time = time.time()

        # Use subprocess to run with timeout
        results = run_with_timeout(
            program_path, timeout_seconds=30  # Should be fast
        )

        end_time = time.time()
        eval_time = end_time - start_time

        if not results:
            print("No results returned.")
            return {
                "validity": 0.0,
                "page_curve_score": 0.0,
                "physics_consistency": 0.0,
                "eval_time": float(eval_time),
                "combined_score": 0.0,
            }

        # Validate structure
        valid = validate_results(results)
        if not valid:
            return {
                "validity": 0.0,
                "page_curve_score": 0.0,
                "physics_consistency": 0.0,
                "eval_time": float(eval_time),
                "combined_score": 0.0,
            }

        # --- Physics Checks ---
        
        # 1. Page Curve Shape Analysis
        S_rad = np.array(results["S_rad_bits"])
        
        # Information recovery check: Entropy should eventually decrease
        # Specifically, final entropy should be significantly lower than max entropy
        max_entropy = np.max(S_rad)
        final_entropy = S_rad[-1]
        
        # Page curve score: 1.0 if information returns (final < 0.1 * max), 
        # scaled down if it stays high (which would imply information loss)
        if max_entropy > 1e-6:
            recovery_ratio = 1.0 - (final_entropy / max_entropy)
            # Clip to [0, 1] just in case
            page_curve_score = float(np.clip(recovery_ratio, 0.0, 1.0))
        else:
            page_curve_score = 0.0

        # 2. Semiclassical Consistency (Kappa)
        kappa = results["kappa"]
        
        # Kappa represents the coupling required to save unitarity.
        # It should be finite. If it's extremely large, the approximation might be breaking down.
        # If it's zero, there's no backreaction (which might be wrong for this model).
        physics_consistency = 0.0
        if np.isfinite(kappa):
            # Check if it's within a "sane" magnitude (heuristic)
            if 1e-5 < abs(kappa) < 1000.0:
                physics_consistency = 1.0
            else:
                physics_consistency = 0.5 # Penalize extreme values
        
        # 3. Unified Equation Check
        has_equation = "unified_equation" in results and results["unified_equation"] is not None
        equation_bonus = 0.1 if has_equation else 0.0

        # Combined score
        validity = 1.0
        combined_score = (validity * 0.3) + (page_curve_score * 0.4) + (physics_consistency * 0.3) + equation_bonus
        combined_score = min(1.0, combined_score) # Cap at 1.0

        print(
            f"Evaluation: valid={valid}, max_S={max_entropy:.2f}, final_S={final_entropy:.2f}, "
            f"kappa={kappa:.4e}, score={combined_score:.4f}, time={eval_time:.2f}s"
        )

        return {
            "validity": float(validity),
            "page_curve_score": float(page_curve_score),
            "physics_consistency": float(physics_consistency),
            "kappa_value": float(kappa),
            "eval_time": float(eval_time),
            "combined_score": float(combined_score),
        }

    except Exception as e:
        print(f"Evaluation failed completely: {str(e)}")
        traceback.print_exc()
        return {
            "validity": 0.0,
            "page_curve_score": 0.0,
            "physics_consistency": 0.0,
            "eval_time": 0.0,
            "combined_score": 0.0,
        }


def evaluate_stage1(program_path):
    """
    First stage evaluation - quick validation check
    """
    try:
        # Use the subprocess approach with a short timeout
        results = run_with_timeout(program_path, timeout_seconds=15)

        valid = validate_results(results)
        
        if not valid:
             return {"validity": 0.0, "combined_score": 0.0, "error": "Invalid results structure"}

        # Check if S_rad is not empty and has data
        S_rad = results.get("S_rad_bits", [])
        if len(S_rad) == 0:
            return {"validity": 0.5, "combined_score": 0.0, "error": "Empty entropy array"}

        return {
            "validity": 1.0,
            "sum_radii": 0.0, # Legacy key for compatibility if needed
            "combined_score": 1.0, # Pass if it runs and returns valid structure
        }

    except TimeoutError as e:
        print(f"Stage 1 evaluation timed out: {e}")
        return {"validity": 0.0, "combined_score": 0.0, "error": "Timeout"}
    except Exception as e:
        print(f"Stage 1 evaluation failed: {e}")
        return {"validity": 0.0, "combined_score": 0.0, "error": str(e)}


def evaluate_stage2(program_path):
    """
    Second stage evaluation - full physics evaluation
    """
    return evaluate(program_path)