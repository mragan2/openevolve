import os
import textwrap

# Path to the file
file_path = r"examples\mtdc_hubble\evaluator.py"

# The content matching your requested structure
new_content = textwrap.dedent(r'''
"""
Physics Evaluator for MTDC (Hubble Tension).
Matches style of Massive Graviton Cosmology evaluator.
"""
import importlib.util
import math
import sys
import tempfile
from pathlib import Path
import numpy as np

# --- DATA ---
# Mock Pantheon+ (z, H, error)
z_SN = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0])
H_SN = np.array([73.0, 73.2, 73.1, 72.8, 68.0, 65.0, 62.0]) 
sigma_SN = np.array([1.5, 1.5, 1.5, 1.5, 2.0, 2.5, 3.0])

# Constants for evaluation
H0_FIDUCIAL = 73.0
OMEGA_M_FIDUCIAL = 0.3

def _sanitize_candidate_file(path):
    """ Removes Markdown artifacts if present. """
    try:
        text = path.read_text(encoding="utf-8")
        if "```" in text:
            lines = [l for l in text.splitlines() if not l.strip().startswith("```")]
            path.write_text("\n".join(lines), encoding="utf-8")
    except: pass

def _load_module_from_path(program_path):
    """ Loads the candidate program dynamically. """
    path = Path(program_path)
    _sanitize_candidate_file(path)
    
    try:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = module
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"Import Error: {e}")
        return None

def _load_module_from_string(source_code):
    """ Handles case where input is raw source string. """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write(source_code)
        tmp_path = tmp.name
    return _load_module_from_path(tmp_path)

# --- MAIN EVALUATOR ---
def evaluate(program_input):
    metrics = {}
    
    # 1. Load the Candidate Program
    module = None
    if isinstance(program_input, dict):
        # OpenEvolve passed executed locals
        class MockModule: pass
        module = MockModule()
        for k, v in program_input.items():
            setattr(module, k, v)
    elif isinstance(program_input, str):
        # Check if it's a path or code
        if "\n" in program_input or "def " in program_input:
             module = _load_module_from_string(program_input)
        else:
             module = _load_module_from_path(program_input)
    
    if not module:
        return {"combined_score": 0.0, "error": "Could not load module"}

    # 2. Extract Function
    try:
        predict_H = getattr(module, "predict_H", None)
        if not predict_H:
             # Fallback if name changed
             predict_H = getattr(module, "H_MTD_corrected", None)
             
        if not predict_H:
            return {"combined_score": 0.0, "error": "Function predict_H not found"}
    except:
        return {"combined_score": 0.0, "error": "Extraction failed"}

    # 3. Calculate Chi-Squared (The Physics Metric)
    chi2 = 0.0
    try:
        for z, Hobs, sig in zip(z_SN, H_SN, sigma_SN):
            # Evaluate model
            H_th = float(predict_H(z, H0_FIDUCIAL, OMEGA_M_FIDUCIAL))
            
            # Penalize invalid physics (Negative H or NaN)
            if math.isnan(H_th) or H_th <= 0:
                chi2 += 1e5
            else:
                chi2 += ((Hobs - H_th) / sig) ** 2
    except Exception as e:
        return {"combined_score": 0.0, "error": f"Runtime: {str(e)}"}

    # 4. Calculate Combined Score
    # Convert Chi2 to 0-1 range. Perfect match = 1.0.
    # We use a gaussian likelihood form.
    # If Chi2 is high, score approaches 0.
    
    metrics["chi2"] = chi2
    
    # Stability check (Bonus for returning reasonable numbers)
    metrics["stability"] = 1.0
    
    # Calculate score
    score = math.exp(-0.5 * min(chi2, 100.0))
    
    metrics["combined_score"] = float(score)
    metrics["score"] = float(score) # OpenEvolve display
    metrics["details"] = f"Chi2: {chi2:.2f}"
    
    return metrics
''')

# Write the file
with open(file_path, "w", encoding="utf-8") as f:
    f.write(new_content)

print(f"Successfully updated: {file_path}")
