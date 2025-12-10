import os

# Path to the file causing the error
file_path = r"examples\mtdc_hubble\evaluator.py"

# The CORRECT content with 'def evaluate(fn_dict)'
new_content = r'''"""
Evaluator for MTDC in OpenEvolve
Evaluates SN + BAO + CMB consistency
"""

import math
import numpy as np
from scipy.integrate import quad

# We import from initial_program mainly for testing, 
# but OpenEvolve passes the *evolved* functions in fn_dict.

c = 2.99792458e5

# Mock data (replace with real Pantheon+, BAO, Planck as needed)
z_SN = np.array([0.01, 0.05, 0.1, 0.2])
H_SN = np.array([73.0, 73.2, 73.1, 72.8])
sigma_SN = np.array([1.5, 1.5, 1.5, 1.5])

def evaluate(fn_dict):
    """ 
    Entry point for OpenEvolve.
    fn_dict contains the functions evolved (e.g., 'predict_H').
    """
    # 1. Extract the evolved function
    try:
        # The framework executes the code and passes the resulting locals in fn_dict
        H_func = fn_dict.get("predict_H")
        if H_func is None:
            return {"score": 0.0, "error": "predict_H not found"}
    except Exception as e:
        return {"score": 0.0, "error": str(e)}

    # 2. Define metric (Chi-squared)
    def calculate_chi2(H_function):
        tot = 0.0
        # Simple loop over mock SN data
        for z, Hobs, sig in zip(z_SN, H_SN, sigma_SN):
            try:
                # We test at fixed parameters H0=73, Omega_m=0.3 for the 'evolution' metric
                # In a full run, you might want to marginalize over H0/Omega_m
                Hth = H_function(z, 73.0, 0.3) 
                
                # Sanity check for NaNs or negative H
                if math.isnan(Hth) or Hth <= 0:
                    return 1e9 
                
                tot += ((Hobs - Hth) / sig)**2
            except:
                return 1e9
        return tot

    # 3. Calculate
    chi = calculate_chi2(H_func)
    
    # 4. Convert Chi2 to Score (0 to 1)
    # Using a Gaussian likelihood form: score ~ exp(-chi2/2)
    # Added a small factor to prevent overly small floats if chi2 is huge
    score = math.exp(-0.5 * min(chi, 100.0))

    return {
        "chi2": chi,
        "score": float(score),
        "details": f"Chi2: {chi:.4f}"
    }
'''

# Write the file
with open(file_path, "w", encoding="utf-8") as f:
    f.write(new_content)

print(f"Successfully fixed: {file_path}")
