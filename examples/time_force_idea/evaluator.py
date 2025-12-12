"""
Hybrid Evaluator: Bridges the gap between "Wild Exploration" and "Strict Structure".
"""
import ast
import importlib.util
import sys
import math
from pathlib import Path
from typing import Dict, Any, List

# -------------------------------------------------------------------
# 1. STATIC ANALYSIS (Keywords & Syntax)
# -------------------------------------------------------------------

KEYWORDS = [
    # Concepts
    "czas", "time", "force", "siła", "entropy", "entropia",
    "horizon", "horyzont", "event", "zdarzeń", "singularity", "osobliwość",
    "relativity", "względność", "observer", "obserwator",
    # Boone's Philosophy
    "blur", "rozmycie", "uncertainty", "niepewność", "decay", "rozpad",
    "stop", "zatrzymanie", "reverse", "odwrócenie"
]

def _score_syntax(src: str) -> float:
    try:
        ast.parse(src)
        return 1.0
    except SyntaxError:
        return 0.0

def _score_keywords(src: str) -> float:
    """Awards points for using domain-specific vocabulary."""
    text = src.lower()
    hits = sum(1 for w in KEYWORDS if w in text)
    # Saturation at 8 keywords = max score
    return min(1.0, hits / 8.0)

def _score_structure(src: str) -> float:
    """
    Rewards complexity: Classes, Functions, and Inheritance.
    But doesn't demand specific class names yet.
    """
    try:
        tree = ast.parse(src)
    except:
        return 0.0

    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    
    # Bonus for inheritance (evolutionary complexity)
    has_inheritance = any(c.bases for c in classes)
    
    # We want at least 1 class and 2 functions
    score_cls = min(1.0, len(classes) / 2.0)
    score_fn = min(1.0, len(funcs) / 3.0)
    
    base = (score_cls + score_fn) / 2
    if has_inheritance:
        base += 0.2
        
    return min(1.0, base)

# -------------------------------------------------------------------
# 2. DYNAMIC ANALYSIS (Runtime Execution)
# -------------------------------------------------------------------

def _run_physics_demo(code_string: str) -> Dict[str, Any]:
    """
    Attempts to execute the 'demo()' function in the candidate code.
    Returns metrics based on physics behavior.
    """
    try:
        # Create isolated module
        spec = importlib.util.spec_from_loader("candidate_dynamic", loader=None)
        module = importlib.util.module_from_spec(spec)
        exec(code_string, module.__dict__)
        
        # Check for demo function
        if not hasattr(module, "demo"):
            return {"run_success": False, "error": "No demo() found"}
            
        # RUN IT
        result = module.demo()
        
        if not isinstance(result, dict):
            return {"run_success": True, "valid_output": False}

        return {
            "run_success": True, 
            "valid_output": True,
            "data": result
        }
        
    except Exception as e:
        return {"run_success": False, "error": str(e)}

# -------------------------------------------------------------------
# MAIN EVALUATION LOGIC
# -------------------------------------------------------------------

def evaluate_stage1(program_path: str) -> Dict[str, float]:
    """
    Stage 1: The 'Bouncer'.
    Checks if code is valid Python and mentions relevant topics.
    """
    path = Path(program_path)
    try:
        src = path.read_text(encoding="utf-8")
    except:
        return {"combined_score": 0.0}

    syntax = _score_syntax(src)
    if syntax < 1.0:
        return {"combined_score": 0.0, "syntax": 0.0}

    # If it's valid python, how interesting is it?
    keywords = _score_keywords(src)
    
    return {
        "combined_score": keywords, # Pass through if it uses good words
        "syntax": syntax,
        "idea_alignment": keywords
    }

def evaluate_stage2(program_path: str) -> Dict[str, float]:
    """
    Stage 2: The 'Engineer'.
    Checks if the code actually RUNS and produces physics.
    """
    path = Path(program_path)
    src = path.read_text(encoding="utf-8")
    
    # Run the demo
    runtime = _run_physics_demo(src)
    
    score = 0.0
    if runtime.get("run_success"):
        score += 0.5 # It runs!
        if runtime.get("valid_output"):
            score += 0.3 # It returns a dict!
            
            # Check physics values if present
            data = runtime.get("data", {})
            # Reward movement (Time or Space)
            if abs(data.get("final_pos", 0)) > 0 or abs(data.get("final_time", 0)) > 0:
                score += 0.2
                
    return {
        "combined_score": score,
        "runtime_stability": score
    }

def evaluate(program_path: str) -> Dict[str, float]:
    """
    Full Evaluation: Combines everything.
    Used for the final ranking.
    """
    # 1. Load
    path = Path(program_path)
    try:
        src = path.read_text(encoding="utf-8")
    except:
        return {"combined_score": 0.0}
        
    # 2. Metrics
    syntax = _score_syntax(src)
    if syntax == 0: return {"combined_score": 0.0}
    
    keywords = _score_keywords(src)
    structure = _score_structure(src)
    
    # 3. Runtime
    runtime = _run_physics_demo(src)
    runtime_score = 0.0
    if runtime.get("run_success"):
        runtime_score = 1.0
        # Bonus for returning physics data
        if runtime.get("valid_output"):
            runtime_score += 0.5

    # 4. Final Weighted Score
    # Phase 1 needs Runtime + Keywords
    # Phase 2 needs Structure + Runtime
    
    # We balance them:
    # 40% It Works (Runtime)
    # 30% It's Smart (Keywords)
    # 30% It's Clean (Structure)
    
    final_score = (
        0.4 * min(1.0, runtime_score) +
        0.3 * keywords +
        0.3 * structure
    )
    
    return {
        "combined_score": final_score,
        "syntax": syntax,
        "idea_alignment": keywords,
        "structure": structure,
        "stability": min(1.0, runtime_score), # Map-Elites Dimension
        "complexity": len(src) # Map-Elites Dimension
    }