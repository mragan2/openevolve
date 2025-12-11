import ast
import importlib.util
import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np

# Phase 2 Keywords
KEYWORDS = [
    "relativistic", "lorentz", "stiffness", "newton", "force", 
    "horizon", "event", "singularity", "dilation", "c", "limit"
]

def _sanitize_candidate_file(path: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8")
        if "```" in text:
            lines = [l for l in text.splitlines() if not l.strip().startswith("```")]
            path.write_text("\n".join(lines), encoding="utf-8")
    except Exception:
        pass

def _load_source(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""

def evaluate(program_path: str) -> Dict[str, float]:
    """
    Standard evaluation for Phase 2.
    Cascade is disabled in config, so this is the only function called.
    """
    metrics: Dict[str, float] = {}
    path = Path(program_path)
    _sanitize_candidate_file(path)
    src = _load_source(path)

    if not src: 
        return {"combined_score": 0.0}

    # 1. Syntax Check
    try:
        tree = ast.parse(src)
        metrics["syntax"] = 1.0
    except SyntaxError:
        return {"combined_score": 0.0}

    # 2. Keyword Alignment (Physics Concepts)
    text = src.lower()
    hits = sum(1 for k in KEYWORDS if k in text)
    metrics["alignment"] = min(1.0, hits / 5.0)

    # 3. Structure Analysis
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    has_inheritance = any(c.bases for c in classes)
    metrics["structure"] = 0.5 + (0.5 if has_inheritance else 0.0)

    # 4. Introspection (Runtime Check)
    try:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = module
        spec.loader.exec_module(module)
        
        # Check for Phase 2 Classes
        has_relativity = hasattr(module, "RelativisticForce")
        has_system = hasattr(module, "SystemState")
        
        metrics["runnable"] = 1.0 if (has_relativity and has_system) else 0.5
    except Exception:
        metrics["runnable"] = 0.0

    # Calculate Score
    score = (
        0.3 * metrics["runnable"] +
        0.3 * metrics["alignment"] +
        0.2 * metrics["structure"] +
        0.2 * metrics["syntax"]
    )
    
    metrics["combined_score"] = float(np.clip(score, 0.0, 1.0))
    return metrics

# Legacy hooks (just in case)
def evaluate_stage1(p): return evaluate(p)
def evaluate_stage2(p): return evaluate(p)
