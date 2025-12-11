"""
Arrows-of-Time Evaluator (Pure Toy Model).

Goal:
- Evolve arrow_fields(t) and entropy_from_fields(t) such that:

  1) There are multiple "arrows of time":
     at least 2 fields with strictly monotonic behavior in t
     and with different signs of d/dt (one increasing, one decreasing).
     Optionally 3 monotonic fields → 3 arrows.

  2) Entropy law is invariant under time reversal:
     if we define S(t) = entropy_from_fields(t), then S(|t|)
     must be non-decreasing with |t|. That is:
         |t_2| > |t_1| ⇒ S(|t_2|) >= S(|t_1|)
     So entropy increases in both t > 0 and t < 0 branches.
"""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np


# --- CONFIGURABLE TARGET: 2 OR 3 ARROWS OF TIME ---

TARGET_ARROWS = 2  # set to 3 if you want 3 monotonic arrows enforced


def _sanitize_candidate_file(path: Path) -> None:
    """Strip Markdown fences if a candidate file was pasted with ``` blocks."""
    try:
        text = path.read_text(encoding="utf-8")
        if "```" in text:
            lines = [l for l in text.splitlines() if not l.strip().startswith("```")]
            path.write_text("\n".join(lines), encoding="utf-8")
    except Exception:
        pass


# --- UTILS: NUMERICAL SLOPES AND MONOTONICITY ---

def _finite_slope(values, xs):
    values = np.asarray(values, dtype=float)
    xs = np.asarray(xs, dtype=float)
    dv = np.diff(values)
    dx = np.diff(xs)
    dx = np.where(dx == 0, 1e-50, dx)
    return dv / dx


def _monotonic_sign(values, xs, tol=1e-4):
    """
    Determine if f(xs) is strictly monotonic and return its sign:

    +1 : strictly increasing (df/dx > +tol everywhere)
    -1 : strictly decreasing (df/dx < -tol everywhere)
     0 : not monotonic or too flat/noisy.

    """
    slopes = _finite_slope(values, xs)
    if np.all(slopes > tol):
        return 1
    if np.all(slopes < -tol):
        return -1
    return 0


# --- ENTROPY FROM THE CANDIDATE MODULE ---

def evaluate_entropy_symmetry(entropy_fn):
    """
    Check that S(|t|) is non-decreasing with |t|.

    Implementation:
    - Sample t in [-T, T]; map to u = |t|.
    - Sort by u, check S(u) monotonicity.
    """
    T = 2.0
    n = 81
    t_vals = np.linspace(-T, T, n)
    u_vals = np.abs(t_vals)
    S_vals = np.array([float(entropy_fn(t)) for t in t_vals], dtype=float)

    # Sort by |t|
    idx = np.argsort(u_vals)
    u_sorted = u_vals[idx]
    S_sorted = S_vals[idx]

    # Require S_sorted to be non-decreasing with u
    slopes = _finite_slope(S_sorted, u_sorted)
    # Allow tiny negative slopes (numerical noise), penalize big drops
    # Score = fraction of steps with slope >= -tol
    tol = 1e-4
    good = np.sum(slopes >= -tol)
    total = len(slopes)
    if total == 0:
        return 0.0
    return float(good) / float(total)


def evaluate_arrow_count(arrow_fn):
    """
    Count how many independent monotonic arrows of time we have.

    - Sample t on [-T, T].
    - For each field A, B, C from arrow_fn(t), determine the monotonic sign:
        +1 (increasing), -1 (decreasing), 0 (not monotonic).
    - Count how many non-zero monotonic directions exist and how many
      distinct signs (+ vs -) are present.
    """
    T = 2.0
    n = 81
    t_vals = np.linspace(-T, T, n)

    A_vals = []
    B_vals = []
    C_vals = []
    for t in t_vals:
        A, B, C = arrow_fn(float(t))
        A_vals.append(float(A))
        B_vals.append(float(B))
        C_vals.append(float(C))

    A_sign = _monotonic_sign(A_vals, t_vals)
    B_sign = _monotonic_sign(B_vals, t_vals)
    C_sign = _monotonic_sign(C_vals, t_vals)

    signs = [s for s in [A_sign, B_sign, C_sign] if s != 0]

    if not signs:
        return 0.0

    n_monotonic = len(signs)
    unique_signs = len(set(signs))  # 1 or 2 (we only have +1/-1)

    # We want both + and - to exist for multiple arrows.
    # Base "arrow richness" score:
    richness = n_monotonic / 3.0   # 0..1

    # Diversity bonus if both directions present
    diversity = 1.0 if unique_signs == 2 else 0.5

    raw_arrows = min(n_monotonic, 3)

    # Map to a target (2 or 3 arrows)
    target = float(TARGET_ARROWS)
    arrow_match = 1.0 - abs(raw_arrows - target) / max(target, 1.0)
    arrow_match = max(0.0, arrow_match)

    return float(richness * diversity * arrow_match)


# --- MAIN EVALUATOR HOOK FOR OPENEOLVE ---

def evaluate(program_path: str) -> dict:
    metrics = {}
    path = Path(program_path)
    _sanitize_candidate_file(path)

    # Dynamic import of candidate program
    try:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
    except Exception:
        return {"combined_score": 0.0}

    arrow_fn = getattr(module, "arrow_fields", None)
    entropy_fn = getattr(module, "entropy_from_fields", None)

    if arrow_fn is None or entropy_fn is None:
        return {"combined_score": 0.0}

    # Metric 1: Entropy time-reversal invariance (S(|t|) non-decreasing)
    try:
        entropy_score = evaluate_entropy_symmetry(entropy_fn)
    except Exception:
        entropy_score = 0.0

    # Metric 2: Number and diversity of arrows of time
    try:
        arrow_score = evaluate_arrow_count(arrow_fn)
    except Exception:
        arrow_score = 0.0

    metrics["entropy_symmetry"] = entropy_score
    metrics["arrow_richness"] = arrow_score

    # Combined score: balance entropy and arrow count
    score = 0.6 * entropy_score + 0.4 * arrow_score

    metrics["combined_score"] = float(np.clip(score, 0.0, 1.0))
    metrics["stability"] = 1.0
    return metrics


def evaluate_stage1(p: str) -> dict:
    return evaluate(p)


def evaluate_stage2(p: str) -> dict:
    return evaluate(p)
