import os
import textwrap

# --- PATHS ---
BASE_DIR = os.getcwd()
NEW_DIR = os.path.join(BASE_DIR, "examples", "arrows_of_time")

TARGET_SEED   = os.path.join(NEW_DIR, "initial_program.py")
TARGET_EVAL   = os.path.join(NEW_DIR, "evaluator.py")
TARGET_CONFIG = os.path.join(NEW_DIR, "config.yaml")
TARGET_BAT    = os.path.join(BASE_DIR, "run_arrows.bat")


# --- 1. SEED PROGRAM: PURE TOY MODEL, NO G, NO COSMOLOGY ---

SEED_PROGRAM = '''"""
Toy multi-arrow-of-time model.

We work with a dimensionless time variable t ∈ ℝ.

- arrow_fields(t) returns 3 scalar "order parameters" A(t), B(t), C(t)
  that can encode 2 or 3 different arrows of time via monotonic behavior.
- entropy_from_fields(t) builds an effective entropy S(t).

The evaluator will:
- reward at least 2 monotonic arrows (one increasing, one decreasing),
- enforce that S(|t|) is non-decreasing (entropy law is invariant under t → -t).
"""

import math
import numpy as np


# EVOLVE-BLOCK-START
def arrow_fields(t: float):
    """
    Return three scalar order parameters A, B, C as functions of t.

    Initial seed:
    - A(t) ~ tanh(t)      : increasing arrow
    - B(t) ~ -tanh(t)     : decreasing arrow
    - C(t) ~ t * tanh(t)  : symmetric "bounce" arrow

    OpenEvolve is free to rewrite this block, but must keep the signature.
    """
    t = float(t)
    A = math.tanh(t)
    B = -math.tanh(t)
    C = t * math.tanh(t)
    return A, B, C


def entropy_from_fields(t: float) -> float:
    """
    Construct an effective entropy S(t) from A, B, C.

    We only need a monotonic functional, not physical units.
    Seed choice:
        S(t) = log(1 + A^2 + B^2 + C^2)

    The evaluator will check that S(|t|) is non-decreasing with |t|.
    """
    A, B, C = arrow_fields(t)
    s2 = A*A + B*B + C*C
    return math.log(1.0 + s2)
# EVOLVE-BLOCK-END


def get_state(t: float):
    """
    Convenience helper for plotting / inspection.
    """
    A, B, C = arrow_fields(t)
    S = entropy_from_fields(t)
    return {"t": float(t), "A": A, "B": B, "C": C, "S": S}
'''


# --- 2. EVALUATOR: COUNTS ARROWS + CHECKS TIME-SYMMETRIC ENTROPY ---

EVALUATOR = '''"""
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
    \"\"\"Strip Markdown fences if a candidate file was pasted with ``` blocks.\"\"\"
    try:
        text = path.read_text(encoding="utf-8")
        if "```" in text:
            lines = [l for l in text.splitlines() if not l.strip().startswith("```")]
            path.write_text("\\n".join(lines), encoding="utf-8")
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
'''


# --- 3. CONFIG: SIMPLE, MODEL-AGNOSTIC, NO COSMOLOGY ---

CONFIG = """
# OpenEvolve: Model strzałek czasu (zabawkowy)
max_iterations: 80
checkpoint_interval: 10
log_level: "INFO"

llm:
  primary_model: "qwen2.5-coder:7b"
  primary_model_weight: 1.0
  api_base: "http://localhost:11434/v1"
  api_key: "YOUR_API_KEY_HERE"

  temperature: 0.8
  top_p: 0.95
  max_tokens: 4096
  timeout: 600

prompt:
  system_message: |
    Jesteś teoretycznym fizykiem projektującym zabawkowy model czasu.

    Kod ma dwie kluczowe funkcje:

    1) arrow_fields(t) -> (A, B, C)
       - t jest bezwymiarowym parametrem czasu (float, może być ujemny i dodatni).
       - A(t), B(t), C(t) to skalarne „parametry porządku”, które mogą być
         monotoniczne w funkcji t.
       - Ewaluator będzie:
           * nagradzał modele, w których przynajmniej DWA z tych pól są
             ściśle monotoniczne w t i mają PRZECIWNE znaki pochodnej d/dt
             (jedno rośnie, drugie maleje),
           * opcjonalnie nagradzał trzecią monotoniczną strzałkę czasu
             (3 niezależne strzałki).

    2) entropy_from_fields(t) -> S(t)
       - Używa A, B, C do zdefiniowania efektywnej entropii S(t).
       - Ewaluator policzy S(|t|) i wymusi:
            jeśli |t2| > |t1|  =>  S(|t2|) >= S(|t1|)
         Czyli entropia musi rosnąć wraz z |t| zarówno dla dodatniego,
         jak i ujemnego czasu. Prawo entropii jest więc niezmiennicze
         względem odwrócenia czasu t -> -t.

    WAŻNE OGRANICZENIA:
    - NIE wprowadzaj stałych fizycznych (G, c, H0 itd.).
      To ma być czysto bezwymiarowy model zabawkowy.
    - Funkcje muszą być gładkie i numerycznie stabilne dla t w zakresie [-2, 2].
    - Unikaj osobliwości, dzielenia przez zero i „dzikich” wykładników.

    Co masz zrobić:
    - Przepisz blok EVOLVE-BLOCK w programie kandydującym tak, aby
      uzyskać bogaty, ale stabilny układ strzałek czasu.
    - Zaprojektuj A(t), B(t), C(t) tak, żeby:
        * były prostymi funkcjami analitycznymi,
        * co najmniej dwa z nich były ściśle monotoniczne z przeciwnymi
          znakami pochodnej d/dt,
        * opcjonalnie trzecie pole opisywało „odbicie”/dodatkową strzałkę.
    - Zdefiniuj entropy_from_fields(t) tak, aby:
        S(t) było gładne, ograniczone od dołu i naturalnie rosło z |t|.

  num_top_programs: 4
  use_template_stochasticity: true

database:
  population_size: 60
  archive_size: 20
  num_islands: 3
  elite_selection_ratio: 0.25
  exploitation_ratio: 0.6

evaluator:
  timeout: 40
  cascade_evaluation: false
  parallel_evaluations: 1
  use_llm_feedback: false

evolution_settings:
  diff_based_evolution: true
  allow_full_rewrites: true
  max_code_length: 8000
"""


# --- 4. LAUNCHER BATCH FILE ---

BAT = r"""@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
python openevolve-run.py examples/arrows_of_time/initial_program.py examples/arrows_of_time/evaluator.py --config examples/arrows_of_time/config.yaml
pause
"""


def main():
    print(f"--- SETUP: {NEW_DIR} ---")
    os.makedirs(NEW_DIR, exist_ok=True)

    with open(TARGET_SEED, "w", encoding="utf-8") as f:
        f.write(SEED_PROGRAM)
    print("✅ Wrote initial_program.py")

    with open(TARGET_EVAL, "w", encoding="utf-8") as f:
        f.write(EVALUATOR)
    print("✅ Wrote evaluator.py")

    with open(TARGET_CONFIG, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(CONFIG).lstrip())
    print("✅ Wrote config.yaml")

    with open(TARGET_BAT, "w", encoding="utf-8") as f:
        f.write(BAT)
    print("✅ Created launcher: run_arrows.bat")

    print("\nReady. Run:")
    print("   run_arrows.bat")
    print("to start evolving arrows of time with entropy conserved under t -> -t.")


if __name__ == "__main__":
    main()
