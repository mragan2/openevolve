import os
import textwrap

# --- PATHS ---
BASE_DIR = os.getcwd()
NEW_DIR = os.path.join(BASE_DIR, "examples", "time_force_idea")

TARGET_SEED   = os.path.join(NEW_DIR, "initial_program.py")
TARGET_EVAL   = os.path.join(NEW_DIR, "evaluator.py")
TARGET_CONFIG = os.path.join(NEW_DIR, "config.yaml")
TARGET_BAT    = os.path.join(BASE_DIR, "run_time_force.bat")


# --- 1. SEED PROGRAM: NAIVE "TIME AS FORCE" IDEA ---

SEED_PROGRAM = '''"""
Zabawkowy model: "czas jest siłą".

Cel:
- Mamy BARDZO prosty punkt startowy:
    IDEA: "Czas działa jak siła, która popycha stan układu w przyszłość."
- OpenEvolve ma ewoluować ten kod tak, aby:
    * powstała coraz bardziej złożona architektura (klasy, metody, warstwy),
    * kod był dobrze udokumentowany (docstringi, komentarze),
    * zachowana była centralna idea: "czas jako siła".

To NIE jest model fizyczny – to generator struktur kodu z jednej idei.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Dict


# EVOLVE-BLOCK-START
SIMPLE_IDEA = "Czas jest siłą, która zmienia stan układu."

@dataclass
class SystemState:
    """Minimalny stan układu, który może być modyfikowany przez 'siłę czasu'."""
    value: float = 0.0
    time: float = 0.0


class TimeForce:
    """
    Bardzo prosty seed:
    - F_time = 1.0 (stała "siła czasu"),
    - apply() tylko zwiększa value o dt * F_time.
    OpenEvolve ma prawo całkowicie przepisać tę klasę.
    """

    def __init__(self, strength: float = 1.0) -> None:
        self.strength = float(strength)

    def apply(self, state: SystemState, dt: float) -> SystemState:
        """Zastosuj prostą 'siłę czasu' do stanu."""
        state.time += dt
        state.value += self.strength * dt
        return state


def simulate_step(state: SystemState, dt: float) -> SystemState:
    """
    Najprostsza pętla aktualizacji:
    - tworzy TimeForce,
    - aplikuje ją raz.
    OpenEvolve może przekształcić to w dużo bardziej złożony silnik.
    """
    tf = TimeForce()
    return tf.apply(state, dt)
# EVOLVE-BLOCK-END


def demo() -> Dict[str, Any]:
    """Prosty test ręczny – OpenEvolve może go rozbudować w testy jednostkowe."""
    s = SystemState(value=0.0, time=0.0)
    s = simulate_step(s, 0.1)
    return {"value": s.value, "time": s.time}


if __name__ == "__main__":
    print("Demo:", demo())
'''


# --- 2. EVALUATOR: MIARA "OD PROSTEJ IDEI DO WYRAFINOWANEGO KODU" ---

EVALUATOR = '''"""
Time-Force Idea Evaluator.

Zadanie:
- Mamy prosty seed: "czas jest siłą".
- OpenEvolve ma z czasem budować coraz bardziej WYRAFINOWANY kod, który:
    1) pozostaje tematycznie związany z ideą "czas jako siła",
    2) wykorzystuje coraz bogatszą strukturę (klasy, funkcje, warstwy abstrakcji),
    3) jest dobrze udokumentowany (docstringi, komentarze, type hints),
    4) jest poprawny składniowo i daje się zaimportować.

Evaluator nie sprawdza poprawności fizycznej – tylko jakość i złożoność kodu.
"""

import ast
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np


KEYWORDS_PL = [
    "czas", "siła", "popycha", "ewolucja", "strzałka czasu",
    "przyszłość", "przeszłość", "dynamika"
]
KEYWORDS_EN = [
    "time", "force", "flow", "arrow of time", "evolution", "state"
]


def _sanitize_candidate_file(path: Path) -> None:
    """Usuwa bloki ``` jeśli kandydat został wklejony jako Markdown."""
    try:
        text = path.read_text(encoding="utf-8")
        if "```" in text:
            lines = [l for l in text.splitlines() if not l.strip().startswith("```")]
            path.write_text("\\n".join(lines), encoding="utf-8")
    except Exception:
        pass


def _load_source(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _syntax_score(src: str) -> float:
    """Sprawdza, czy kod parsuje się jako AST. 1.0 jeśli tak, 0.0 jeśli nie."""
    try:
        ast.parse(src)
        return 1.0
    except SyntaxError:
        return 0.0


def _idea_alignment_score(src: str) -> float:
    """Sprawdza, na ile tekst kodu nadal kręci się wokół 'czas jako siła'."""
    text = src.lower()
    hits = 0
    total = len(KEYWORDS_PL) + len(KEYWORDS_EN)
    for w in KEYWORDS_PL + KEYWORDS_EN:
        if w in text:
            hits += 1
    if total == 0:
        return 0.0
    return min(1.0, hits / total)


def _structure_score_from_ast(src: str) -> float:
    """
    Mierzy wyrafinowanie struktury:
    - liczba klas
    - liczba funkcji
    - głębokość zagnieżdżenia bloków
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return 0.0

    class Counter(ast.NodeVisitor):
        def __init__(self) -> None:
            self.n_classes = 0
            self.n_funcs = 0
            self.max_depth = 0

        def generic_visit(self, node, depth=0):
            self.max_depth = max(self.max_depth, depth)
            super().generic_visit(node)

        def visit_ClassDef(self, node):
            self.n_classes += 1
            for child in ast.iter_child_nodes(node):
                self.generic_visit(child, depth=1)

        def visit_FunctionDef(self, node):
            self.n_funcs += 1
            for child in ast.iter_child_nodes(node):
                self.generic_visit(child, depth=1)

    c = Counter()
    c.visit(tree)

    # Normalizacja:
    cls_score = min(1.0, c.n_classes / 3.0)
    fn_score  = min(1.0, c.n_funcs / 6.0)
    depth_score = min(1.0, c.max_depth / 4.0)

    # Średnia ważona – chcemy zarówno klasy, jak i funkcje, i trochę zagnieżdżenia
    return 0.4 * cls_score + 0.4 * fn_score + 0.2 * depth_score


def _documentation_score(src: str) -> float:
    """
    Docstringi i komentarze:
    - liczymy # linii komentarzy vs całość,
    - sprawdzamy, czy są docstringi na poziomie modułu / klas / funkcji.
    """
    lines = src.splitlines()
    if not lines:
        return 0.0

    n_comment = sum(1 for l in lines if l.strip().startswith("#"))
    comment_ratio = n_comment / max(len(lines), 1)

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return 0.0

    has_module_doc = ast.get_docstring(tree) is not None

    n_docstrings = 1 if has_module_doc else 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if ast.get_docstring(node) is not None:
                n_docstrings += 1

    docstring_score = min(1.0, n_docstrings / 5.0)
    comment_score = min(1.0, comment_ratio / 0.2)  # 20% komentarzy = max

    return 0.5 * docstring_score + 0.5 * comment_score


def _introspection_score(module: Any) -> float:
    """
    Lekka zachęta do tworzenia API:
    - liczba publicznych callables,
    - obecność klasy TimeForce (lub podobnej),
    - funkcje zaczynające się od "test_".
    """
    names = [n for n in dir(module) if not n.startswith("_")]
    objs = [getattr(module, n) for n in names]

    n_callables = sum(callable(o) for o in objs)
    n_tests = sum(
        1 for n, o in zip(names, objs)
        if callable(o) and n.startswith("test_")
    )
    has_timeforce = any(
        ("timeforce" in n.lower() or "time_force" in n.lower())
        for n in names
    )

    api_score = min(1.0, n_callables / 8.0)
    test_score = min(1.0, n_tests / 3.0)
    tf_bonus = 0.2 if has_timeforce else 0.0

    return float(0.6 * api_score + 0.3 * test_score + tf_bonus)


def evaluate(program_path: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    path = Path(program_path)
    _sanitize_candidate_file(path)

    src = _load_source(path)
    if not src:
        return {"combined_score": 0.0}

    # 1. Składnia
    syntax = _syntax_score(src)
    if syntax == 0.0:
        return {"combined_score": 0.0}
    metrics["syntax"] = syntax

    # 2. Alignment z ideą
    metrics["idea_alignment"] = _idea_alignment_score(src)

    # 3. Struktura (AST)
    metrics["structure"] = _structure_score_from_ast(src)

    # 4. Dokumentacja
    metrics["documentation"] = _documentation_score(src)

    # 5. Import + introspekcja
    try:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        metrics["introspection"] = _introspection_score(module)
    except Exception:
        metrics["introspection"] = 0.0

    # Łączny wynik: balans między "wyrafinowaniem" a trzymaniem się idei
    score = (
        0.25 * metrics.get("idea_alignment", 0.0) +
        0.25 * metrics.get("structure", 0.0) +
        0.20 * metrics.get("documentation", 0.0) +
        0.20 * metrics.get("introspection", 0.0) +
        0.10 * metrics.get("syntax", 0.0)
    )

    metrics["combined_score"] = float(np.clip(score, 0.0, 1.0))
    metrics["stability"] = 1.0
    return metrics


def evaluate_stage1(p: str) -> Dict[str, float]:
    return evaluate(p)


def evaluate_stage2(p: str) -> Dict[str, float]:
    return evaluate(p)
'''


# --- 3. CONFIG: PO POLSKU, "ROZWIŃ IDEĘ W WYRAFINOWANY KOD" ---

CONFIG = """
# OpenEvolve: "Czas jako siła" – od idei do wyrafinowanego kodu
max_iterations: 80
checkpoint_interval: 10
log_level: "INFO"

llm:
  primary_model: "qwen2.5-coder:7b"
  primary_model_weight: 1.0
  api_base: "http://localhost:11434/v1"
  api_key: "YOUR_API_KEY_HERE"

  temperature: 0.85
  top_p: 0.95
  max_tokens: 8192
  timeout: 600

prompt:
  system_message: |
    Jesteś doświadczonym architektem oprogramowania i teoretykiem,
    który ma jedno zadanie:

      WZIĄĆ BARDZO PROSTĄ IDEĘ:
        "Czas jest siłą, która popycha stan układu w przyszłość"

      i PRZEKSZTAŁCIĆ ją w coraz bardziej wyrafinowany kod.

    W programie kandydującym istnieje blok EVOLVE-BLOCK z:
      - klasą TimeForce (lub podobną),
      - prostym stanem SystemState,
      - prostą funkcją simulate_step.

    Ewoluując kod, kieruj się następującymi zasadami:

    1) ZACHOWAJ IDEĘ:
       - Komentarze, docstringi i nazwy symboli powinny jasno wskazywać,
         że czas jest traktowany jako "siła" lub "napęd" zmiany stanu.
       - Nie musisz być realistyczny fizycznie – to zabawkowy model.

    2) DODAWAJ STRUKTURĘ:
       - Rozbijaj logikę na sensowne klasy i moduły (np. TimeForce, integrator,
         różne strategie działania czasu, itp.).
       - Wprowadzaj dodatkowe warstwy abstrakcji tylko wtedy, gdy mają sens
         (np. interfejs Force, z którego dziedziczy TimeForce).

    3) DOKUMENTUJ:
       - Pisz docstringi na poziomie modułu, klas i funkcji.
       - Dodawaj komentarze, które tłumaczą "filozofię" modelu
         (jak rozumiesz czas jako siłę).

    4) API I TESTY:
       - Dodawaj funkcje pomocnicze i "API" (np. run_simulation, build_model).
       - Możesz tworzyć proste funkcje test_* sprawdzające zachowanie klasy TimeForce.

    5) STABILNOŚĆ:
       - Kod ma się poprawnie parsować i importować.
       - Unikaj niepotrzebnych zależności – wystarczy standardowa biblioteka Pythona.

    Nie wprowadzaj tu stałych fizycznych ani kosmologii (G, c, H0 itd.).
    Myśl o tym projekcie jak o warsztacie: z jednej idei "czas jako siła"
    budujesz coraz lepiej zaprojektowaną bibliotekę Pythona.

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
python openevolve-run.py examples/time_force_idea/initial_program.py examples/time_force_idea/evaluator.py --config examples/time_force_idea/config.yaml
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
    print("✅ Created launcher: run_time_force.bat")

    print("\nGotowe. Uruchom:")
    print("   run_time_force.bat")
    print("aby zacząć ewolucję: od prostej idei 'czas jest siłą' do wyrafinowanej biblioteki Pythona.")


if __name__ == "__main__":
    main()
