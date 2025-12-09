"""
Physics Evaluator for Massive Graviton Cosmology (Current Stable Version)
Compatible with existing OpenEvolve configuration and scoring logic.
"""

import importlib.util
import math
import sys
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------
# TARGET CONSTANTS
# ---------------------------------------------------------------------
TARGET_OMEGA_MG = 0.7
TARGET_LAMBDA = 1.1e-52
H0_SI = 2.2e-18
H0_SQ = H0_SI ** 2
M_G_REF = 8.1e-69  # 4.6 Gly Compton wavelength equivalent

# ---------------------------------------------------------------------
# ΛCDM reference functions
# ---------------------------------------------------------------------
OMEGA_M_DEFAULT = 0.3
OMEGA_R_DEFAULT = 9e-5
OMEGA_L_DEFAULT = 1.0 - OMEGA_M_DEFAULT - OMEGA_R_DEFAULT
G_NEWTON = 6.67430e-11


def H_LCDM_array(a: np.ndarray) -> np.ndarray:
    """Standard ΛCDM expansion history."""
    a = np.asarray(a, dtype=float)
    return H0_SI * np.sqrt(OMEGA_R_DEFAULT / a**4 + OMEGA_M_DEFAULT / a**3 + OMEGA_L_DEFAULT)


def get_model_H_func(module):
    """Locate any H(a) implementation in the candidate module."""
    for name in ("H_mg_phenomenological", "prediction", "H_total", "invented_hubble"):
        fn = getattr(module, name, None)
        if callable(fn):
            return lambda a: np.vectorize(fn)(np.asarray(a, dtype=float), M_G_REF)
    raise AttributeError("Candidate missing H(a) implementation.")


# ---------------------------------------------------------------------
# Metric calculators (original tolerances)
# ---------------------------------------------------------------------
def compute_H0_ratio_score(H_LCDM_vals, H_model_vals):
    H0_lcdm = float(H_LCDM_vals[-1])
    H0_model = float(H_model_vals[-1])
    if not np.isfinite(H0_model) or H0_model <= 0:
        return 0.0
    sigma = 0.05
    ratio = H0_model / H0_lcdm
    return float(np.exp(-((ratio - 1.0) / sigma) ** 2))


def compute_rho_q_today_score(rho_q_today, rho_crit_today):
    if not np.isfinite(rho_q_today) or rho_crit_today <= 0:
        return 0.0
    frac = abs(rho_q_today) / rho_crit_today
    target, width = 0.05, 0.03
    return float(np.exp(-((frac - target) / width) ** 2))


def compute_quantum_small_early_score(a_grid, rho_q_vals, rho_crit_vals):
    mask = a_grid < 1e-2
    if not np.any(mask):
        return 0.0
    valid = rho_crit_vals[mask] > 0
    if not np.any(valid):
        return 0.0
    frac = np.abs(rho_q_vals[mask][valid]) / rho_crit_vals[mask][valid]
    scale = 1e-3
    return float(np.mean(np.exp(-(frac / scale) ** 2)))


def compute_monotonic_H_score(a_grid, H_model_vals):
    diffs = np.diff(H_model_vals)
    if not np.all(np.isfinite(diffs)):
        return 0.0
    num_viol = np.count_nonzero(diffs > 0)
    frac_viol = num_viol / len(diffs)
    return float(np.clip(1.0 - frac_viol / 0.1, 0.0, 1.0))


# ---------------------------------------------------------------------
# Helper: sanitize candidate file (strip ```python / ``` / lone 'python'
#         and normalize problematic Unicode to ASCII)
# ---------------------------------------------------------------------
def _sanitize_candidate_file(path: Path):
    """
    Czyści plik kandydata z typowych artefaktów po LLM:
      - linie z ```... (początek/koniec code fence),
      - pojedyncza linia "python" na początku (pozostałość po ```python),
      - zamienia problematyczne znaki Unicode (’,“,”,≈,×,÷,−) na bezpieczne ASCII.

    Modyfikuje plik IN PLACE tylko wtedy, gdy wprowadził realną zmianę.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"[sanitize] Could not read file {path}: {e}")
        return

    lines = text.splitlines()
    cleaned = []
    inside_fence = False

    for line in lines:
        stripped = line.strip()

        # Usuwamy linie z ``` (początek/koniec code fence)
        if stripped.startswith("```"):
            inside_fence = not inside_fence
            continue

        # Usuwamy samotną linię "python" na początku pliku
        if not cleaned and stripped.lower() == "python":
            continue

        cleaned.append(line)

    new_text = "\n".join(cleaned)

    # Normalizacja kilku typowych znaków Unicode, które powodowały błędy składni:
    replacements = {
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "≈": "=",   # zamiast operatora „≈” – zwykłe "="
        "×": "*",
        "÷": "/",
        "−": "-",
    }
    for bad, good in replacements.items():
        if bad in new_text:
            new_text = new_text.replace(bad, good)

    new_text = new_text.strip() + "\n"

    if new_text != text:
        try:
            path.write_text(new_text, encoding="utf-8")
            print(f"[sanitize] Cleaned candidate file {path}")
        except Exception as e:
            print(f"[sanitize] Could not write cleaned file {path}: {e}")


# ---------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------
def evaluate(program_path):
    metrics = {}
    path = Path(program_path)

    # SANITYZACJA pliku kandydata
    _sanitize_candidate_file(path)

    # 1. Load candidate
    try:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = module
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"Import failed: {e}")
        return {"combined_score": 0.0}

    metrics.update({"dark_energy_match": 0.0, "lambda_match": 0.0, "stability": 0.0})

    try:
        H_func = module.H_mg_phenomenological
        lambda_func = module.lambda_eff_from_mg
    except AttributeError:
        return metrics

    # Dark Energy Match
    try:
        val_today = float(H_func(1.0, M_G_REF))
        target_val = TARGET_OMEGA_MG * H0_SQ
        err = abs(val_today - target_val) / (target_val + 1e-30)
        metrics["dark_energy_match"] = 1.0 / (1.0 + err)
    except Exception:
        pass

    # Lambda Match
    try:
        val_lambda = float(lambda_func(M_G_REF))
        if val_lambda > 0:
            log_diff = abs(math.log10(val_lambda) - math.log10(TARGET_LAMBDA))
            metrics["lambda_match"] = 1.0 / (1.0 + log_diff)
    except Exception:
        pass

    # Stability
    try:
        val_past = float(H_func(0.5, M_G_REF))
        ratio = abs(val_past) / H0_SQ
        if ratio < 1e3 and np.isfinite(val_past):
            metrics["stability"] = 1.0
        else:
            metrics["stability"] = 0.1
    except Exception:
        pass

    # Extended cosmological metrics
    try:
        a_grid = np.logspace(-4, 0, 256)
        H_LCDM_vals = H_LCDM_array(a_grid)
        H_model = get_model_H_func(module)
        H_model_vals = H_model(a_grid)
        rho_crit_vals = 3 * H_LCDM_vals**2 / (8 * np.pi * G_NEWTON)
        rho_crit_today = rho_crit_vals[-1]

        rho_q_fn = getattr(module, "rho_quantum", None)
        if rho_q_fn is not None:
            rho_q_vals = np.array(
                [rho_q_fn(float(a), float(Hc), float(M_G_REF)) for a, Hc in zip(a_grid, H_LCDM_vals)]
            )
        else:
            rho_q_vals = np.zeros_like(a_grid)

        rho_q_today = rho_q_vals[-1]

        metrics["H0_ratio_score"] = compute_H0_ratio_score(H_LCDM_vals, H_model_vals)
        metrics["rho_q_today_score"] = compute_rho_q_today_score(rho_q_today, rho_crit_today)
        metrics["quantum_small_early_score"] = compute_quantum_small_early_score(
            a_grid, rho_q_vals, rho_crit_vals
        )
        metrics["monotonic_H_score"] = compute_monotonic_H_score(a_grid, H_model_vals)
    except Exception:
        metrics.update(
            {
                "H0_ratio_score": 0.0,
                "rho_q_today_score": 0.0,
                "quantum_small_early_score": 0.0,
                "monotonic_H_score": 0.0,
            }
        )

    # Combined weighted score (Adjusted weights to prioritize rho_q_today_score)
    metrics["combined_score"] = float(
        np.clip(
            0.18 * metrics.get("dark_energy_match", 0.0)
            + 0.12 * metrics.get("lambda_match", 0.0)
            + 0.10 * metrics.get("stability", 0.0)
            + 0.18 * metrics.get("H0_ratio_score", 0.0)
            + 0.40 * metrics.get("rho_q_today_score", 0.0)
            + 0.00 * metrics.get("quantum_small_early_score", 0.0)
            + 0.02 * metrics.get("monotonic_H_score", 0.0),
            0.0,
            1.0,
        )
    )

    return metrics


# ---------------------------------------------------------------------
# Stage-based evaluation for cascade_evaluation
# ---------------------------------------------------------------------
def evaluate_stage1(program_path):
    """
    First-stage evaluation for cascade_evaluation.

    Szybka ocena:
      - sprawdza, czy program się importuje,
      - czy istnieją funkcje H_mg_phenomenological i lambda_eff_from_mg,
      - liczy tylko dark_energy_match, lambda_match i stability,
      - zwraca metrics z combined_score w [0, 1].

    Dzięki temu słabe kandydaty odrzucasz tanio, a droższa pełna ewaluacja
    (z rho_q, H0_ratio itd.) jest uruchamiana tylko dla lepszych programów.
    """
    metrics = {}
    path = Path(program_path)

    # SANITYZACJA pliku kandydata
    _sanitize_candidate_file(path)

    # 1. Ładowanie modułu kandydata
    try:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = module
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"[stage1] Import failed: {e}")
        metrics.update(
            {
                "dark_energy_match": 0.0,
                "lambda_match": 0.0,
                "stability": 0.0,
                "combined_score": 0.0,
            }
        )
        return metrics

    # 2. Inicjalizacja podstawowych metryk
    metrics.update({"dark_energy_match": 0.0, "lambda_match": 0.0, "stability": 0.0})

    # 3. Pobranie funkcji fizycznych
    try:
        H_func = module.H_mg_phenomenological
        lambda_func = module.lambda_eff_from_mg
    except AttributeError:
        # Brak wymaganych funkcji – od razu zwracamy zero
        metrics["combined_score"] = 0.0
        return metrics

    # 4. Dark Energy Match (jak w evaluate, ale bez reszty metryk)
    try:
        val_today = float(H_func(1.0, M_G_REF))
        target_val = TARGET_OMEGA_MG * H0_SQ
        err = abs(val_today - target_val) / (target_val + 1e-30)
        metrics["dark_energy_match"] = 1.0 / (1.0 + err)
    except Exception as e:
        print(f"[stage1] dark_energy_match failed: {e}")

    # 5. Lambda Match
    try:
        val_lambda = float(lambda_func(M_G_REF))
        if val_lambda > 0:
            log_diff = abs(math.log10(val_lambda) - math.log10(TARGET_LAMBDA))
            metrics["lambda_match"] = 1.0 / (1.0 + log_diff)
    except Exception as e:
        print(f"[stage1] lambda_match failed: {e}")

    # 6. Stability
    try:
        val_past = float(H_func(0.5, M_G_REF))
        ratio = abs(val_past) / H0_SQ
        if ratio < 1e3 and np.isfinite(val_past):
            metrics["stability"] = 1.0
        else:
            metrics["stability"] = 0.1
    except Exception as e:
        print(f"[stage1] stability failed: {e}")

    # 7. Prosty combined_score na bazie trzech metryk (skalowany do [0, 1])
    w_de = 0.5   # dark_energy_match
    w_l  = 0.3   # lambda_match
    w_st = 0.2   # stability

    raw_score = (
        w_de * metrics.get("dark_energy_match", 0.0)
        + w_l * metrics.get("lambda_match", 0.0)
        + w_st * metrics.get("stability", 0.0)
    )

    metrics["combined_score"] = float(np.clip(raw_score, 0.0, 1.0))
    return metrics


def evaluate_stage2(program_path):
    """
    Second-stage evaluation for cascade_evaluation.

    Pełna ewaluacja – deleguje bezpośrednio do funkcji evaluate(...),
    która liczy wszystkie metryki (H0_ratio_score, rho_q_today_score,
    quantum_small_early_score, monotonic_H_score itd.) oraz ich
    pełną, ważoną kombinację w combined_score.
    """
    return evaluate(program_path)
