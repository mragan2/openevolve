"""
Merged Black Hole Information + Semiclassical Unification (single EVOLVE block)

Top-level (stable):
  - page_entropy_bits
  - page_curve_bits
  - monte_carlo_entropy
  - analyze_information_recovery

Inside the single EVOLVE block:
  - EvaporationConfig (NEW dataclass for robust configuration)
  - Evaporation model (greybody, bh_mass, bh_entropy_bits, continuous Page curve)
  - Entropic backreaction (kappa from S_gen stationarity, diagonal T_ent)
  - Unified symbolic equation (optional SymPy)
  - Plotting and end-to-end evolve_page_and_unify() hook

Notes:
  * This evolution focuses on robustness (dataclass config) and
    physical extensibility (particle-dependent greybody factors).
  * Plotting is refactored out of the core evolve block for better modularity.
"""

from __future__ import annotations

import math
import os
from typing import List, Dict, Any, Iterable, Optional
from dataclasses import dataclass, field # Import dataclass

import numpy as np
import matplotlib.pyplot as plt

# Optional symbolic layer
try:
    from sympy import symbols, Eq, IndexedBase, pi
    _HAS_SYMPY = True
except Exception:  # pragma: no cover
    _HAS_SYMPY = False

__all__ = [
    "page_entropy_bits",
    "page_curve_bits",
    "monte_carlo_entropy",
    "analyze_information_recovery",
    # EVOLVE exports (kept stable names)
    "EvaporationConfig", # NEW
    "greybody_factor",
    "bh_mass",
    "bh_entropy_bits",
    "continuous_page_curve",
    "analyze_page_curve",
    "calibrate_kappa_via_page_stationarity",
    "entropic_stress_tensor_diagonal",
    "build_unified_symbolic_equation",
    "evolve_page_and_unify",
    # Plotting is now external to the core logic
    "plot_evolution_results",
]

EULER_GAMMA = 0.5772156649015329
ALPHA_HAWKING = 1.0  # dM/dt = -alpha/M^2 in code units
EPS = 1e-12

_env_flag = os.getenv("ENABLE_INTERACTIVE_PLOTS", "0").lower()
ENABLE_INTERACTIVE_PLOTS = _env_flag in ("1", "true", "yes")


# ------------------------------ Stable utilities ------------------------------

def _harmonic_number(n: int) -> float:
    """Nth harmonic number with series/asymptotics."""
    if n <= 0:
        return 0.0
    if n < 100:
        return sum(1.0 / k for k in range(1, n + 1))
    inv_n = 1.0 / n
    return (
        math.log(n)
        + EULER_GAMMA
        + 0.5 * inv_n
        - (inv_n ** 2) / 12.0
        + (inv_n ** 4) / 120.0
    )


def page_entropy_bits(d_a: int, d_b: int) -> float:
    """Exact Page entropy (bits) for a bipartite pure state."""
    if d_a <= 0 or d_b <= 0:
        return 0.0
    if d_a > d_b:
        d_a, d_b = d_b, d_a
    h_ab = _harmonic_number(d_a * d_b)
    h_b = _harmonic_number(d_b)
    s_nats = h_ab - h_b - (d_a - 1) / (2.0 * d_b)
    return s_nats / math.log(2.0)


def page_curve_bits(n_qubits_total: int) -> List[float]:
    """Return the discrete Page curve (entropy after each emitted qubit)."""
    return [
        page_entropy_bits(1 << m, 1 << (n_qubits_total - m))
        for m in range(1, n_qubits_total + 1)
    ]


def monte_carlo_entropy(
    n_qubits_total: int,
    samples: int = 50,
    seed: Optional[int] = None,
) -> List[float]:
    """Monte Carlo estimate of radiation entropy via random pure states.
    Uses Haar-like random vectors; results are averaged over 'samples'.
    """
    if seed is not None:
        np.random.seed(int(seed))
    entropies: List[float] = []
    for m in range(1, n_qubits_total + 1):
        d_r = 1 << m
        d_bh = 1 << (n_qubits_total - m)
        total = 0.0
        for _ in range(samples):
            psi = (np.random.randn(d_r * d_bh) + 1j * np.random.randn(d_r * d_bh)).astype(np.complex128)
            psi /= np.linalg.norm(psi)
            rho = psi.reshape(d_r, d_bh) @ psi.reshape(d_r, d_bh).conj().T
            # Numerical hygiene: clip tiny negatives and renormalize trace
            eigs = np.linalg.eigvalsh(rho)
            eigs = np.real_if_close(eigs).astype(np.float64)
            eigs = np.clip(eigs, 0.0, 1.0)
            s = float(-np.sum(eigs * np.log2(np.clip(eigs + EPS, EPS, 1.0))))
            total += s
        entropies.append(total / max(samples, 1))
    return entropies


def analyze_information_recovery(n_qubits_total: int = 8) -> Dict[str, Any]:
    """Summarize the discrete Page curve and locate Page time."""
    curve = page_curve_bits(n_qubits_total)
    arr = np.asarray(curve, dtype=float)
    k = int(np.argmax(arr))
    return {
        "page_time": k + 1,
        "total_qubits": n_qubits_total,
        "interpretation": "Information becomes recoverable after Page time",
    }


# ============================= EVOLVE REGION =============================
# EVOLVE-BLOCK-START

# ----- Configuration Dataclass -----

@dataclass
class EvaporationConfig:
    """Configuration for the evaporation and unification model."""
    M0: float = 5.0
    alpha: float = ALPHA_HAWKING
    use_greybody: bool = False
    spin: float = 0.0
    use_island: bool = False
    island_offset_bits: float = 0.0
    island_smoothness: float = 1.0
    grid_points: int = 600
    w_S: float = 1.0 / 3.0
    volume_eff: float = 1.0
    return_symbolic: bool = True
    # NEW: Particle type for more detailed greybody factors
    particle_type: str = "scalar" # Options: 'scalar', 'fermion', 'vector'


# ----- Evaporation & continuous Page curve -----

def greybody_factor(
    mass: float,
    spin: float = 0.0,
    particle_type: str = "scalar"
) -> float:
    """More detailed greybody correction based on particle type."""
    spin_term = max(0.2, 1.0 - 0.3 * abs(spin))

    # Placeholder for a more complex model: different particle types
    # have different emission cross-sections.
    if particle_type == "fermion":
        base_factor = 0.35 # Fermions (e.g., neutrinos)
    elif particle_type == "vector":
        base_factor = 0.15 # Vectors (e.g., photons)
    else: # scalar
        base_factor = 0.25 # Default scalar field

    base = base_factor + 0.5 * math.tanh(max(mass, 0.0) / 5.0)
    return float(min(1.0, max(0.0, base))) * spin_term


def bh_mass(
    time_s: float,
    m0: float,
    alpha: float = ALPHA_HAWKING,
    use_greybody: bool = False,
    spin: float = 0.0,
    particle_type: str = "scalar", # Propagate new param
) -> float:
    """Mass evolution: dM/dt = -alpha/M^2 (code units).
    If use_greybody, alpha is scaled by greybody_factor.
    """
    coeff = alpha * (
        greybody_factor(m0, spin, particle_type) if use_greybody else 1.0
    )
    m_cubed = max(m0 ** 3 - 3.0 * coeff * max(time_s, 0.0), 0.0)
    return m_cubed ** (1.0 / 3.0)


def bh_evaporation_time(
    m0: float,
    alpha: float,
    use_greybody: bool,
    spin: float,
    particle_type: str = "scalar" # Propagate new param
) -> float:
    """Total evaporation time, respecting greybody and particle type."""
    alpha_eff = alpha * (
        greybody_factor(m0, spin, particle_type) if use_greybody else 1.0
    )
    return m0 ** 3 / (3.0 * max(alpha_eff, EPS))


def bh_entropy_bits(mass: float, spin: float = 0.0) -> float:
    """Bekenstein-Hawking entropy (bits) with simple spin correction."""
    spin = min(max(spin, 0.0), 0.999)
    area = 8.0 * math.pi * max(mass, 0.0) ** 2 * (1.0 + math.sqrt(1.0 - spin ** 2))
    S_nat = area / 4.0
    return S_nat / math.log(2.0)


def _finite_diff(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Centered finite difference with edge fallbacks; returns dy/dx."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return np.gradient(y, x, edge_order=1)
    dydx = np.empty_like(y)
    dx = x[2:] - x[:-2]
    np.clip(dx, EPS, None, out=dx)
    dydx[1:-1] = (y[2:] - y[:-2]) / dx
    dx0 = max(x[1] - x[0], EPS)
    dxn = max(x[-1] - x[-2], EPS)
    dydx[0] = (y[1] - y[0]) / dx0
    dydx[-1] = (y[-1] - y[-2]) / dxn
    return dydx


def continuous_page_curve(
    times: Iterable[float],
    m0: float,
    alpha: float = ALPHA_HAWKING,
    use_island: bool = False,
    island_offset_bits: float = 0.0,
    island_smoothness: float = 1.0,
    use_greybody: bool = False,
    spin: float = 0.0,
    particle_type: str = "scalar", # Propagate new param
) -> np.ndarray:
    """Construct a continuous Page-like curve by balancing emitted bits vs remaining S_BH."""
    times_arr = np.asarray(list(times), dtype=float)
    total_entropy = bh_entropy_bits(m0, spin)
    result = np.empty_like(times_arr, dtype=float)

    for idx, t_val in enumerate(times_arr):
        mass = bh_mass(
            t_val, m0, alpha,
            use_greybody=use_greybody, spin=spin, particle_type=particle_type
        )
        S_bh = bh_entropy_bits(mass, spin)
        emitted_bits = max(total_entropy - S_bh, 0.0)
        page_bits = min(emitted_bits, S_bh)  # Page bound

        if use_island:
            # Smooth offset emulating QES effects
            delta = (page_bits - S_bh) * island_smoothness
            sigmoid = 1.0 / (1.0 + math.exp(-delta))
            page_bits = S_bh + island_offset_bits * sigmoid

        result[idx] = page_bits

    return result


def analyze_page_curve(entropies: Iterable[float]) -> Dict[str, float]:
    """Diagnostics for a discrete/continuous Page-like curve."""
    ent = np.asarray(list(entropies), dtype=float)
    k = int(np.argmax(ent))
    return {
        "page_time_index": k,
        "page_time_fraction": (k + 1) / len(ent),
        "max_entropy_bits": float(ent[k]),
        "final_entropy_bits": float(ent[-1]),
        "information_returned_bits": float(ent[k] - ent[-1]),
    }

# NOTE: plot_page_curves removed from here. A new plotting function
# will be added to the demo section, which consumes the results
# from evolve_page_and_unify.


# ----- Entropic backreaction & unified equation -----

def calibrate_kappa_via_page_stationarity(
    t: np.ndarray,
    S_rad_bits: np.ndarray,
    m0: float,
    alpha: float = ALPHA_HAWKING,
    spin: float = 0.0,
    use_greybody: bool = False,
    particle_type: str = "scalar", # Propagate new param
    window_frac: float = 0.05,
) -> float:
    """Choose kappa so that S_gen'(t_Page) = S_rad'(t_Page) + S_BH'(t_Page) ~ 0
    in a neighborhood of the Page time. Returns a dimensionless kappa.
    """
    S_bh_bits = np.array(
        [
            bh_entropy_bits(
                bh_mass(
                    tt, m0, alpha,
                    use_greybody=use_greybody, spin=spin, particle_type=particle_type
                ),
                spin
            )
            for tt in t
        ],
        dtype=float,
    )
    dSrad = _finite_diff(t, S_rad_bits)
    dSBH = _finite_diff(t, S_bh_bits)

    k_page = int(np.argmax(S_rad_bits))
    w = max(2, int(window_frac * len(t)))
    sl = slice(max(0, k_page - w), min(len(t), k_page + w + 1))

    num = float(np.sum(dSBH[sl] * dSrad[sl]))
    den = float(np.sum(dSrad[sl] ** 2)) + EPS
    kappa = -num / den
    return kappa


def entropic_stress_tensor_diagonal(
    dSrad_dt_bits: float,
    kappa: float,
    volume_eff: float = 1.0,
    w_S: float = 1.0 / 3.0,
) -> np.ndarray:
    """Diagonal fluid T^mu_nu = diag(rho, -p, -p, -p) from entropy production rate.
    Uses rho_S ∝ kappa * (dS_rad/dt)/V_eff; p_S = w_S * rho_S.
    """
    rho_S = (kappa * dSrad_dt_bits) / max(volume_eff, EPS)
    p_S = w_S * rho_S
    return np.diag([rho_S, -p_S, -p_S, -p_S]).astype(float)


def build_unified_symbolic_equation(include_entropic: bool = True):
    """Return SymPy Eq for:
       G_{mu nu} + Lambda g_{mu nu} = (8 pi G / c^4) (T_m + <T_q> [+ T_ent])
       or None if SymPy is not available.
    """
    if not _HAS_SYMPY:
        return None

    # Constants / indices
    c, G, hbar = symbols("c G ħ", positive=True, real=True)
    Λ = symbols("Λ", real=True)
    μ, ν = symbols("μ ν", integer=True)

    # Tensors
    g = IndexedBase("g")
    Gmn = IndexedBase("G")
    T_m = IndexedBase("T_matter")
    T_q = IndexedBase("T_quantum")
    T_ent = IndexedBase("T_ent")

    rhs = (8 * pi * G / c**4) * (T_m[μ, ν] + T_q[μ, ν] + (T_ent[μ, ν] if include_entropic else 0))
    return Eq(Gmn[μ, ν] + Λ * g[μ, ν], rhs)


def evolve_page_and_unify(config: EvaporationConfig | None = None) -> Dict[str, Any]:
    """End-to-end:
       1) Build continuous S_rad(t) using EvaporationConfig.
       2) Calibrate kappa via S_gen stationarity near Page time.
       3) Produce entropic T^mu_nu samples and (optionally) the unified symbolic Eq.
    """
    # Use dataclass default if no config provided
    cfg = config if config is not None else EvaporationConfig()

    t_end = bh_evaporation_time(
        cfg.M0, cfg.alpha, cfg.use_greybody, cfg.spin, cfg.particle_type
    )
    t = np.linspace(0.0, t_end, cfg.grid_points, dtype=float)

    S_rad = continuous_page_curve(
        t, cfg.M0, alpha=cfg.alpha,
        use_island=cfg.use_island,
        island_offset_bits=cfg.island_offset_bits,
        island_smoothness=cfg.island_smoothness,
        use_greybody=cfg.use_greybody,
        spin=cfg.spin,
        particle_type=cfg.particle_type,
    )

    kappa = calibrate_kappa_via_page_stationarity(
        t, S_rad, cfg.M0, alpha=cfg.alpha, spin=cfg.spin,
        use_greybody=cfg.use_greybody, particle_type=cfg.particle_type
    )
    dSrad_dt = _finite_diff(t, S_rad)

    T_ent_diag = np.array([
        entropic_stress_tensor_diagonal(
            dSrad_dt[i], kappa, volume_eff=cfg.volume_eff, w_S=cfg.w_S
        )
        for i in range(len(t))
    ], dtype=float)
    
    # NEW: Calculate S_bh for the results dictionary to aid plotting
    S_bh = np.array([
        bh_entropy_bits(
            bh_mass(
                tt, cfg.M0, cfg.alpha, 
                use_greybody=cfg.use_greybody, spin=cfg.spin, particle_type=cfg.particle_type
            ),
            cfg.spin
        ) for tt in t
    ], dtype=float)


    result: Dict[str, Any] = {
        "time": t,
        "S_rad_bits": S_rad,
        "S_bh_bits": S_bh, # NEW: Add S_BH for analysis
        "kappa": float(kappa),
        "T_entropic_diag": T_ent_diag,  # shape: (len(t), 4, 4)
        "config": cfg, # NEW: Store the config used for the run
    }

    if cfg.return_symbolic:
        result["unified_equation"] = build_unified_symbolic_equation(include_entropic=True)

    return result

# EVOLVE-BLOCK-END
# =========================== END EVOLVE REGION ===========================


# -------------------------- Demo and Visualization --------------------------

def plot_evolution_results(
    results: Dict[str, Any],
    n_qubits_discrete: int = 12,
    show: bool = False,
    save_path: Optional[str] = None,
):
    """
    Visualize the results from evolve_page_and_unify.
    This replaces the old plot_page_curves by consuming the
    results dictionary directly, separating logic from plotting.
    """
    
    t = results.get("time")
    S_rad = results.get("S_rad_bits")
    S_bh = results.get("S_bh_bits")
    cfg = results.get("config", EvaporationConfig()) # Get config from results

    if t is None or S_rad is None or S_bh is None:
        print("Warning: Results dictionary is missing required data for plotting.")
        return

    t_end = t[-1] if len(t) > 0 else 1.0
    
    # 1. Discrete Page curve for comparison
    disc_qubits = max(1, n_qubits_discrete)
    disc_x = np.arange(1, disc_qubits + 1)
    disc_y = np.array(page_curve_bits(disc_qubits), dtype=float)

    plt.figure(figsize=(12, 7))
    
    # Plot discrete
    plt.plot(
        disc_x, disc_y, 
        marker="o", markersize=4, linestyle=":", 
        label=f"Discrete Page curve ({disc_qubits} qubits)"
    )

    # Rescale continuous time to match discrete x-axis
    t_scaled = (t / max(t_end, EPS)) * disc_qubits
    
    # Plot continuous S_rad (the "real" Page curve)
    plt.plot(t_scaled, S_rad, label=f"S_rad (M0={cfg.M0}, spin={cfg.spin})", color="blue", linewidth=2)
    
    # Plot continuous S_BH (the "Hawking" curve)
    plt.plot(t_scaled, S_bh, label=f"S_BH (Bekenstein-Hawking)", color="red", linestyle="--", linewidth=2)
    
    # Plot S_gen (approx)
    S_gen = S_rad + S_bh
    plt.plot(t_scaled, S_gen, label="S_gen (S_rad + S_BH)", color="green", linestyle="-.", alpha=0.6)

    plt.axvline(disc_qubits / 2, linestyle=":", color="grey", label="Page time ~ N/2")

    plt.xlabel("Radiated qubits or rescaled time")
    plt.ylabel("Entropy (bits)")
    plt.title("Black Hole Evaporation & Generalized Entropy")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if bool(show) and ENABLE_INTERACTIVE_PLOTS:
        plt.show()
    else:
        plt.close()


def _demo():
    # 1) Configure and run the evolution using the new dataclass
    print("[1] Running unified evolution...")
    config = EvaporationConfig(
        M0=5.0,
        alpha=ALPHA_HAWKING,
        use_greybody=True,
        spin=0.5,
        particle_type="fermion", # Use new config field
        grid_points=300,
        return_symbolic=True,
    )
    res = evolve_page_and_unify(config)
    
    print(f"[Unified Coupling] kappa = {res['kappa']:.4e}")
    if _HAS_SYMPY and res.get("unified_equation") is not None:
        print("\n[Unified Equation]")
        print(res["unified_equation"])

    # 2) Plot the results using the new dedicated function
    print("\n[2] Generating plot...")
    plot_evolution_results(res, n_qubits_discrete=12, show=False)
    print("Plot generation complete (non-interactive).")


if __name__ == "__main__":
    _demo()