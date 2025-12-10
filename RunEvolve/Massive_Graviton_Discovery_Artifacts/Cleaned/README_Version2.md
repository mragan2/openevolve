# Massive Graviton Cosmology Scaffold

A small, self-contained scaffold for exploring a massive graviton contribution to the Hubble parameter, with selectable dark-energy behavior.

## Features
- **Modes**:
  - `constant`: cosmological-constant–like (`a_factor = 1`)
  - `dynamical`: smooth sigmoid interpolation between early/late H₀ plus a small power-law tilt
- Mass-to-λ mapping: `lambda_eff_from_mg`
- Convenience API: `get_phenomenology(a, m_g, mode=..., alpha=...)`
- Vacuum energy helper: `rho_quantum` returns `0.7 * rho_crit`
- Basic tests (`pytest`) for normalization, scaling, and API sanity.

## Installation
This is a lightweight module; just ensure Python ≥3.10 and `pytest` for tests:
```bash
pip install pytest
```

## Usage
Place `mg_cosmology.py` in your path and import:

```python
from mg_cosmology import (
    H_mg_phenomenological,
    get_phenomenology,
    lambda_eff_from_mg,
    DarkEnergyMode,
    M_G_REF,
)

# Constant mode (default)
H2_const = H_mg_phenomenological(a=1.0, m_g=M_G_REF, mode="constant")

# Dynamical mode (sigmoid + small tilt)
H2_dyn = H_mg_phenomenological(a=0.7, m_g=M_G_REF, mode="dynamical")

# Phenomenology bundle (returns H^2 contribution, lambda_eff)
H2, lam = get_phenomenology(1.0, M_G_REF, mode=DarkEnergyMode.CONSTANT)

# Lambda scaling with custom alpha
lam_custom = lambda_eff_from_mg(M_G_REF, alpha=0.3)
```

## Running Tests
```
pytest
```

## Parameters of Interest (dynamical mode)
- `transition_midpoint` (default 0.559)
- `transition_width` (default 0.252)
- `H0_early_km_s_Mpc` (default 67.0)
- `H0_late_km_s_Mpc` (default 73.0)
- `epsilon` power-law tilt (default -0.081)

These are exposed as kwargs in `H_mg_phenomenological(..., mode="dynamical", ...)`.

## Notes
- Guardrails: `a <= 0` is floored to `1e-8`.
- Sigmoid uses a tanh-based stable form to avoid overflow.
- Mass scaling is quadratic: `(m_g / M_G_REF)^2`.