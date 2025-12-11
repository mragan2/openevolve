"""
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
