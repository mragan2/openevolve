import math

import pytest

from mg_cosmology import (
    DarkEnergyMode,
    H0_SQ_MAG,
    H_mg_phenomenological,
    M_G_REF,
    get_phenomenology,
    lambda_eff_from_mg,
)


def test_constant_mode_normalization():
    H2 = H_mg_phenomenological(1.0, M_G_REF, mode=DarkEnergyMode.CONSTANT)
    assert math.isclose(H2, 0.7 * H0_SQ_MAG, rel_tol=1e-6)


def test_mass_scaling_quadratic():
    H2_ref = H_mg_phenomenological(1.0, M_G_REF)
    H2_2x = H_mg_phenomenological(1.0, 2 * M_G_REF)
    assert math.isclose(H2_2x / H2_ref, 4.0, rel_tol=1e-6)


def test_dynamical_mode_returns_finite():
    H2 = H_mg_phenomenological(0.5, M_G_REF, mode=DarkEnergyMode.DYNAMICAL)
    assert math.isfinite(H2)
    H2 = H_mg_phenomenological(2.0, M_G_REF, mode=DarkEnergyMode.DYNAMICAL)
    assert math.isfinite(H2)


def test_get_phenomenology_tuple():
    H2, lam = get_phenomenology(1.0, M_G_REF, mode="constant")
    assert isinstance(H2, float)
    assert isinstance(lam, float)


def test_lambda_eff_scales_as_mass_squared():
    lam_ref = lambda_eff_from_mg(M_G_REF)
    lam_3x = lambda_eff_from_mg(3 * M_G_REF)
    assert math.isclose(lam_3x / lam_ref, 9.0, rel_tol=1e-6)