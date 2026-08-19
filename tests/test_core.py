import numpy as np
from src.prepayment import cpr_to_smm, smm_to_cpr, stressed_cpr
from src.irrbb import discount_factor, scenario_curves


def test_cpr_smm_roundtrip():
    for cpr in [0.0, 0.05, 0.20, 0.75]:
        assert abs(float(smm_to_cpr(cpr_to_smm(cpr))) - cpr) < 1e-10


def test_stressed_cpr_bounds():
    assert stressed_cpr(0.8, 2.0) == 1.0
    assert np.isclose(stressed_cpr(0.2, 0.8), 0.16)


def test_discount_factor():
    assert np.isclose(discount_factor(0.05, 2), np.exp(-0.10))


def test_six_scenarios_plus_baseline():
    curves = scenario_curves()
    assert curves["scenario"].nunique() == 7
    assert len(curves[curves.scenario == "Parallel Up"]) == 19
