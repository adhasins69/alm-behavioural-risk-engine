import numpy as np
from src.prepayment import cpr_to_smm, smm_to_cpr, stressed_cpr
from src.irrbb import discount_factor, scenario_curves, baseline_curve


def test_cpr_smm_roundtrip_zero():
    assert float(smm_to_cpr(cpr_to_smm(0.0))) == 0.0


def test_cpr_smm_roundtrip_typical():
    for cpr in [0.05, 0.20, 0.75]:
        assert abs(float(smm_to_cpr(cpr_to_smm(cpr))) - cpr) < 1e-10


def test_stressed_cpr_upper_bound():
    assert stressed_cpr(0.8, 2.0) == 1.0


def test_stressed_cpr_multiplier():
    assert np.isclose(stressed_cpr(0.2, 0.8), 0.16)


def test_discount_factor_formula():
    assert np.isclose(discount_factor(0.05, 2), np.exp(-0.10))


def test_discount_factor_declines_with_time():
    assert discount_factor(0.05, 5) < discount_factor(0.05, 1)


def test_six_scenarios_plus_baseline():
    curves = scenario_curves()
    assert curves["scenario"].nunique() == 7


def test_parallel_up_has_19_buckets():
    curves = scenario_curves()
    assert len(curves[curves.scenario == "Parallel Up"]) == 19


def test_baseline_curve_positive():
    assert (baseline_curve()["baseline_rate"] > 0).all()
