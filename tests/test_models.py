from src.data_generator import (
    build_balance_sheet, generate_deposit_accounts, generate_deposit_timeseries,
    generate_loans, generate_market_rates, latest_nmd_balances,
)
from src.deposit_models import estimate_core_deposits, fit_deposit_beta
from src.prepayment import portfolio_cpr, train_prepayment_model
from src.irrbb import calculate_eve, cashflow_reconciliation


def _models():
    market = generate_market_rates(96)
    dep_ts = generate_deposit_timeseries(market)
    accts = generate_deposit_accounts(1200)
    target = 0.85 * float(latest_nmd_balances(dep_ts).sum())
    loans = generate_loans(2500, target_outstanding=target)
    beta, _ = fit_deposit_beta(dep_ts)
    core = estimate_core_deposits(dep_ts, beta, accts)
    _, metrics, _, test = train_prepayment_model(loans)
    bs, tier1 = build_balance_sheet(dep_ts, loans)
    cpr = portfolio_cpr(test["predicted_monthly_prepay"])
    return beta, core, metrics, cpr, bs, tier1


def test_three_deposit_betas():
    beta, *_ = _models()
    assert len(beta) == 3


def test_deposit_betas_in_unit_interval():
    beta, *_ = _models()
    assert beta["short_run_beta"].between(0, 1).all()


def test_beta_ordering_matches_synthetic_design():
    beta, *_ = _models()
    b = beta.set_index("segment")["short_run_beta"]
    assert b["Retail transactional"] < b["Retail non-transactional"] < b["Wholesale"]


def test_core_caps_respected():
    _, core, *_ = _models()
    assert (core["core_proportion"] <= core["core_cap"] + 1e-12).all()


def test_maturity_caps_respected():
    _, core, *_ = _models()
    assert (core["behavioural_maturity_years"] <= core["maturity_cap_years"] + 1e-12).all()


def test_prepayment_auc_is_discriminatory():
    *_, metrics, _, _, _ = _models()
    assert metrics["roc_auc"] >= 0.65


def test_time_split_is_out_of_time():
    *_, metrics, _, _, _ = _models()
    assert metrics["train_end"] < metrics["test_start"]


def test_eve_has_baseline_plus_six_scenarios():
    _, core, _, cpr, bs, tier1 = _models()
    eve = calculate_eve(bs, core, cpr, tier1)
    assert len(eve) == 7


def test_baseline_delta_eve_zero():
    _, core, _, cpr, bs, tier1 = _models()
    eve = calculate_eve(bs, core, cpr, tier1)
    assert abs(eve.loc[eve.scenario == "Baseline", "delta_EVE"].iloc[0]) < 1e-8


def test_cashflows_reconcile_to_principal():
    _, core, _, cpr, bs, _ = _models()
    recon = cashflow_reconciliation(bs, core, cpr)
    assert (recon["difference"].abs() < 1.0).all()
