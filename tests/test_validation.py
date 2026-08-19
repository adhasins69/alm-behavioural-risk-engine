from src.data_generator import (
    build_balance_sheet, generate_deposit_accounts, generate_deposit_timeseries,
    generate_loans, generate_market_rates, latest_nmd_balances, portfolio_reconciliation,
)
from src.deposit_models import estimate_core_deposits, fit_deposit_beta
from src.prepayment import portfolio_cpr, train_prepayment_model
from src.irrbb import calculate_eve, duration_convexity_check
from src.validation import validation_summary


def _full():
    market = generate_market_rates(96)
    dep_ts = generate_deposit_timeseries(market)
    accts = generate_deposit_accounts(1200)
    loans = generate_loans(2500, target_outstanding=0.85 * float(latest_nmd_balances(dep_ts).sum()))
    bs, tier1 = build_balance_sheet(dep_ts, loans)
    beta, _ = fit_deposit_beta(dep_ts)
    core = estimate_core_deposits(dep_ts, beta, accts)
    _, metrics, _, test = train_prepayment_model(loans)
    cpr = portfolio_cpr(test["predicted_monthly_prepay"])
    eve = calculate_eve(bs, core, cpr, tier1)
    recon = portfolio_reconciliation(dep_ts, loans, bs, tier1)
    val = validation_summary(beta, metrics, eve, core, bs, recon, cpr)
    return bs, core, cpr, val


def test_validation_has_many_controls():
    *_, val = _full()
    assert len(val) >= 15


def test_no_validation_failures():
    *_, val = _full()
    assert not (val["status"] == "FAIL").any()


def test_duration_convexity_output_finite():
    bs, core, cpr, _ = _full()
    out = duration_convexity_check(bs, core, cpr)
    assert out.notna().all().all()


def test_duration_convexity_closer_than_duration_only():
    bs, core, cpr, _ = _full()
    out = duration_convexity_check(bs, core, cpr).iloc[0]
    err_d = abs(out["duration_only_change"] - out["full_revaluation_change"])
    err_dc = abs(out["duration_convexity_change"] - out["full_revaluation_change"])
    assert err_dc <= err_d
