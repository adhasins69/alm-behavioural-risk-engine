from src.data_generator import generate_market_rates, generate_deposit_timeseries, generate_deposit_accounts, generate_loans, synthetic_balance_sheet
from src.deposit_models import fit_deposit_beta, estimate_core_deposits
from src.prepayment import train_prepayment_model, portfolio_cpr
from src.irrbb import calculate_eve


def test_models_end_to_end():
    market = generate_market_rates(96)
    dep_ts = generate_deposit_timeseries(market)
    accts = generate_deposit_accounts(1200)
    loans = generate_loans(2500)
    beta, _ = fit_deposit_beta(dep_ts)
    assert len(beta) == 3
    core = estimate_core_deposits(dep_ts, beta, accts)
    assert core["core_proportion"].between(0, 0.90).all()
    _, metrics, _, test = train_prepayment_model(loans)
    assert 0.5 <= metrics["roc_auc"] <= 1.0
    cpr = portfolio_cpr(test["predicted_monthly_prepay"])
    eve = calculate_eve(synthetic_balance_sheet(), core, cpr)
    assert len(eve) == 7
    assert eve["delta_EVE"].notna().all()
