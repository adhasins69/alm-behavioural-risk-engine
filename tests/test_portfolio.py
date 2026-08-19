import numpy as np
from src.data_generator import (
    build_balance_sheet, generate_deposit_timeseries, generate_loans,
    generate_market_rates, latest_nmd_balances, portfolio_reconciliation,
)


def _portfolio():
    market = generate_market_rates(96)
    deposits = generate_deposit_timeseries(market)
    target = 0.85 * float(latest_nmd_balances(deposits).sum())
    loans = generate_loans(2500, target_outstanding=target)
    bs, tier1 = build_balance_sheet(deposits, loans)
    return deposits, loans, bs, tier1


def test_loan_target_scaling():
    deposits, loans, _, _ = _portfolio()
    target = 0.85 * float(latest_nmd_balances(deposits).sum())
    assert np.isclose(loans["outstanding_balance"].sum(), target, rtol=1e-12)


def test_nmd_balance_sheet_reconciliation():
    deposits, loans, bs, tier1 = _portfolio()
    r = portfolio_reconciliation(deposits, loans, bs, tier1)
    diff = r.loc[r.check == "NMD source → balance sheet", "difference"].iloc[0]
    assert abs(diff) < 1.0


def test_loan_balance_sheet_reconciliation():
    deposits, loans, bs, tier1 = _portfolio()
    r = portfolio_reconciliation(deposits, loans, bs, tier1)
    diff = r.loc[r.check == "Loan source → balance sheet", "difference"].iloc[0]
    assert abs(diff) < 1.0


def test_accounting_identity():
    deposits, loans, bs, tier1 = _portfolio()
    assets = bs.loc[bs.side == "Asset", "amount"].sum()
    liabilities = bs.loc[bs.side == "Liability", "amount"].sum()
    assert abs(assets - liabilities - tier1) < 1.0


def test_cash_residual_positive():
    _, _, bs, _ = _portfolio()
    cash = bs.loc[bs["product"] == "Cash", "amount"].iloc[0]
    assert cash > 0


def test_no_negative_portfolio_amounts():
    _, loans, bs, _ = _portfolio()
    assert (loans["outstanding_balance"] > 0).all()
    assert (bs["amount"] > 0).all()
