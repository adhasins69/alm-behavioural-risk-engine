from pathlib import Path

from src.data_generator import (
    generate_deposit_accounts, generate_deposit_timeseries, generate_loans,
    generate_market_rates, latest_nmd_balances,
)
from src.database import initialise_database

if __name__ == "__main__":
    market = generate_market_rates()
    deposits = generate_deposit_timeseries(market)
    accounts = generate_deposit_accounts()
    target_loans = 0.85 * float(latest_nmd_balances(deposits).sum())
    loans = generate_loans(target_outstanding=target_loans)
    path = Path("database/alm_demo.db")
    initialise_database(path, market, deposits, accounts, loans)
    print(f"Created {path}")
