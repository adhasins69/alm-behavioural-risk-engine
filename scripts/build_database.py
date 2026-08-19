from pathlib import Path
from src.data_generator import generate_market_rates, generate_deposit_timeseries, generate_deposit_accounts, generate_loans
from src.database import initialise_database

if __name__ == "__main__":
    market = generate_market_rates()
    deposits = generate_deposit_timeseries(market)
    accounts = generate_deposit_accounts()
    loans = generate_loans()
    path = Path("database/alm_demo.db")
    initialise_database(path, market, deposits, accounts, loans)
    print(f"Created {path}")
