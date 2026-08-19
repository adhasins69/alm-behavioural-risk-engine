from __future__ import annotations

import sqlite3
from pathlib import Path
import pandas as pd


def initialise_database(path: str | Path, market: pd.DataFrame, deposits_ts: pd.DataFrame,
                        deposit_accounts: pd.DataFrame, loans: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as con:
        market.to_sql("market_rates", con, if_exists="replace", index=False)
        deposits_ts.to_sql("deposit_monthly", con, if_exists="replace", index=False)
        deposit_accounts.to_sql("deposit_accounts", con, if_exists="replace", index=False)
        loans.to_sql("loans", con, if_exists="replace", index=False)
        con.execute("CREATE INDEX IF NOT EXISTS idx_dep_segment ON deposit_monthly(segment)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_loan_product ON loans(product_type)")


def query(path: str | Path, sql: str) -> pd.DataFrame:
    with sqlite3.connect(path) as con:
        return pd.read_sql_query(sql, con)
