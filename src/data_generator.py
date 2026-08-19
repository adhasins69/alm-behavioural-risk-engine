from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42
SEGMENTS = ["Retail transactional", "Retail non-transactional", "Wholesale"]


def generate_market_rates(n_months: int = 120, start: str = "2016-01-31", seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=n_months, freq="ME")
    anchors_x = np.array([0, 24, 48, 66, 78, 96, n_months - 1])
    anchors_y = np.array([3.25, 3.00, 3.00, 1.75, 2.00, 3.00, 2.75])
    policy = np.interp(np.arange(n_months), anchors_x, anchors_y)
    policy += rng.normal(0, 0.035, n_months)
    policy = np.clip(policy, 1.25, 4.25)
    curve = pd.DataFrame({"date": dates, "policy_rate": policy})
    premiums = {"1m_rate": 0.05, "3m_rate": 0.12, "6m_rate": 0.20, "1y_rate": 0.33,
                "2y_rate": 0.48, "3y_rate": 0.60, "5y_rate": 0.78, "10y_rate": 1.02}
    for col, prem in premiums.items():
        curve[col] = np.clip(policy + prem + rng.normal(0, 0.045, n_months), 0.05, None)
    return curve


def generate_deposit_timeseries(market: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    settings = {
        "Retail transactional": {"beta": 0.22, "speed": 0.16, "base": 0.20, "bal": 4.1e9, "vol": 0.010},
        "Retail non-transactional": {"beta": 0.47, "speed": 0.24, "base": 0.45, "bal": 2.8e9, "vol": 0.018},
        "Wholesale": {"beta": 0.78, "speed": 0.36, "base": 0.85, "bal": 1.5e9, "vol": 0.035},
    }
    out = []
    for seg, p in settings.items():
        dep_rate = p["base"] + p["beta"] * market.loc[0, "3m_rate"] * 0.45
        bal = p["bal"]
        for _, row in market.reset_index(drop=True).iterrows():
            target = p["base"] + p["beta"] * row["3m_rate"]
            dep_rate = dep_rate + p["speed"] * (target - dep_rate) + rng.normal(0, 0.025)
            rate_gap = row["3m_rate"] - dep_rate
            growth = 0.0022 - p["vol"] * max(rate_gap - 2.0, 0) / 10 + rng.normal(0, p["vol"] / 3)
            bal = max(bal * (1 + growth), p["bal"] * 0.45)
            out.append({"date": row["date"], "segment": seg, "market_rate": row["3m_rate"],
                        "deposit_rate": max(dep_rate, 0.01), "balance": bal,
                        "spread_to_market": row["3m_rate"] - max(dep_rate, 0.01)})
    return pd.DataFrame(out)


def generate_deposit_accounts(n: int = 3500, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 2)
    segment = rng.choice(SEGMENTS, n, p=[0.50, 0.35, 0.15])
    salary = rng.binomial(1, np.where(segment == "Retail transactional", 0.72, 0.25))
    digital = rng.binomial(1, 0.70, n)
    tenure = np.clip(rng.gamma(3.0, 20.0, n), 2, 180)
    balance = np.exp(rng.normal(np.where(segment == "Wholesale", 15.2, 10.5), 1.0))
    spread = np.clip(rng.normal(np.where(segment == "Wholesale", 0.55, 1.50), 0.45), 0.05, 3.5)
    base_h = np.select([segment == "Retail transactional", segment == "Retail non-transactional", segment == "Wholesale"],
                       [0.012, 0.021, 0.040], default=0.02)
    monthly_h = np.clip(base_h * np.exp(0.28 * (spread - 1.2) - 0.45 * salary - 0.004 * tenure), 0.002, 0.18)
    u = rng.uniform(size=n)
    event_time = np.ceil(np.log(1 - u) / np.log(1 - monthly_h)).astype(int)
    censor = rng.integers(24, 73, n)
    duration = np.minimum(event_time, censor)
    event = (event_time <= censor).astype(int)
    return pd.DataFrame({"account_id": [f"D{i:05d}" for i in range(n)], "segment": segment,
                         "salary_credit_flag": salary, "digital_flag": digital, "tenure_months": tenure,
                         "balance": balance, "spread_to_market": spread,
                         "duration_months": duration, "runoff_event": event})


def generate_loans(n: int = 7000, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 3)
    asof = pd.Timestamp("2026-06-30")
    orig_months_ago = rng.integers(3, 121, n)
    origination_date = asof - pd.to_timedelta(orig_months_ago * 30, unit="D")
    original_term = rng.choice([180, 240, 300, 360], n, p=[0.12, 0.28, 0.30, 0.30])
    seasoning = orig_months_ago
    remaining = np.maximum(original_term - seasoning, 12)
    original_balance = np.exp(rng.normal(12.6, 0.55, n))
    outstanding = original_balance * np.clip(remaining / original_term, 0.08, 0.98)
    contract_rate = np.clip(rng.normal(4.25, 0.65, n), 2.3, 7.5)
    market_refi = np.clip(rng.normal(3.85, 0.35, n), 2.2, 5.5)
    incentive = contract_rate - market_refi
    ltv = np.clip(rng.beta(5, 2, n), 0.25, 0.98)
    penalty = rng.binomial(1, 0.25, n)
    income_band = rng.choice(["Low", "Middle", "Upper-middle", "High"], n, p=[0.18, 0.44, 0.28, 0.10])
    product = rng.choice(["Home Loan", "Home Financing-i"], n, p=[0.62, 0.38])
    logit = -5.20 + 1.05 * incentive + 0.010 * seasoning - 1.35 * penalty - 1.00 * (ltv - 0.7)
    prob = 1 / (1 + np.exp(-logit))
    prepay = rng.binomial(1, np.clip(prob, 0.005, 0.65), n)
    return pd.DataFrame({"loan_id": [f"L{i:05d}" for i in range(n)], "origination_date": origination_date,
                         "original_balance": original_balance, "outstanding_balance": outstanding,
                         "original_term_months": original_term, "remaining_term_months": remaining,
                         "contract_rate": contract_rate, "market_refinancing_rate": market_refi,
                         "refinancing_incentive": incentive, "ltv": ltv, "seasoning_months": seasoning,
                         "prepayment_penalty_flag": penalty, "income_band": income_band,
                         "product_type": product, "prepay_event": prepay})


def synthetic_balance_sheet() -> pd.DataFrame:
    rows = [
        ("Asset", "Fixed-rate retail loans", 10.7e9, 4.45, 4.8),
        ("Asset", "Floating-rate retail loans", 4.0e9, 4.10, 0.25),
        ("Asset", "Securities", 2.7e9, 3.85, 3.2),
        ("Asset", "Cash", 1.1e9, 2.00, 0.01),
        ("Liability", "Retail transactional NMD", 4.1e9, 0.85, 3.7),
        ("Liability", "Retail non-transactional NMD", 2.8e9, 1.55, 2.7),
        ("Liability", "Wholesale NMD", 1.5e9, 2.60, 1.8),
        ("Liability", "Term deposits", 4.8e9, 3.05, 1.0),
        ("Liability", "Wholesale funding", 2.4e9, 3.35, 1.6),
    ]
    return pd.DataFrame(rows, columns=["side", "product", "amount", "coupon_rate", "behavioural_maturity_years"])
