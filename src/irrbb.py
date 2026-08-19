from __future__ import annotations

import numpy as np
import pandas as pd

BUCKETS = [
    ("O/N", 0.003), ("1M", 0.042), ("3M", 0.167), ("6M", 0.375), ("9M", 0.625), ("1Y", 0.875),
    ("1.5Y", 1.25), ("2Y", 1.75), ("3Y", 2.5), ("4Y", 3.5), ("5Y", 4.5), ("6Y", 5.5),
    ("7Y", 6.5), ("8Y", 7.5), ("9Y", 8.5), ("10Y", 9.5), ("15Y", 12.5), ("20Y", 17.5), (">20Y", 25.0),
]
MYR_SHOCKS = {
    "Parallel Up": [200] * 19,
    "Parallel Down": [-200] * 19,
    "Steepener": [-114, -110, -101, -85, -68, -52, -29, -2, 33, 71, 100, 123, 140, 154, 165, 173, 189, 199, 202],
    "Flattener": [140, 137, 129, 115, 100, 86, 66, 43, 12, -20, -46, -65, -81, -93, -102, -109, -123, -132, -134],
    "Short Rate Up": [175, 173, 168, 159, 150, 141, 128, 113, 94, 73, 57, 44, 34, 27, 21, 16, 8, 2, 0],
    "Short Rate Down": [-175, -173, -168, -159, -150, -141, -128, -113, -94, -73, -57, -44, -34, -27, -21, -16, -8, -2, 0],
}
CPR_MULTIPLIER = {
    "Parallel Up": 0.8, "Parallel Down": 1.2, "Steepener": 0.8,
    "Flattener": 1.2, "Short Rate Up": 0.8, "Short Rate Down": 1.2,
}


def baseline_curve() -> pd.DataFrame:
    years = np.array([x[1] for x in BUCKETS])
    rates = 0.026 + 0.0075 * (1 - np.exp(-years / 4.5)) + 0.0006 * np.sqrt(years)
    return pd.DataFrame({"bucket": [x[0] for x in BUCKETS], "years": years, "baseline_rate": rates})


def scenario_curves() -> pd.DataFrame:
    base = baseline_curve()
    frames = [base.assign(scenario="Baseline", shock_bps=0, rate=base["baseline_rate"])]
    for name, shocks in MYR_SHOCKS.items():
        frames.append(base.assign(
            scenario=name, shock_bps=shocks,
            rate=np.maximum(base["baseline_rate"] + np.array(shocks) / 10000, 0.0001),
        ))
    return pd.concat(frames, ignore_index=True)


def discount_factor(rate: float | np.ndarray, years: float | np.ndarray) -> float | np.ndarray:
    return np.exp(-np.asarray(rate) * np.asarray(years))


def _nearest_rate(curve: pd.DataFrame, years: float) -> float:
    idx = np.abs(curve["years"].to_numpy() - years).argmin()
    return float(curve.iloc[idx]["rate"])


def build_behavioural_cashflows(balance_sheet: pd.DataFrame, core_df: pd.DataFrame,
                                base_cpr: float, scenario: str = "Baseline") -> pd.DataFrame:
    rows: list[tuple[str, int, float, float]] = []
    cpr = min(1.0, base_cpr * CPR_MULTIPLIER.get(scenario, 1.0))
    core_map = core_df.set_index("segment") if not core_df.empty else None
    for _, r in balance_sheet.iterrows():
        side_sign = 1 if r["side"] == "Asset" else -1
        amount = float(r["amount"])
        product = r["product"]
        maturity = float(r["behavioural_maturity_years"])
        if product == "Fixed-rate retail loans":
            n = max(1, int(np.ceil(maturity)))
            rem = amount
            for yr in range(1, n + 1):
                sched = min(amount / n, rem)
                prep = min(max(rem - sched, 0) * cpr, max(rem - sched, 0))
                cf = min(rem, sched + prep)
                rem -= cf
                rows.append((product, side_sign, min(float(yr), maturity), cf))
            if rem > 1e-6:
                rows.append((product, side_sign, maturity, rem))
        elif "NMD" in product and core_map is not None:
            if "transactional" in product.lower() and "non-transactional" not in product.lower():
                seg = "Retail transactional"
            elif "non-transactional" in product.lower():
                seg = "Retail non-transactional"
            else:
                seg = "Wholesale"
            core_prop = float(core_map.loc[seg, "core_proportion"])
            mat = float(core_map.loc[seg, "behavioural_maturity_years"])
            rows.append((product, side_sign, 0.003, amount * (1 - core_prop)))
            rows.append((product, side_sign, mat, amount * core_prop))
        else:
            rows.append((product, side_sign, max(maturity, 0.003), amount))
    return pd.DataFrame(rows, columns=["product", "sign", "years", "cashflow"])


def cashflow_reconciliation(balance_sheet: pd.DataFrame, core_df: pd.DataFrame, base_cpr: float) -> pd.DataFrame:
    cf = build_behavioural_cashflows(balance_sheet, core_df, base_cpr, "Baseline")
    expected = balance_sheet.set_index("product")["amount"]
    actual = cf.groupby("product")["cashflow"].sum()
    out = pd.DataFrame({"expected_principal": expected, "cashflow_principal": actual}).fillna(0).reset_index()
    out["difference"] = out["cashflow_principal"] - out["expected_principal"]
    return out


def _pv_cashflows(cf: pd.DataFrame, curve: pd.DataFrame) -> float:
    pv = 0.0
    for _, row in cf.iterrows():
        rate = _nearest_rate(curve, float(row["years"]))
        pv += float(row["sign"]) * float(row["cashflow"]) * float(discount_factor(rate, float(row["years"])))
    return pv


def calculate_eve(balance_sheet: pd.DataFrame, core_df: pd.DataFrame,
                  base_cpr: float, tier1_capital: float) -> pd.DataFrame:
    curves = scenario_curves()
    results = []
    for scenario in ["Baseline"] + list(MYR_SHOCKS):
        curve = curves[curves["scenario"] == scenario]
        cf = build_behavioural_cashflows(balance_sheet, core_df, base_cpr, scenario)
        results.append({"scenario": scenario, "EVE": _pv_cashflows(cf, curve)})
    out = pd.DataFrame(results)
    base = float(out.loc[out["scenario"] == "Baseline", "EVE"].iloc[0])
    out["delta_EVE"] = out["EVE"] - base
    out["delta_EVE_pct_tier1"] = out["delta_EVE"] / tier1_capital
    return out


def calculate_nii(balance_sheet: pd.DataFrame, beta_summary: pd.DataFrame, shock_bps: float = 0.0) -> float:
    betas = beta_summary.set_index("segment")["short_run_beta"].clip(0, 1).to_dict()
    shock_pct_points = shock_bps / 100.0
    income = expense = 0.0
    for _, r in balance_sheet.iterrows():
        rate, product, amount = float(r["coupon_rate"]), r["product"], float(r["amount"])
        if r["side"] == "Asset":
            pass_through = 1.0 if "Floating" in product else 0.12
            income += amount * (rate + pass_through * shock_pct_points) / 100
        else:
            if product == "Retail transactional NMD":
                beta = betas.get("Retail transactional", 0.2)
            elif product == "Retail non-transactional NMD":
                beta = betas.get("Retail non-transactional", 0.45)
            elif product == "Wholesale NMD":
                beta = betas.get("Wholesale", 0.75)
            else:
                beta = 0.65
            expense += amount * max(rate + beta * shock_pct_points, 0) / 100
    return income - expense


def duration_convexity_check(balance_sheet: pd.DataFrame, core_df: pd.DataFrame,
                             base_cpr: float, shock_bps: float = 200.0) -> pd.DataFrame:
    """Compare full DCF revaluation with duration and duration+convexity for fixed-rate loans."""
    cf = build_behavioural_cashflows(balance_sheet, core_df, base_cpr, "Baseline")
    cf = cf[cf["product"] == "Fixed-rate retail loans"].copy()
    base = baseline_curve().assign(rate=lambda x: x["baseline_rate"])

    def shifted_price(shift: float) -> float:
        curve = base.copy()
        curve["rate"] = np.maximum(curve["baseline_rate"] + shift, 0.0001)
        return _pv_cashflows(cf, curve)

    p0 = shifted_price(0.0)
    eps = 0.0001
    p_up_small, p_dn_small = shifted_price(eps), shifted_price(-eps)
    modified_duration = -(p_up_small - p_dn_small) / (2 * eps * p0)
    convexity = (p_up_small + p_dn_small - 2 * p0) / (eps ** 2 * p0)
    dy = shock_bps / 10000.0
    full = shifted_price(dy) - p0
    dur_only = p0 * (-modified_duration * dy)
    dur_conv = p0 * (-modified_duration * dy + 0.5 * convexity * dy ** 2)
    return pd.DataFrame([{
        "shock_bps": shock_bps,
        "baseline_price": p0,
        "modified_duration": modified_duration,
        "convexity": convexity,
        "full_revaluation_change": full,
        "duration_only_change": dur_only,
        "duration_convexity_change": dur_conv,
        "duration_convexity_error": dur_conv - full,
    }])
