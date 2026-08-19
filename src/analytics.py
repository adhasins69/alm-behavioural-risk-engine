from __future__ import annotations

import numpy as np
import pandas as pd

from .irrbb import calculate_eve, calculate_nii

GAP_BUCKETS = [
    ("O/N-1M", 1 / 12), ("1-3M", 0.25), ("3-6M", 0.5), ("6-12M", 1.0),
    ("1-2Y", 2.0), ("2-3Y", 3.0), ("3-5Y", 5.0), (">5Y", 99.0),
]


def contractual_core(core_df: pd.DataFrame) -> pd.DataFrame:
    out = core_df.copy()
    out["core_proportion"] = 0.0
    out["core_balance"] = 0.0
    out["behavioural_maturity_years"] = 0.003
    return out


def contractual_vs_behavioural_eve(balance_sheet: pd.DataFrame, core_df: pd.DataFrame,
                                   base_cpr: float, tier1_capital: float) -> pd.DataFrame:
    behavioural = calculate_eve(balance_sheet, core_df, base_cpr, tier1_capital).rename(
        columns={"EVE": "behavioural_EVE", "delta_EVE": "behavioural_delta_EVE"}
    )
    contractual = calculate_eve(balance_sheet, contractual_core(core_df), 0.0, tier1_capital).rename(
        columns={"EVE": "contractual_EVE", "delta_EVE": "contractual_delta_EVE"}
    )
    return behavioural[["scenario", "behavioural_EVE", "behavioural_delta_EVE"]].merge(
        contractual[["scenario", "contractual_EVE", "contractual_delta_EVE"]], on="scenario"
    )


def behavioural_bridge(balance_sheet: pd.DataFrame, core_df: pd.DataFrame, base_cpr: float,
                       tier1_capital: float) -> pd.DataFrame:
    full = calculate_eve(balance_sheet, core_df, base_cpr, tier1_capital)
    worst_scenario = full[full.scenario != "Baseline"].sort_values("delta_EVE").iloc[0]["scenario"]
    contractual = calculate_eve(balance_sheet, contractual_core(core_df), 0.0, tier1_capital)
    nmd_only = calculate_eve(balance_sheet, core_df, 0.0, tier1_capital)

    def de(df: pd.DataFrame) -> float:
        return float(df.loc[df.scenario == worst_scenario, "delta_EVE"].iloc[0])

    vals = [de(contractual), de(nmd_only), de(full)]
    return pd.DataFrame({
        "step": ["Contractual view", "+ NMD behaviour", "+ Loan prepayment"],
        "delta_EVE": vals,
        "incremental_effect": [vals[0], vals[1] - vals[0], vals[2] - vals[1]],
        "scenario": worst_scenario,
    })


def _bucket_label(years: float) -> str:
    for label, upper in GAP_BUCKETS:
        if years <= upper:
            return label
    return ">5Y"


def repricing_gap(balance_sheet: pd.DataFrame, core_df: pd.DataFrame,
                  base_cpr: float, behavioural: bool = True) -> pd.DataFrame:
    rows = []
    core_map = core_df.set_index("segment")
    for _, r in balance_sheet.iterrows():
        amount = float(r["amount"])
        product = r["product"]
        years = float(r["behavioural_maturity_years"])
        if "NMD" in product:
            if "transactional" in product.lower() and "non-transactional" not in product.lower():
                seg = "Retail transactional"
            elif "non-transactional" in product.lower():
                seg = "Retail non-transactional"
            else:
                seg = "Wholesale"
            if behavioural:
                cp = float(core_map.loc[seg, "core_proportion"])
                mat = float(core_map.loc[seg, "behavioural_maturity_years"])
                rows.append((r["side"], amount * (1 - cp), 0.003))
                rows.append((r["side"], amount * cp, mat))
                continue
            years = 0.003
        elif product == "Fixed-rate retail loans" and behavioural:
            years = max(0.5, years * (1 - 0.55 * min(base_cpr, 0.75)))
        rows.append((r["side"], amount, years))

    tmp = pd.DataFrame(rows, columns=["side", "amount", "years"])
    tmp["bucket"] = tmp["years"].map(_bucket_label)
    assets = tmp[tmp.side == "Asset"].groupby("bucket")["amount"].sum()
    liabilities = tmp[tmp.side == "Liability"].groupby("bucket")["amount"].sum()
    order = [x[0] for x in GAP_BUCKETS]
    out = pd.DataFrame({"bucket": order})
    out["rate_sensitive_assets"] = out.bucket.map(assets).fillna(0.0)
    out["rate_sensitive_liabilities"] = out.bucket.map(liabilities).fillna(0.0)
    out["gap"] = out["rate_sensitive_assets"] - out["rate_sensitive_liabilities"]
    out["cumulative_gap"] = out["gap"].cumsum()
    out["view"] = "Behavioural" if behavioural else "Contractual"
    return out


def sensitivity_analysis(balance_sheet: pd.DataFrame, beta_summary: pd.DataFrame,
                         core_df: pd.DataFrame, base_cpr: float, tier1_capital: float) -> pd.DataFrame:
    cases = [
        ("Base", 1.0, 1.0, 1.0),
        ("Deposit beta -25%", 0.75, 1.0, 1.0),
        ("Deposit beta +25%", 1.25, 1.0, 1.0),
        ("Core proportion -20%", 1.0, 0.80, 1.0),
        ("Core proportion +20%", 1.0, 1.20, 1.0),
        ("CPR -25%", 1.0, 1.0, 0.75),
        ("CPR +25%", 1.0, 1.0, 1.25),
    ]
    rows = []
    for name, beta_mult, core_mult, cpr_mult in cases:
        b = beta_summary.copy()
        b["short_run_beta"] = (b["short_run_beta"] * beta_mult).clip(0, 1)
        c = core_df.copy()
        c["core_proportion"] = (c["core_proportion"] * core_mult).clip(0, c["core_cap"])
        c["core_balance"] = c["balance"] * c["core_proportion"]
        cpr = min(1.0, base_cpr * cpr_mult)
        eve = calculate_eve(balance_sheet, c, cpr, tier1_capital)
        worst = float(eve[eve.scenario != "Baseline"]["delta_EVE"].min())
        nii_base = calculate_nii(balance_sheet, b, 0)
        nii_up = calculate_nii(balance_sheet, b, 200)
        rows.append({"case": name, "worst_delta_EVE": worst, "parallel_up_delta_NII": nii_up - nii_base})
    out = pd.DataFrame(rows)
    base_eve = float(out.loc[out.case == "Base", "worst_delta_EVE"].iloc[0])
    base_nii = float(out.loc[out.case == "Base", "parallel_up_delta_NII"].iloc[0])
    out["EVE_impact_vs_base"] = out["worst_delta_EVE"] - base_eve
    out["NII_impact_vs_base"] = out["parallel_up_delta_NII"] - base_nii
    return out


def monitoring_history(beta_summary: pd.DataFrame, core_df: pd.DataFrame,
                       base_cpr: float, prepayment_auc: float, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    quarters = pd.date_range("2024-03-31", periods=10, freq="QE")
    retail_beta = float(beta_summary.set_index("segment").loc["Retail transactional", "short_run_beta"])
    core_pct = float(np.average(core_df["core_proportion"], weights=core_df["balance"]))
    rows = []
    for i, q in enumerate(quarters):
        drift = (i - len(quarters) + 1) * 0.002
        beta = max(0, retail_beta + drift + rng.normal(0, 0.012))
        core = np.clip(core_pct - drift + rng.normal(0, 0.012), 0, 1)
        cpr = np.clip(base_cpr + 0.4 * drift + rng.normal(0, 0.006), 0, 1)
        auc = np.clip(prepayment_auc + rng.normal(0, 0.012), 0.5, 1)
        rows.append({
            "quarter": q, "retail_transactional_beta": beta,
            "weighted_core_proportion": core, "portfolio_CPR": cpr,
            "prepayment_AUC": auc,
        })
    out = pd.DataFrame(rows)
    out["status"] = np.where(out["prepayment_AUC"] < 0.65, "WARNING", "PASS")
    return out
