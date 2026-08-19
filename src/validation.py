from __future__ import annotations

import numpy as np
import pandas as pd

from .irrbb import cashflow_reconciliation, scenario_curves
from .prepayment import cpr_to_smm, smm_to_cpr


def validation_summary(beta_summary: pd.DataFrame, prepay_metrics: dict, eve: pd.DataFrame,
                       core_df: pd.DataFrame, balance_sheet: pd.DataFrame,
                       portfolio_recon: pd.DataFrame, base_cpr: float) -> pd.DataFrame:
    checks: list[tuple[str, str, str]] = []

    def add(name: str, ok: bool, detail: str, warn_only: bool = False) -> None:
        if ok:
            status = "PASS"
        else:
            status = "WARNING" if warn_only else "FAIL"
        checks.append((name, status, detail))

    add("Portfolio source reconciliation", bool((portfolio_recon["difference"].abs() < 1.0).all()),
        f"Max absolute difference = RM {portfolio_recon['difference'].abs().max():,.2f}")

    assets = float(balance_sheet.loc[balance_sheet.side == "Asset", "amount"].sum())
    liabilities = float(balance_sheet.loc[balance_sheet.side == "Liability", "amount"].sum())
    accounting_diff = float(portfolio_recon.loc[portfolio_recon.check == "Accounting identity", "difference"].iloc[0])
    add("Balance-sheet identity", abs(accounting_diff) < 1.0,
        f"Assets RM {assets/1e9:.2f}bn; liabilities RM {liabilities/1e9:.2f}bn plus Tier 1 reconcile")

    add("Deposit beta plausible range", bool(beta_summary["short_run_beta"].between(0, 1).all()),
        "All estimated short-run betas lie between 0 and 1.")
    b = beta_summary.set_index("segment")["short_run_beta"]
    ordering = b["Retail transactional"] < b["Retail non-transactional"] < b["Wholesale"]
    add("Deposit beta economic ordering", bool(ordering),
        "Retail transactional < retail non-transactional < wholesale.")
    add("Deposit beta statistical signal", bool((beta_summary["p_value"] < 0.10).all()),
        f"Largest beta p-value = {beta_summary['p_value'].max():.4f}", warn_only=True)

    cap_ok = bool((core_df["core_proportion"] <= core_df["core_cap"] + 1e-12).all())
    mat_ok = bool((core_df["behavioural_maturity_years"] <= core_df["maturity_cap_years"] + 1e-12).all())
    add("NMD core caps enforced", cap_ok, "Core proportions stay within configured category caps.")
    add("NMD maturity caps enforced", mat_ok, "Behavioural maturities stay within configured category caps.")

    auc = float(prepay_metrics["roc_auc"])
    add("Prepayment discrimination", auc >= 0.65, f"Out-of-time ROC-AUC = {auc:.3f}", warn_only=True)
    pr_auc = float(prepay_metrics["pr_auc"])
    add("Prepayment PR-AUC available", np.isfinite(pr_auc) and pr_auc > 0, f"Out-of-time PR-AUC = {pr_auc:.3f}")
    brier = float(prepay_metrics["brier"])
    add("Prepayment calibration error", brier < 0.20, f"Brier score = {brier:.3f}", warn_only=True)
    time_split_ok = pd.Timestamp(prepay_metrics["train_end"]) < pd.Timestamp(prepay_metrics["test_start"])
    add("Out-of-time split integrity", bool(time_split_ok),
        f"Train end {prepay_metrics['train_end'].date()} < test start {prepay_metrics['test_start'].date()}")

    roundtrip = float(smm_to_cpr(cpr_to_smm(0.18)))
    add("CPR/SMM round-trip", abs(roundtrip - 0.18) < 1e-10, f"Error = {abs(roundtrip-0.18):.2e}")
    add("Portfolio CPR bounds", 0 <= base_cpr <= 1, f"Baseline CPR = {base_cpr:.2%}")

    curves = scenario_curves()
    add("Six IRRBB scenarios configured", curves["scenario"].nunique() == 7,
        f"Found {curves['scenario'].nunique()-1} shocked scenarios plus baseline.")
    base_delta = float(eve.loc[eve.scenario == "Baseline", "delta_EVE"].iloc[0])
    add("Baseline ΔEVE equals zero", abs(base_delta) < 1e-8, f"Baseline ΔEVE = RM {base_delta:,.2f}")
    add("IRRBB results complete", bool(eve[["EVE", "delta_EVE"]].notna().all().all()), "No missing EVE values.")
    add("IRRBB scenario sensitivity", eve[eve.scenario != "Baseline"]["delta_EVE"].abs().max() > 0,
        "At least one rate shock changes EVE.")

    cf_recon = cashflow_reconciliation(balance_sheet, core_df, base_cpr)
    add("Principal cash-flow reconciliation", bool((cf_recon["difference"].abs() < 1.0).all()),
        f"Max absolute principal difference = RM {cf_recon['difference'].abs().max():,.2f}")

    return pd.DataFrame(checks, columns=["check", "status", "detail"])
