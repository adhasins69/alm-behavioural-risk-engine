from __future__ import annotations

import pandas as pd
from .prepayment import cpr_to_smm, smm_to_cpr


def validation_summary(beta_summary: pd.DataFrame, prepay_metrics: dict, eve: pd.DataFrame) -> pd.DataFrame:
    checks = []
    beta_ok = beta_summary["short_run_beta"].between(-0.2, 1.2).all()
    checks.append(("Deposit beta plausible range", "PASS" if beta_ok else "WARNING",
                   "Betas should be economically interpretable."))
    auc = prepay_metrics["roc_auc"]
    checks.append(("Prepayment discrimination", "PASS" if auc >= 0.65 else "WARNING",
                   f"Out-of-time ROC-AUC = {auc:.3f}"))
    cpr = 0.18
    roundtrip = float(smm_to_cpr(cpr_to_smm(cpr)))
    checks.append(("CPR/SMM round-trip", "PASS" if abs(roundtrip-cpr) < 1e-10 else "FAIL",
                   f"Error = {abs(roundtrip-cpr):.2e}"))
    shocked = eve[eve["scenario"] != "Baseline"]
    checks.append(("IRRBB scenarios calculated", "PASS" if len(shocked) == 6 else "FAIL",
                   "Six prescribed MYR scenarios expected."))
    return pd.DataFrame(checks, columns=["check", "status", "detail"])
