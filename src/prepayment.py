from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["refinancing_incentive", "seasoning_months", "remaining_term_months", "ltv", "prepayment_penalty_flag"]


def cpr_to_smm(cpr: float | np.ndarray) -> float | np.ndarray:
    cpr = np.asarray(cpr)
    return 1 - np.power(1 - np.clip(cpr, 0, 1), 1 / 12)


def smm_to_cpr(smm: float | np.ndarray) -> float | np.ndarray:
    smm = np.asarray(smm)
    return 1 - np.power(1 - np.clip(smm, 0, 1), 12)


def train_prepayment_model(loans: pd.DataFrame):
    d = loans.sort_values("origination_date").copy()
    unique_dates = pd.Series(d["origination_date"].sort_values().unique())
    cutoff = pd.Timestamp(unique_dates.iloc[max(0, int(len(unique_dates) * 0.70) - 1)])
    train = d[d["origination_date"] <= cutoff].copy()
    test = d[d["origination_date"] > cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("Unable to create non-overlapping out-of-time train/test split")

    pipe = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000))])
    pipe.fit(train[FEATURES], train["prepay_event"])
    p = pipe.predict_proba(test[FEATURES])[:, 1]

    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    logit_p = np.log(clipped / (1 - clipped))
    X = np.column_stack([np.ones(len(logit_p)), logit_p])
    coef, *_ = np.linalg.lstsq(X, test["prepay_event"].to_numpy(), rcond=None)

    metrics = {
        "roc_auc": float(roc_auc_score(test["prepay_event"], p)),
        "pr_auc": float(average_precision_score(test["prepay_event"], p)),
        "brier": float(brier_score_loss(test["prepay_event"], p)),
        "train_n": len(train), "test_n": len(test),
        "train_end": pd.Timestamp(train["origination_date"].max()),
        "test_start": pd.Timestamp(test["origination_date"].min()),
        "calibration_intercept_linear": float(coef[0]),
        "calibration_slope_linear": float(coef[1]),
    }
    std_coef = pipe.named_steps["model"].coef_[0]
    coefs = pd.DataFrame({
        "feature": FEATURES,
        "standardised_coef": std_coef,
        "odds_ratio_per_1sd": np.exp(std_coef),
    })
    test = test.copy()
    test["predicted_monthly_prepay"] = p
    return pipe, metrics, coefs, test


def calibration_table(test: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    d = test.copy()
    d["decile"] = pd.qcut(d["predicted_monthly_prepay"], bins, labels=False, duplicates="drop")
    return d.groupby("decile", as_index=False).agg(
        predicted=("predicted_monthly_prepay", "mean"),
        observed=("prepay_event", "mean"),
        count=("prepay_event", "size"),
    )


def portfolio_cpr(predicted_monthly_prob: pd.Series | np.ndarray) -> float:
    smm = float(np.mean(np.clip(np.asarray(predicted_monthly_prob), 0, 1)))
    return float(smm_to_cpr(smm))


def stressed_cpr(base_cpr: float, multiplier: float) -> float:
    return float(min(1.0, max(0.0, base_cpr * multiplier)))
