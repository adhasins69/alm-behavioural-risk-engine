from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

FEATURES = ["refinancing_incentive", "seasoning_months", "remaining_term_months", "ltv", "prepayment_penalty_flag"]


def cpr_to_smm(cpr: float | np.ndarray) -> float | np.ndarray:
    cpr = np.asarray(cpr)
    return 1 - np.power(1 - np.clip(cpr, 0, 1), 1 / 12)


def smm_to_cpr(smm: float | np.ndarray) -> float | np.ndarray:
    smm = np.asarray(smm)
    return 1 - np.power(1 - np.clip(smm, 0, 1), 12)


def train_prepayment_model(loans: pd.DataFrame):
    d = loans.sort_values("origination_date").copy()
    split = int(len(d) * 0.70)
    train, test = d.iloc[:split], d.iloc[split:]
    pipe = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000))])
    pipe.fit(train[FEATURES], train["prepay_event"])
    p = pipe.predict_proba(test[FEATURES])[:, 1]
    metrics = {"roc_auc": roc_auc_score(test["prepay_event"], p),
               "brier": brier_score_loss(test["prepay_event"], p),
               "train_n": len(train), "test_n": len(test)}
    coefs = pd.DataFrame({"feature": FEATURES,
                          "standardised_coef": pipe.named_steps["model"].coef_[0]})
    test = test.copy()
    test["predicted_monthly_prepay"] = p
    return pipe, metrics, coefs, test


def portfolio_cpr(predicted_monthly_prob: pd.Series | np.ndarray) -> float:
    smm = float(np.mean(np.clip(np.asarray(predicted_monthly_prob), 0, 1)))
    return float(smm_to_cpr(smm))


def stressed_cpr(base_cpr: float, multiplier: float) -> float:
    return float(min(1.0, max(0.0, base_cpr * multiplier)))
