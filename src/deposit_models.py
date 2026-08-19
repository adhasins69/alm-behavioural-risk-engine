from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

CORE_CAPS = {
    "Retail transactional": (0.90, 5.0),
    "Retail non-transactional": (0.70, 4.5),
    "Wholesale": (0.50, 4.0),
}


def fit_deposit_beta(ts: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    summaries, models = [], {}
    for seg, g in ts.sort_values("date").groupby("segment"):
        g = g.copy()
        g["d_dep"] = g["deposit_rate"].diff()
        g["d_mkt"] = g["market_rate"].diff()
        fit_df = g.dropna()
        X = sm.add_constant(fit_df[["d_mkt"]])
        model = sm.OLS(fit_df["d_dep"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        ci = model.conf_int().loc["d_mkt"]
        bp_p = float(het_breuschpagan(model.resid, model.model.exog)[1])
        summaries.append({
            "segment": seg,
            "short_run_beta": float(model.params["d_mkt"]),
            "std_error": float(model.bse["d_mkt"]),
            "ci_low": float(ci.iloc[0]),
            "ci_high": float(ci.iloc[1]),
            "p_value": float(model.pvalues["d_mkt"]),
            "r_squared": float(model.rsquared),
            "durbin_watson": float(durbin_watson(model.resid)),
            "breusch_pagan_p": bp_p,
            "n_obs": int(model.nobs),
        })
        models[seg] = model
    return pd.DataFrame(summaries), models


def fit_error_correction(ts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seg, g in ts.sort_values("date").groupby("segment"):
        g = g.copy()
        g["d_dep"] = g["deposit_rate"].diff()
        g["d_mkt"] = g["market_rate"].diff()
        g["d_mkt_l1"] = g["d_mkt"].shift(1)
        g["dep_l1"] = g["deposit_rate"].shift(1)
        g["mkt_l1"] = g["market_rate"].shift(1)
        d = g.dropna()
        X = sm.add_constant(d[["dep_l1", "mkt_l1", "d_mkt", "d_mkt_l1"]])
        m = sm.OLS(d["d_dep"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        speed = float(m.params["dep_l1"])
        long_beta = np.nan if abs(speed) < 1e-8 else float(-m.params["mkt_l1"] / speed)
        rows.append({
            "segment": seg, "speed_of_adjustment": speed, "long_run_beta": long_beta,
            "current_change_beta": float(m.params["d_mkt"]), "r_squared": float(m.rsquared),
            "adjustment_p_value": float(m.pvalues["dep_l1"]),
        })
    return pd.DataFrame(rows)


def kaplan_meier(accounts: pd.DataFrame, segment: str | None = None) -> pd.DataFrame:
    d = accounts if segment is None else accounts[accounts["segment"] == segment]
    times = np.sort(d["duration_months"].unique())
    surv = 1.0
    rows = [{"month": 0, "survival": 1.0}]
    for t in times:
        at_risk = (d["duration_months"] >= t).sum()
        events = ((d["duration_months"] == t) & (d["runoff_event"] == 1)).sum()
        if at_risk > 0:
            surv *= 1 - events / at_risk
        rows.append({"month": int(t), "survival": float(surv)})
    return pd.DataFrame(rows)


def fit_cox_runoff(accounts: pd.DataFrame):
    d = accounts.copy()
    X = pd.get_dummies(
        d[["segment", "salary_credit_flag", "digital_flag", "tenure_months", "spread_to_market"]],
        columns=["segment"], drop_first=True, dtype=float,
    )
    X["tenure_months"] = X["tenure_months"] / 12.0
    model = PHReg(d["duration_months"].astype(float), X.astype(float),
                  status=d["runoff_event"].astype(int), ties="efron")
    result = model.fit(disp=0)
    table = pd.DataFrame({
        "feature": X.columns,
        "coef": result.params,
        "hazard_ratio": np.exp(result.params),
        "p_value": result.pvalues,
    })
    return table, result


def estimate_core_deposits(ts: pd.DataFrame, beta_summary: pd.DataFrame,
                           accounts: pd.DataFrame) -> pd.DataFrame:
    latest = ts.sort_values("date").groupby("segment").tail(1).set_index("segment")
    beta_map = beta_summary.set_index("segment")["short_run_beta"].clip(0, 1).to_dict()
    rows = []
    for seg in CORE_CAPS:
        km = kaplan_meier(accounts, seg)
        eligible = km[km["month"] <= 36]
        stable = float(eligible.iloc[-1]["survival"]) if len(eligible) else 0.5
        passthrough = float(np.clip(beta_map.get(seg, 0.5), 0, 1))
        raw_core = stable * (1 - passthrough)
        cap, max_mat = CORE_CAPS[seg]
        core_prop = min(raw_core, cap)
        maturity = min(0.5 + 5.0 * core_prop, max_mat)
        balance = float(latest.loc[seg, "balance"])
        rows.append({
            "segment": seg, "balance": balance, "stable_proportion": stable,
            "pass_through": passthrough, "core_proportion": core_prop,
            "core_balance": balance * core_prop, "behavioural_maturity_years": maturity,
            "core_cap": cap, "maturity_cap_years": max_mat,
        })
    return pd.DataFrame(rows)
