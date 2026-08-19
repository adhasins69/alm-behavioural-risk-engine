import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_generator import (
    generate_market_rates,
    generate_deposit_timeseries,
    generate_deposit_accounts,
    generate_loans,
    synthetic_balance_sheet,
)
from src.deposit_models import (
    fit_deposit_beta,
    fit_error_correction,
    kaplan_meier,
    fit_cox_runoff,
    estimate_core_deposits,
)
from src.prepayment import train_prepayment_model, portfolio_cpr
from src.irrbb import scenario_curves, calculate_eve, calculate_nii
from src.validation import validation_summary

st.set_page_config(page_title="ALM Behavioural Risk Engine", page_icon="📈", layout="wide")


@st.cache_data
def build_data():
    market = generate_market_rates()
    dep_ts = generate_deposit_timeseries(market)
    dep_accts = generate_deposit_accounts()
    loans = generate_loans()
    balance_sheet = synthetic_balance_sheet()
    return market, dep_ts, dep_accts, loans, balance_sheet


@st.cache_resource
def fit_models(dep_ts, dep_accts, loans):
    beta, _ = fit_deposit_beta(dep_ts)
    ecm = fit_error_correction(dep_ts)
    cox_table, _ = fit_cox_runoff(dep_accts)
    _, prepay_metrics, prepay_coefs, prepay_test = train_prepayment_model(loans)
    return beta, ecm, cox_table, prepay_metrics, prepay_coefs, prepay_test


market, dep_ts, dep_accts, loans, balance_sheet = build_data()
beta, ecm, cox_table, prepay_metrics, prepay_coefs, prepay_test = fit_models(dep_ts, dep_accts, loans)
core = estimate_core_deposits(dep_ts, beta, dep_accts)
base_cpr = portfolio_cpr(prepay_test["predicted_monthly_prepay"])
eve = calculate_eve(balance_sheet, core, base_cpr)
nii_base = calculate_nii(balance_sheet, beta, 0)
nii_up = calculate_nii(balance_sheet, beta, 200)
validation = validation_summary(beta, prepay_metrics, eve)

st.title("ALM Behavioural Risk Engine")
st.caption("Deposit Behaviour • Loan Prepayment • IRRBB Stress Testing")
st.info(
    "Educational portfolio prototype using fully synthetic banking data. "
    "Basel/BNM-inspired; not a production model or regulatory submission."
)

worst = eve[eve["scenario"] != "Baseline"].sort_values("delta_EVE").iloc[0]
latest_deposits = dep_ts.sort_values("date").groupby("segment").tail(1)["balance"].sum()

cols = st.columns(6)
cols[0].metric("Synthetic deposits", f"RM {latest_deposits/1e9:.1f}bn")
cols[1].metric("Synthetic loans", f"RM {loans['outstanding_balance'].sum()/1e9:.1f}bn")
cols[2].metric("Baseline CPR", f"{base_cpr:.1%}")
cols[3].metric("Worst ΔEVE", f"RM {worst['delta_EVE']/1e6:,.0f}m")
cols[4].metric("Parallel-up ΔNII", f"RM {(nii_up-nii_base)/1e6:,.0f}m")
cols[5].metric("Model checks", f"{(validation['status']=='PASS').sum()}/{len(validation)} pass")

tabs = st.tabs([
    "Executive Summary",
    "Deposit Beta",
    "NMD Stability",
    "Loan Prepayment",
    "IRRBB / EVE",
    "NII",
    "Validation",
    "Methodology",
])

with tabs[0]:
    st.subheader("Why behavioural modelling matters")
    st.write(
        "Contractual maturities do not fully describe banking-book risk. Non-maturity deposits can remain "
        "stable for years, deposit rates can reprice only partially, and borrowers can prepay fixed-rate loans. "
        "This engine estimates those behaviours and feeds them into stressed cash flows."
    )
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            eve[eve.scenario != "Baseline"],
            x="scenario",
            y="delta_EVE",
            title="ΔEVE across six MYR scenarios",
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.dataframe(
            core[["segment", "stable_proportion", "pass_through", "core_proportion", "behavioural_maturity_years"]]
            .style.format({
                "stable_proportion": "{:.1%}",
                "pass_through": "{:.1%}",
                "core_proportion": "{:.1%}",
                "behavioural_maturity_years": "{:.2f}",
            }),
            use_container_width=True,
        )

with tabs[1]:
    st.subheader("Deposit rate pass-through (deposit beta)")
    st.latex(r"\Delta r_t^D = \alpha + \beta\Delta r_t^M + \epsilon_t")
    st.write(
        "β measures the short-run proportion of a market-rate movement passed through to deposit pricing "
        "in the synthetic portfolio. HAC standard errors are used."
    )
    st.dataframe(
        beta.style.format({"short_run_beta": "{:.3f}", "p_value": "{:.4f}", "r_squared": "{:.3f}"}),
        use_container_width=True,
    )
    fig = px.line(dep_ts, x="date", y="deposit_rate", color="segment", title="Synthetic deposit rates by segment")
    fig.add_scatter(
        x=market.date,
        y=market["3m_rate"],
        mode="lines",
        name="3M market rate",
        line=dict(dash="dash"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Error-correction model")
    st.latex(
        r"\Delta r_t^D = \alpha + \lambda r_{t-1}^D + \theta r_{t-1}^M + "
        r"\gamma_0\Delta r_t^M + \gamma_1\Delta r_{t-1}^M + \epsilon_t"
    )
    st.dataframe(
        ecm.style.format({
            "speed_of_adjustment": "{:.3f}",
            "long_run_beta": "{:.3f}",
            "current_change_beta": "{:.3f}",
            "r_squared": "{:.3f}",
        }),
        use_container_width=True,
    )

with tabs[2]:
    st.subheader("NMD stability / runoff")
    st.latex(r"h_i(t)=h_0(t)\exp(\beta^T X_i)")
    curves = []
    for seg in dep_accts.segment.unique():
        k = kaplan_meier(dep_accts, seg)
        k["segment"] = seg
        curves.append(k)
    km_df = pd.concat(curves)
    st.plotly_chart(
        px.line(km_df, x="month", y="survival", color="segment", title="Kaplan–Meier deposit survival curves"),
        use_container_width=True,
    )
    st.write(
        "Cox proportional-hazards model: hazard ratios above 1 indicate higher runoff hazard, "
        "holding other included factors constant."
    )
    st.dataframe(cox_table.style.format({"coef": "{:.3f}", "hazard_ratio": "{:.3f}"}), use_container_width=True)
    st.subheader("Core NMD estimate")
    st.write(
        "Conceptual approximation: Core = Stable × (1 − Pass-through), then constrained by the "
        "BNM Exposure Draft caps used here as demonstration parameters."
    )
    st.dataframe(
        core.style.format({
            "balance": "RM {:,.0f}",
            "stable_proportion": "{:.1%}",
            "pass_through": "{:.1%}",
            "core_proportion": "{:.1%}",
            "core_balance": "RM {:,.0f}",
            "behavioural_maturity_years": "{:.2f}",
        }),
        use_container_width=True,
    )

with tabs[3]:
    st.subheader("Loan prepayment model")
    st.latex(r"P(Prepay=1|X)=\frac{1}{1+e^{-X\beta}}")
    c1, c2 = st.columns(2)
    c1.metric("Out-of-time ROC-AUC", f"{prepay_metrics['roc_auc']:.3f}")
    c2.metric("Brier score", f"{prepay_metrics['brier']:.3f}")
    st.dataframe(prepay_coefs.style.format({"standardised_coef": "{:.3f}"}), use_container_width=True)
    st.latex(r"CPR = 1-(1-SMM)^{12}")
    st.metric("Model-implied baseline portfolio CPR", f"{base_cpr:.2%}")
    tmp = prepay_test.copy()
    tmp["decile"] = pd.qcut(tmp["predicted_monthly_prepay"], 10, labels=False, duplicates="drop")
    cal = tmp.groupby("decile", as_index=False).agg(
        predicted=("predicted_monthly_prepay", "mean"),
        observed=("prepay_event", "mean"),
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cal.predicted, y=cal.observed, mode="lines+markers", name="Observed"))
    max_p = float(max(cal.predicted.max(), cal.observed.max()))
    fig.add_trace(go.Scatter(x=[0, max_p], y=[0, max_p], mode="lines", name="Perfect calibration"))
    fig.update_layout(
        title="Prepayment calibration by probability decile",
        xaxis_title="Predicted",
        yaxis_title="Observed",
    )
    st.plotly_chart(fig, use_container_width=True)

with tabs[4]:
    st.subheader("IRRBB scenario engine")
    st.write(
        "The six MYR shock paths are taken from the September 2025 BNM IRRBB Exposure Draft and are used "
        "here only for an educational prototype."
    )
    curves = scenario_curves()
    st.plotly_chart(
        px.line(curves, x="years", y="rate", color="scenario", title="Baseline and shocked yield curves"),
        use_container_width=True,
    )
    st.latex(r"DF_i(t)=e^{-R_i(t)t},\quad EVE_i=\sum_k CF_{i,k}DF_i(t_k),\quad \Delta EVE_i=EVE_i-EVE_0")
    st.dataframe(
        eve.style.format({"EVE": "RM {:,.0f}", "delta_EVE": "RM {:,.0f}", "delta_EVE_pct_tier1": "{:.2%}"}),
        use_container_width=True,
    )
    st.plotly_chart(
        px.bar(eve[eve.scenario != "Baseline"], x="scenario", y="delta_EVE", title="Economic value sensitivity"),
        use_container_width=True,
    )

with tabs[5]:
    st.subheader("12-month simplified NII sensitivity")
    shock = st.slider("Parallel shock (bps)", -300, 300, 200, 25)
    nii_user = calculate_nii(balance_sheet, beta, shock)
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline NII", f"RM {nii_base/1e6:,.1f}m")
    c2.metric(f"NII @ {shock:+} bps", f"RM {nii_user/1e6:,.1f}m")
    c3.metric("ΔNII", f"RM {(nii_user-nii_base)/1e6:,.1f}m")
    st.write(
        "Deposit-rate changes are scaled by estimated segment deposit betas, while floating-rate assets "
        "reprice more fully than fixed-rate assets."
    )

with tabs[6]:
    st.subheader("Model validation & monitoring checks")
    st.dataframe(validation, use_container_width=True)
    st.write(
        "A production framework would add independent validation, richer back-testing, stability monitoring, "
        "data-quality controls, approvals and change governance."
    )

with tabs[7]:
    st.subheader("Methodology & limitations")
    st.markdown(
        """
**Models implemented**
- OLS/HAC deposit beta and an error-correction specification.
- Kaplan–Meier NMD survival and Cox proportional-hazards runoff modelling.
- Time-split logistic regression for loan prepayment, plus SMM/CPR conversion.
- BNM-inspired core NMD segmentation/caps and scenario-dependent CPR multipliers.
- Discounted-cash-flow EVE under six MYR shock scenarios.
- Simplified 12-month NII sensitivity using deposit pass-through.

**Limitations**
- All customer, rate and balance-sheet data are synthetic.
- The cash-flow engine is deliberately simplified and does not represent a bank's full ALM system.
- No hedging book, basis-risk decomposition, automatic option valuation or management actions are modelled.
- The September 2025 BNM document is an Exposure Draft; this project does not claim regulatory compliance.
"""
    )
