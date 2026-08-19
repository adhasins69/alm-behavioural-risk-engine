import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics import (
    behavioural_bridge,
    contractual_vs_behavioural_eve,
    monitoring_history,
    repricing_gap,
    sensitivity_analysis,
)
from src.data_generator import (
    AS_OF_DATE,
    build_balance_sheet,
    generate_deposit_accounts,
    generate_deposit_timeseries,
    generate_loans,
    generate_market_rates,
    latest_nmd_balances,
    portfolio_reconciliation,
)
from src.deposit_models import (
    estimate_core_deposits,
    fit_cox_runoff,
    fit_deposit_beta,
    fit_error_correction,
    kaplan_meier,
)
from src.irrbb import (
    calculate_eve,
    calculate_nii,
    cashflow_reconciliation,
    duration_convexity_check,
    scenario_curves,
)
from src.prepayment import calibration_table, portfolio_cpr, train_prepayment_model
from src.validation import validation_summary

st.set_page_config(page_title="ALM Behavioural Risk Engine", page_icon="📈", layout="wide")


@st.cache_data
def build_data():
    market = generate_market_rates()
    dep_ts = generate_deposit_timeseries(market)
    dep_accts = generate_deposit_accounts()
    nmd_total = float(latest_nmd_balances(dep_ts).sum())
    loans = generate_loans(target_outstanding=0.85 * nmd_total)
    balance_sheet, tier1 = build_balance_sheet(dep_ts, loans)
    recon = portfolio_reconciliation(dep_ts, loans, balance_sheet, tier1)
    return market, dep_ts, dep_accts, loans, balance_sheet, tier1, recon


@st.cache_resource
def fit_models(dep_ts, dep_accts, loans):
    beta, _ = fit_deposit_beta(dep_ts)
    ecm = fit_error_correction(dep_ts)
    cox_table, _ = fit_cox_runoff(dep_accts)
    _, prepay_metrics, prepay_coefs, prepay_test = train_prepayment_model(loans)
    return beta, ecm, cox_table, prepay_metrics, prepay_coefs, prepay_test


market, dep_ts, dep_accts, loans, balance_sheet, tier1_capital, portfolio_recon = build_data()
beta, ecm, cox_table, prepay_metrics, prepay_coefs, prepay_test = fit_models(dep_ts, dep_accts, loans)
core = estimate_core_deposits(dep_ts, beta, dep_accts)
base_cpr = portfolio_cpr(prepay_test["predicted_monthly_prepay"])
eve = calculate_eve(balance_sheet, core, base_cpr, tier1_capital)
nii_base = calculate_nii(balance_sheet, beta, 0)
nii_up = calculate_nii(balance_sheet, beta, 200)
validation = validation_summary(beta, prepay_metrics, eve, core, balance_sheet, portfolio_recon, base_cpr)
contractual_compare = contractual_vs_behavioural_eve(balance_sheet, core, base_cpr, tier1_capital)
bridge = behavioural_bridge(balance_sheet, core, base_cpr, tier1_capital)
gap_contractual = repricing_gap(balance_sheet, core, base_cpr, behavioural=False)
gap_behavioural = repricing_gap(balance_sheet, core, base_cpr, behavioural=True)
sensitivity = sensitivity_analysis(balance_sheet, beta, core, base_cpr, tier1_capital)
monitoring = monitoring_history(beta, core, base_cpr, prepay_metrics["roc_auc"])
dc_check = duration_convexity_check(balance_sheet, core, base_cpr)
cf_recon = cashflow_reconciliation(balance_sheet, core, base_cpr)

latest_deposits = float(latest_nmd_balances(dep_ts).sum())
modelled_loans = float(loans["outstanding_balance"].sum())
worst = eve[eve["scenario"] != "Baseline"].sort_values("delta_EVE").iloc[0]
passed = int((validation["status"] == "PASS").sum())
warned = int((validation["status"] == "WARNING").sum())
failed = int((validation["status"] == "FAIL").sum())

with st.sidebar:
    st.header("Model scope")
    st.write(f"**Synthetic as-of date:** {AS_OF_DATE.date()}")
    st.write("**Data:** fully synthetic MYR retail banking portfolio")
    st.write("**Loan / NMD target:** 85%")
    st.write("**Prepayment split:** older originations train / newer originations test")
    st.write("**Regulatory framing:** Basel/BNM-inspired educational prototype")
    st.divider()
    st.caption("No real customer data. Not a production model, regulatory submission, or claim of compliance.")

st.title("ALM Behavioural Risk Engine")
st.caption("Deposit Behaviour • Loan Prepayment • IRRBB Stress Testing")
st.info(
    "Educational portfolio prototype using fully synthetic banking data. "
    "All displayed portfolio balances now reconcile to the same generated source portfolios."
)

cols = st.columns(6)
cols[0].metric("NMD deposits", f"RM {latest_deposits / 1e9:.1f}bn")
cols[1].metric("Fixed-rate loans", f"RM {modelled_loans / 1e9:.1f}bn")
cols[2].metric("Baseline CPR", f"{base_cpr:.1%}")
cols[3].metric("Worst ΔEVE", f"RM {worst['delta_EVE'] / 1e6:,.0f}m")
cols[4].metric("Parallel-up ΔNII", f"RM {(nii_up - nii_base) / 1e6:,.0f}m")
cols[5].metric("Technical controls", f"{passed}/{len(validation)} pass")

st.caption(
    f"Technical controls: {passed} PASS / {warned} WARNING / {failed} FAIL. "
    "These are automated prototype checks, not independent bank model validation."
)

tabs = st.tabs([
    "Executive Summary", "Portfolio Reconciliation", "Deposit Beta", "NMD Stability",
    "Loan Prepayment", "IRRBB / EVE", "NII", "Repricing Gap", "Sensitivity",
    "Monitoring", "Validation", "Methodology",
])

with tabs[0]:
    st.subheader("What this engine does")
    st.markdown("**Synthetic portfolio → behavioural models → behavioural cash flows → six interest-rate shocks → ΔEVE / ΔNII.**")
    st.write(
        "Contractual maturities can misstate banking-book risk. Non-maturity deposits may remain stable, "
        "deposit rates may only partially follow market rates, and borrowers may prepay fixed-rate loans. "
        "This project estimates those behaviours and changes the cash-flow profile before stressing rates."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            px.bar(eve[eve.scenario != "Baseline"], x="scenario", y="delta_EVE",
                   title="ΔEVE across six MYR scenarios", labels={"delta_EVE": "ΔEVE (RM)"}),
            use_container_width=True,
        )
        st.caption(f"Worst modelled scenario: **{worst['scenario']}**, ΔEVE = RM {worst['delta_EVE']/1e6:,.1f}m.")
    with c2:
        st.dataframe(
            core[["segment", "stable_proportion", "pass_through", "core_proportion", "behavioural_maturity_years"]]
            .style.format({
                "stable_proportion": "{:.1%}", "pass_through": "{:.1%}",
                "core_proportion": "{:.1%}", "behavioural_maturity_years": "{:.2f}",
            }), use_container_width=True,
        )
        st.caption("Lower pass-through and higher stability increase the modelled core NMD proportion.")
    st.subheader("Behavioural modelling bridge")
    st.dataframe(
        bridge.style.format({"delta_EVE": "RM {:,.0f}", "incremental_effect": "RM {:,.0f}"}),
        use_container_width=True,
    )
    st.caption("The bridge isolates how NMD behaviour and loan prepayment change stressed EVE relative to a contractual view.")

with tabs[1]:
    st.subheader("Portfolio reconciliation — one source of truth")
    st.write(
        "The exact generated deposit and loan balances feed the ALM balance sheet. "
        "The model no longer uses a separate hard-coded fixed-rate loan amount."
    )
    st.dataframe(
        portfolio_recon.style.format({"source": "RM {:,.0f}", "balance_sheet": "RM {:,.0f}", "difference": "RM {:,.2f}"}),
        use_container_width=True,
    )
    st.subheader("Reconciled synthetic balance sheet")
    st.dataframe(balance_sheet.style.format({"amount": "RM {:,.0f}", "coupon_rate": "{:.2f}%", "behavioural_maturity_years": "{:.2f}"}), use_container_width=True)
    assets = balance_sheet.loc[balance_sheet.side == "Asset", "amount"].sum()
    liabilities = balance_sheet.loc[balance_sheet.side == "Liability", "amount"].sum()
    st.success(f"Assets RM {assets/1e9:.2f}bn = liabilities RM {liabilities/1e9:.2f}bn + synthetic Tier 1 RM {tier1_capital/1e9:.2f}bn.")

with tabs[2]:
    st.subheader("Deposit rate pass-through (deposit beta)")
    st.latex(r"\Delta r_t^D = \alpha + \beta\Delta r_t^M + \epsilon_t")
    st.write("β estimates the short-run fraction of a market-rate change passed through to deposit pricing.")
    st.dataframe(beta.style.format({
        "short_run_beta": "{:.3f}", "std_error": "{:.3f}", "ci_low": "{:.3f}", "ci_high": "{:.3f}",
        "p_value": "{:.4f}", "r_squared": "{:.3f}", "durbin_watson": "{:.2f}", "breusch_pagan_p": "{:.4f}",
    }), use_container_width=True)
    fig = px.line(dep_ts, x="date", y="deposit_rate", color="segment", title="Synthetic deposit rates by segment")
    fig.add_scatter(x=market.date, y=market["3m_rate"], mode="lines", name="3M market rate", line=dict(dash="dash"))
    st.plotly_chart(fig, use_container_width=True)
    b = beta.set_index("segment")["short_run_beta"]
    st.info(
        f"Transactional retail beta {b['Retail transactional']:.2f}; non-transactional retail {b['Retail non-transactional']:.2f}; "
        f"wholesale {b['Wholesale']:.2f}. Wholesale is intentionally more rate-sensitive in the synthetic design."
    )
    st.subheader("Error-correction model")
    st.latex(r"\Delta r_t^D = \alpha + \lambda r_{t-1}^D + \theta r_{t-1}^M + \gamma_0\Delta r_t^M + \gamma_1\Delta r_{t-1}^M + \epsilon_t")
    st.dataframe(ecm.style.format({
        "speed_of_adjustment": "{:.3f}", "long_run_beta": "{:.3f}", "current_change_beta": "{:.3f}",
        "r_squared": "{:.3f}", "adjustment_p_value": "{:.4f}",
    }), use_container_width=True)

with tabs[3]:
    st.subheader("NMD stability / runoff")
    st.latex(r"h_i(t)=h_0(t)\exp(\beta^T X_i)")
    curves = []
    for seg in dep_accts.segment.unique():
        k = kaplan_meier(dep_accts, seg)
        k["segment"] = seg
        curves.append(k)
    km_df = pd.concat(curves)
    st.plotly_chart(px.line(km_df, x="month", y="survival", color="segment", title="Kaplan–Meier deposit survival curves"), use_container_width=True)
    st.write("Hazard ratio > 1 means higher modelled runoff hazard, holding the other included variables constant.")
    st.dataframe(cox_table.style.format({"coef": "{:.3f}", "hazard_ratio": "{:.3f}", "p_value": "{:.4f}"}), use_container_width=True)
    st.subheader("Core NMD estimate")
    st.write("Conceptual approximation: **Core = Stable × (1 − Pass-through)**, then constrained by configured category caps.")
    st.dataframe(core.style.format({
        "balance": "RM {:,.0f}", "stable_proportion": "{:.1%}", "pass_through": "{:.1%}",
        "core_proportion": "{:.1%}", "core_balance": "RM {:,.0f}", "behavioural_maturity_years": "{:.2f}",
        "core_cap": "{:.0%}", "maturity_cap_years": "{:.1f}",
    }), use_container_width=True)

with tabs[4]:
    st.subheader("Loan prepayment model")
    st.latex(r"P(Prepay=1|X)=\frac{1}{1+e^{-X\beta}}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Out-of-time ROC-AUC", f"{prepay_metrics['roc_auc']:.3f}")
    c2.metric("PR-AUC", f"{prepay_metrics['pr_auc']:.3f}")
    c3.metric("Brier score", f"{prepay_metrics['brier']:.3f}")
    st.caption(
        f"Time-based split: training through {prepay_metrics['train_end'].date()}, testing from {prepay_metrics['test_start'].date()}. "
        "The date ranges do not overlap."
    )
    st.dataframe(prepay_coefs.style.format({"standardised_coef": "{:.3f}", "odds_ratio_per_1sd": "{:.3f}"}), use_container_width=True)
    st.latex(r"CPR = 1-(1-SMM)^{12}")
    st.metric("Model-implied baseline portfolio CPR", f"{base_cpr:.2%}")
    cal = calibration_table(prepay_test)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cal.predicted, y=cal.observed, mode="lines+markers", name="Observed"))
    maxp = float(cal.predicted.max())
    fig.add_trace(go.Scatter(x=[0, maxp], y=[0, maxp], mode="lines", name="Perfect calibration"))
    fig.update_layout(title="Prepayment calibration by probability decile", xaxis_title="Predicted", yaxis_title="Observed")
    st.plotly_chart(fig, use_container_width=True)

with tabs[5]:
    st.subheader("IRRBB scenario engine")
    st.write("Six MYR shock paths are used as educational scenario assumptions; the project does not claim regulatory approval or compliance.")
    curves = scenario_curves()
    st.plotly_chart(px.line(curves, x="years", y="rate", color="scenario", title="Baseline and shocked yield curves"), use_container_width=True)
    st.latex(r"DF_i(t)=e^{-R_i(t)t},\quad EVE_i=\sum_k CF_{i,k}DF_i(t_k),\quad \Delta EVE_i=EVE_i-EVE_0")
    st.dataframe(eve.style.format({"EVE": "RM {:,.0f}", "delta_EVE": "RM {:,.0f}", "delta_EVE_pct_tier1": "{:.2%}"}), use_container_width=True)
    st.plotly_chart(px.bar(eve[eve.scenario != "Baseline"], x="scenario", y="delta_EVE", title="Economic value sensitivity"), use_container_width=True)
    st.subheader("Contractual vs behavioural")
    st.dataframe(contractual_compare.style.format({
        "behavioural_EVE": "RM {:,.0f}", "behavioural_delta_EVE": "RM {:,.0f}",
        "contractual_EVE": "RM {:,.0f}", "contractual_delta_EVE": "RM {:,.0f}",
    }), use_container_width=True)
    st.subheader("Duration + convexity sanity check")
    st.latex(r"\frac{\Delta P}{P}\approx-D_{mod}\Delta y+\frac{1}{2}C(\Delta y)^2")
    st.dataframe(dc_check.style.format({
        "baseline_price": "RM {:,.0f}", "modified_duration": "{:.3f}", "convexity": "{:.3f}",
        "full_revaluation_change": "RM {:,.0f}", "duration_only_change": "RM {:,.0f}",
        "duration_convexity_change": "RM {:,.0f}", "duration_convexity_error": "RM {:,.0f}",
    }), use_container_width=True)
    st.caption("This holds behavioural cash flows fixed and checks the fixed-rate loan asset's +200 bp price move against a second-order approximation.")

with tabs[6]:
    st.subheader("12-month simplified NII sensitivity")
    shock = st.slider("Parallel shock (bps)", -300, 300, 200, 25)
    nii_user = calculate_nii(balance_sheet, beta, shock)
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline NII", f"RM {nii_base / 1e6:,.1f}m")
    c2.metric(f"NII @ {shock:+} bps", f"RM {nii_user / 1e6:,.1f}m")
    c3.metric("ΔNII", f"RM {(nii_user - nii_base) / 1e6:,.1f}m")
    st.info("Floating-rate assets reprice more fully, while NMD interest expense moves according to estimated deposit beta.")

with tabs[7]:
    st.subheader("Contractual vs behavioural repricing gap")
    gaps = pd.concat([gap_contractual, gap_behavioural], ignore_index=True)
    st.plotly_chart(px.bar(gaps, x="bucket", y="gap", color="view", barmode="group", title="Periodic repricing gap"), use_container_width=True)
    st.plotly_chart(px.line(gaps, x="bucket", y="cumulative_gap", color="view", markers=True, title="Cumulative repricing gap"), use_container_width=True)
    st.dataframe(gaps.style.format({
        "rate_sensitive_assets": "RM {:,.0f}", "rate_sensitive_liabilities": "RM {:,.0f}",
        "gap": "RM {:,.0f}", "cumulative_gap": "RM {:,.0f}",
    }), use_container_width=True)

with tabs[8]:
    st.subheader("Behavioural assumption sensitivity")
    st.write("Each case changes one major behavioural assumption while holding the others broadly constant.")
    sens_plot = sensitivity[sensitivity.case != "Base"].copy()
    st.plotly_chart(px.bar(sens_plot, x="EVE_impact_vs_base", y="case", orientation="h", title="Tornado-style impact on worst ΔEVE"), use_container_width=True)
    st.dataframe(sensitivity.style.format({
        "worst_delta_EVE": "RM {:,.0f}", "parallel_up_delta_NII": "RM {:,.0f}",
        "EVE_impact_vs_base": "RM {:,.0f}", "NII_impact_vs_base": "RM {:,.0f}",
    }), use_container_width=True)

with tabs[9]:
    st.subheader("Quarterly model monitoring")
    st.write("Synthetic monitoring snapshots illustrate how a risk team could track parameter drift and predictive performance over time.")
    mlong = monitoring.melt(
        id_vars=["quarter", "status"],
        value_vars=["retail_transactional_beta", "weighted_core_proportion", "portfolio_CPR", "prepayment_AUC"],
        var_name="metric", value_name="value",
    )
    st.plotly_chart(px.line(mlong, x="quarter", y="value", color="metric", markers=True, title="Model monitoring history"), use_container_width=True)
    st.dataframe(monitoring, use_container_width=True)

with tabs[10]:
    st.subheader("Automated prototype validation controls")
    st.dataframe(validation, use_container_width=True)
    st.write(f"**PASS:** {passed}  |  **WARNING:** {warned}  |  **FAIL:** {failed}")
    st.caption("These are developer/model-owner controls. A production bank model would also require independent validation and formal governance.")
    with st.expander("Principal cash-flow reconciliation"):
        st.dataframe(cf_recon.style.format({"expected_principal": "RM {:,.0f}", "cashflow_principal": "RM {:,.0f}", "difference": "RM {:,.2f}"}), use_container_width=True)

with tabs[11]:
    st.subheader("Methodology & limitations")
    st.markdown(
        """
**Models implemented**
- OLS/HAC deposit beta with confidence intervals and residual diagnostics.
- Error-correction specification for deposit repricing dynamics.
- Kaplan–Meier NMD survival and Cox proportional-hazards runoff modelling.
- Time-split logistic regression for loan prepayment, with ROC-AUC, PR-AUC, Brier score and calibration.
- Core NMD segmentation/caps and scenario-dependent CPR multipliers.
- Discounted-cash-flow EVE under six MYR shock scenarios.
- Simplified 12-month NII sensitivity using deposit pass-through.
- Contractual-vs-behavioural repricing gap, sensitivity analysis and monitoring.
- Duration/convexity sanity check against full fixed-rate loan revaluation.

**Important limitations**
- All customer, rate and balance-sheet data are synthetic.
- The cash-flow engine is deliberately simplified and is not a bank ALM production system.
- No hedging book, CSRBB, automatic-option valuation, basis-risk decomposition or management actions are modelled.
- The validation controls are developer checks, not independent model validation.
- Regulatory references are educational framing only; this project does not claim compliance.
        """
    )
