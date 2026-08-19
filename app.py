import streamlit as st

st.set_page_config(
    page_title="ALM Behavioural Risk Engine",
    page_icon="📊",
    layout="wide",
)

st.title("ALM Behavioural Risk Engine")
st.caption("Deposit Behaviour • Loan Prepayment • IRRBB Stress Testing")

st.info(
    "This is an educational portfolio prototype using synthetic banking data. "
    "It is not a production banking model or regulatory submission."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Synthetic Deposits", "RM 8.4bn")

with col2:
    st.metric("Synthetic Loans", "RM 10.7bn")

with col3:
    st.metric("Current Build", "Foundation")

st.subheader("What this project will model")
st.markdown(
    """
- Non-maturity deposit rate pass-through (deposit beta)
- Deposit stability and runoff behaviour
- Loan prepayment behaviour
- Behavioural cash flows
- IRRBB stress testing
- EVE and NII sensitivity
- Model validation and monitoring
    """
)

st.success("Foundation is working. Next step: generate the synthetic banking dataset.")
