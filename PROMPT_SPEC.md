# Original Build Specification for an AI Coding Agent

Build a complete, GitHub-ready project called **ALM Behavioural Risk Engine — Deposit Behaviour, Loan Prepayment & IRRBB Stress Testing** for a graduate banking risk-modelling portfolio.

The implementation must use fully synthetic data and must never claim to be a production banking model or regulatory submission.

Core requirements:
1. Generate synthetic MYR market-rate history across multiple rate regimes.
2. Segment NMDs into retail transactional, retail non-transactional and wholesale.
3. Estimate deposit beta / pass-through using OLS and an error-correction model.
4. Estimate NMD stability using Kaplan-Meier survival and Cox proportional hazards.
5. Estimate core NMD as a transparent function of stability and pass-through, with configurable BNM Exposure Draft caps.
6. Generate synthetic fixed-rate retail loans and model monthly prepayment with interpretable logistic regression using a time-based holdout.
7. Convert SMM to CPR and apply scenario-specific CPR multipliers.
8. Build behavioural cash flows and apply six MYR IRRBB shock scenarios.
9. Calculate baseline EVE, scenario EVE, ΔEVE, ΔEVE/Tier 1 and simplified 12-month ΔNII.
10. Add validation, sensitivity analysis, UAT documentation, SQLite support, automated tests and a professional Streamlit dashboard.
11. Keep calculations centralised in reusable Python modules, not duplicated inside notebooks/dashboard code.
12. Prefer interpretable statistical methods over unnecessary deep learning.
13. Use fixed random seeds and clear model limitations.
14. Ensure `pytest -q` passes and `streamlit run app.py` is the main application command.

The final repository should contain a README, requirements, app, source modules, tests, configuration, model-development documentation and UAT test cases. If extending later, add Monte Carlo uncertainty analysis and a performing-to-arrears early-warning model only after the ALM core is stable.
