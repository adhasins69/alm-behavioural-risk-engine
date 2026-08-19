# Model Development Document

## 1. Executive summary
This educational prototype demonstrates how customer behaviour can alter the effective cash-flow profile used in Asset Liability Management and IRRBB analysis. All data are synthetic.

## 2. Scope
The prototype covers non-maturity deposit (NMD) rate pass-through, NMD stability/runoff, core deposit estimation, fixed-rate retail loan prepayment, scenario-dependent cash flows, EVE and simplified NII sensitivity.

## 3. Data
Synthetic monthly interest-rate history, synthetic deposit segment pricing/balances, synthetic deposit-account survival data and synthetic fixed-rate retail loan data are generated with a fixed random seed.

## 4. Deposit beta
Short-run pass-through is estimated using OLS with HAC standard errors on changes in deposit rates versus changes in a synthetic 3M market rate. An error-correction specification is also estimated to distinguish short-run response from a long-run relationship.

## 5. NMD stability
Kaplan-Meier curves estimate survival without a material runoff event. A Cox proportional-hazards model relates runoff hazard to segment, salary-credit flag, digital flag, tenure and spread-to-market.

## 6. Core deposits
A transparent educational approximation is used: `Core = Stable × (1 - Pass-through)`, constrained by the segment caps in the September 2025 BNM IRRBB Exposure Draft.

## 7. Loan prepayment
A time-split logistic model estimates monthly prepayment probability from refinancing incentive, seasoning, remaining term, LTV and prepayment-penalty flag. Monthly probability is aggregated into SMM and converted to CPR.

## 8. IRRBB integration
Behavioural NMD maturities and prepayment-adjusted loan cash flows feed a discounted-cash-flow engine. Six MYR interest-rate shocks from the September 2025 BNM Exposure Draft are applied. EVE is calculated under baseline and shocked curves and reported as ΔEVE.

## 9. NII
A simplified 12-month NII projection assumes floating assets reprice more strongly than fixed-rate assets and NMD pricing follows estimated deposit betas.

## 10. Validation
The repository includes automated CPR/SMM, discount-factor and end-to-end checks, plus dashboard validation indicators.

## 11. Limitations
This is not a production model, regulatory submission, or claim of compliance. Synthetic data have no real predictive validity; the balance sheet and cash-flow mechanics are deliberately simplified; hedging, basis-risk decomposition, automatic options, accounting effects and management actions are not modelled.
