# Model Development Document

## 1. Executive summary

This educational prototype demonstrates how customer behaviour can alter the effective cash-flow profile used in Asset Liability Management (ALM) and Interest Rate Risk in the Banking Book (IRRBB) analysis. All data are synthetic and have no real-world predictive validity.

## 2. Business purpose

The purpose is to demonstrate a complete model-development chain relevant to banking risk modelling: data generation, behavioural model estimation, behavioural cash-flow transformation, IRRBB/NII measurement, sensitivity analysis, validation controls, monitoring and UAT-style testing.

## 3. Scope

The prototype covers NMD deposit rate pass-through, NMD stability/runoff, core-deposit estimation and behavioural maturity, fixed-rate retail loan prepayment, scenario-dependent behavioural cash flows, EVE/ΔEVE, simplified 12-month NII sensitivity, contractual-vs-behavioural repricing gap, sensitivity analysis, monitoring and automated technical controls.

## 4. Portfolio construction and reconciliation

The model uses a single reconciled data flow. Current NMD balances are generated first. The fixed-rate loan portfolio is then scaled to 85% of the generated NMD balance. The exact generated NMD and loan balances feed the ALM balance sheet. Supplementary synthetic floating-rate loans, securities, term deposits and wholesale funding are parameterised as proportions of NMD balances, while cash is calculated as the residual required to satisfy:

`Assets = Liabilities + Synthetic Tier 1 Capital`

The dashboard reports source-to-balance-sheet reconciliation differences and requires numerical differences to be effectively zero.

## 5. Data

Synthetic monthly interest-rate history, deposit segment pricing/balances, account-level deposit survival data and fixed-rate retail loan data are generated with fixed random seeds. No real customer or bank-confidential data are used.

## 6. Deposit beta

Short-run pass-through is estimated using OLS with HAC robust covariance on changes in deposit rates versus changes in a synthetic 3M market rate:

`ΔDepositRate_t = α + β ΔMarketRate_t + ε_t`

Reported diagnostics include coefficient standard error, 95% confidence interval, p-value, R², Durbin-Watson statistic and Breusch-Pagan p-value.

## 7. Error-correction specification

An error-correction model separates immediate market-rate response from the longer-run relationship:

`ΔrD_t = α + λ rD_(t-1) + θ rM_(t-1) + γ0 ΔrM_t + γ1 ΔrM_(t-1) + ε_t`

The speed-of-adjustment parameter and implied long-run beta are reported by NMD segment.

## 8. NMD stability

Kaplan-Meier curves estimate survival without a modelled material runoff event. A Cox proportional-hazards model relates runoff hazard to segment, salary-credit flag, digital flag, customer tenure and spread-to-market.

## 9. Core deposits

A transparent educational approximation is used:

`Core = Stable × (1 - Pass-through)`

The result is constrained by configured segment core-proportion and maturity caps. These caps are demonstration assumptions rather than a claim of current regulatory compliance.

## 10. Loan prepayment

A logistic model estimates monthly prepayment probability using refinancing incentive, seasoning, remaining term, LTV and prepayment-penalty flag. The split is strictly time-based: the training set contains older origination dates and the test set starts after the final training origination date.

Diagnostics include ROC-AUC, PR-AUC, Brier score, standardised coefficients, odds ratios and probability-decile calibration. Monthly predicted prepayment probability is aggregated to SMM and converted to CPR.

## 11. IRRBB integration

Behavioural NMD maturities and prepayment-adjusted loan cash flows feed a discounted-cash-flow engine. Six MYR interest-rate shock paths are used as educational scenario assumptions. EVE is calculated under baseline and shocked curves and reported as ΔEVE and ΔEVE relative to synthetic Tier 1 capital.

## 12. Contractual vs behavioural comparison

The engine runs a contractual view with NMDs treated as immediately repricing and no prepayment adjustment, then compares it with a behavioural view. A bridge isolates the incremental impact of NMD behaviouralisation and loan prepayment on stressed ΔEVE.

## 13. Duration and convexity sanity check

For the fixed-rate loan asset, a +200 bp full discounted-cash-flow revaluation is compared with duration-only and duration-plus-convexity approximations. Behavioural cash flows are held fixed for this check so it tests the price/yield approximation rather than behavioural optionality.

## 14. NII

A simplified 12-month NII projection assumes floating assets reprice more strongly than fixed-rate assets while NMD pricing follows estimated deposit betas. This is intentionally a simplified earnings-sensitivity demonstration.

## 15. Validation controls

The dashboard implements automated developer/model-owner controls for portfolio source reconciliation, accounting identity, deposit-beta range and economic ordering, statistical-signal warnings, NMD core and maturity caps, prepayment ROC-AUC/PR-AUC/Brier score, strict out-of-time split integrity, CPR/SMM arithmetic, CPR bounds, six-scenario completeness, baseline ΔEVE = 0, EVE completeness and scenario sensitivity, and principal cash-flow reconciliation.

These controls are not independent bank model validation.

## 16. Automated tests

The repository includes 29 pytest checks covering mathematical identities, scenario definitions, portfolio scaling/reconciliation, accounting identity, model outputs, NMD caps, out-of-time splitting, EVE calculations, principal cash-flow reconciliation and duration/convexity approximation.

## 17. Monitoring

Synthetic quarterly monitoring snapshots track retail transactional beta, weighted core proportion, portfolio CPR and prepayment ROC-AUC. This demonstrates the type of parameter/performance drift monitoring that could support recalibration decisions.

## 18. Limitations

This is not a production model, regulatory submission or claim of compliance. Synthetic data have no real predictive validity. The balance sheet and product mechanics are simplified. Hedging, CSRBB, basis-risk decomposition, automatic options, accounting effects, management actions, independent validation and formal model governance are outside scope.
