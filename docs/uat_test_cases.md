# UAT Test Cases

| Test ID | Requirement | Input | Expected result |
|---|---|---|---|
| UAT-01 | CPR/SMM conversion | CPR 20% | Round-trip returns 20% within numerical tolerance |
| UAT-02 | Parallel-up shock | MYR curve +200 bps | All 19 bucket rates increase by 200 bps |
| UAT-03 | NMD cap | Raw core above category cap | Applied core proportion does not exceed configured cap |
| UAT-04 | Prepayment stress | Base CPR × scenario multiplier | Stressed CPR equals multiplier result, bounded to [0, 100%] |
| UAT-05 | EVE scenarios | Baseline + six shocks | Seven EVE rows and baseline ΔEVE = 0 |
| UAT-06 | Negative balances | Generated loan portfolio | No negative outstanding balances |
| UAT-07 | Loan source reconciliation | Generated loan accounts | Sum outstanding balance equals ALM fixed-rate loan amount |
| UAT-08 | NMD source reconciliation | Latest generated NMD balances | Sum latest NMD balances equals ALM NMD amounts |
| UAT-09 | Accounting identity | Reconciled balance sheet | Assets = liabilities + synthetic Tier 1 within tolerance |
| UAT-10 | Cash-flow reconciliation | Baseline behavioural cash flows | Principal cash flows reconcile to product principal within tolerance |
| UAT-11 | Out-of-time split | Loan origination dates | Test start date is strictly later than training end date |
| UAT-12 | Beta ordering | Synthetic NMD design | Retail transactional < retail non-transactional < wholesale |
| UAT-13 | Duration/convexity sanity | +200 bp fixed-loan shock | Duration+convexity approximation is at least as close as duration-only |
| UAT-14 | Validation transparency | Weak statistical signal | Control is reported as WARNING rather than hidden or forced to PASS |
