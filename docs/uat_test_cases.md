# UAT Test Cases

| Test ID | Requirement | Input | Expected result |
|---|---|---|---|
| UAT-01 | CPR/SMM conversion | CPR 20% | Round-trip returns 20% within numerical tolerance |
| UAT-02 | Parallel-up shock | MYR curve +200 bps | All 19 bucket rates increase by 200 bps |
| UAT-03 | NMD cap | Retail transactional raw core >90% | Applied core proportion <=90% |
| UAT-04 | Prepayment stress | Base CPR x 1.2 | Stressed CPR equals multiplier, capped at 100% |
| UAT-05 | EVE scenarios | Baseline + six shocks | Seven EVE rows, six non-baseline ΔEVE values |
| UAT-06 | Negative balances | Synthetic portfolio | No negative generated outstanding balances |
