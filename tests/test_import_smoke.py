def test_deployed_module_imports():
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
    from src.irrbb import (
        calculate_eve,
        calculate_nii,
        cashflow_reconciliation,
        duration_convexity_check,
        scenario_curves,
    )
    from src.prepayment import calibration_table, portfolio_cpr, train_prepayment_model
    from src.validation import validation_summary

    assert callable(behavioural_bridge)
    assert callable(calculate_eve)
    assert callable(build_balance_sheet)
    assert callable(train_prepayment_model)
    assert callable(validation_summary)
