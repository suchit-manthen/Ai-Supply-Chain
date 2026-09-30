from config import (HORIZON, HOLDOUT, SEED, SERVICE_LEVEL, Z, N_FOLDS, MIN_TRAIN_DAYS,
                    SELECTION_METRIC, SELECTION_FALLBACK_METRIC, LOW_DEMAND_THRESHOLD,
                    FORECAST_INTERVAL_LEVEL, INTERVAL_Z, DRIFT_REL_THRESHOLD, DRIFT_ABS_POINTS,
                    ORDERING_COST, HOLDING_RATE)


def test_config_values_are_sane():
    assert HORIZON > 0
    assert HOLDOUT > 0
    assert SEED == 42
    assert 0 < SERVICE_LEVEL < 1
    assert N_FOLDS >= 1
    assert MIN_TRAIN_DAYS > 0
    assert SELECTION_METRIC == "mape"
    assert SELECTION_FALLBACK_METRIC == "rmse"
    assert LOW_DEMAND_THRESHOLD > 0
    assert 0 < FORECAST_INTERVAL_LEVEL < 1
    assert round(INTERVAL_Z, 3) == 1.282  # 80% two-sided interval
    assert 0 < DRIFT_REL_THRESHOLD < 1
    assert DRIFT_ABS_POINTS > 0
    assert ORDERING_COST > 0
    assert 0 < HOLDING_RATE < 1


def test_z_matches_95_percent_service_level():
    assert round(Z, 3) == 1.645


def test_config_is_single_source_of_truth():
    # data.py re-exports the same objects rather than redefining them
    import data
    assert data.HORIZON == HORIZON
    assert data.HOLDOUT == HOLDOUT
    assert data.SEED == SEED
