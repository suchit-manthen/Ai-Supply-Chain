import pandas as pd

from models import ALL_MODELS, MODEL_INFO
from selection import mean_metrics, choose_metric, select_best


def _make_eval(mape_by_model, rmse_by_model=None, baseline=None):
    """Build a minimal evaluation dict shaped like evaluate_all_models()."""
    rmse_by_model = rmse_by_model or {}
    models = {}
    for a in ALL_MODELS:
        m = dict(mae=1.0, rmse=rmse_by_model.get(a, 1.0), mape=mape_by_model.get(a, 10.0),
                 naive_mape=12.0, r2=0.5)
        models[a] = dict(folds=[dict(metrics=m), dict(metrics=m)], mean=m)
    b = dict(mae=1.0, rmse=1.0, mape=12.0, naive_mape=12.0, r2=0.0)
    if baseline:
        b.update(baseline)
    return dict(models=models, baseline=dict(folds=[dict(metrics=b)], mean=b))


def test_all_models_match_model_info():
    assert ALL_MODELS == list(MODEL_INFO.keys())
    assert len(ALL_MODELS) == 4


def test_mean_metrics_averages_across_folds():
    folds = [dict(metrics=dict(mae=1, rmse=2, mape=3, naive_mape=4, r2=0.5)),
             dict(metrics=dict(mae=3, rmse=4, mape=5, naive_mape=6, r2=0.7))]
    out = mean_metrics(folds)
    assert out["mae"] == 2.0
    assert out["rmse"] == 3.0
    assert out["mape"] == 4.0
    assert out["r2"] == 0.6


def test_choose_metric_uses_mape_normally():
    assert choose_metric(100.0) == ("mape", False)


def test_choose_metric_falls_back_on_low_demand():
    assert choose_metric(1.0) == ("rmse", True)


def test_select_best_picks_lowest_mean_mape():
    hist = pd.DataFrame(dict(sales=[100.0] * 10))
    ev = _make_eval({"ARIMA": 5.0, "Prophet": 7.0, "XGBoost": 6.0, "Random Forest": 8.0})
    choice = select_best(ev, hist)
    assert choice["model"] == "ARIMA"
    assert choice["metric"] == "mape"
    assert choice["score"] == 5.0
    assert choice["fallback_used"] is False
    assert choice["improvement"] == 7.0  # baseline 12.0 - 5.0
    assert "Lowest mean validation MAPE" in choice["reason"]
    assert choice["ranking"][0]["model"] == "ARIMA"


def test_select_best_falls_back_to_rmse_on_low_demand():
    hist = pd.DataFrame(dict(sales=[1.0] * 10))  # mean demand below threshold
    ev = _make_eval(
        {"ARIMA": 1.0, "Prophet": 2.0, "XGBoost": 3.0, "Random Forest": 4.0},
        rmse_by_model={"ARIMA": 9.0, "Prophet": 1.0, "XGBoost": 5.0, "Random Forest": 6.0},
        baseline={"rmse": 8.0},
    )
    choice = select_best(ev, hist)
    assert choice["metric"] == "rmse"
    assert choice["fallback_used"] is True
    assert choice["model"] == "Prophet"  # lowest RMSE, even though ARIMA has lowest MAPE
    assert "unreliable" in choice["reason"]


def test_select_best_is_deterministic_on_ties():
    hist = pd.DataFrame(dict(sales=[100.0] * 10))
    ev = _make_eval({"ARIMA": 5.0, "Prophet": 5.0, "XGBoost": 5.0, "Random Forest": 5.0})
    first = select_best(ev, hist)["model"]
    second = select_best(ev, hist)["model"]
    assert first == second == "ARIMA"  # ALL_MODELS order breaks ties deterministically
