"""ML layer: four candidate forecasting algorithms plus the shared evaluation harness.

No algorithm is hard-wired to a SKU. Every SKU is evaluated against all four
candidates with walk-forward validation and the best-performing one is selected
by selection.py.

ARIMA (SARIMAX)  -> smooth, autocorrelated demand with a weekly cycle
Prophet          -> trend + weekly/yearly seasonality with holiday windows
XGBoost          -> non-linear interactions (discount, price, weather, calendar)
Random Forest    -> noisy / outlier-prone demand
"""
import logging, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf
from xgboost import XGBRegressor
from config import HOLDOUT, HORIZON, FORECAST_INTERVAL_LEVEL, INTERVAL_Z, PROPHET_YEARLY_MIN_DAYS

warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

MODEL_INFO = {
    "ARIMA": dict(aspect="Smooth, autocorrelated demand with a weekly cycle",
                  why="SARIMAX(2,1,1)(1,0,1)[7] captures trend and a 7-day seasonal cycle with very few "
                      "parameters and can use planned discounts as an exogenous regressor. A strong "
                      "candidate when demand is stable and self-similar over time.",
                  color="#3B5BDB"),
    "Prophet": dict(aspect="Trend + weekly/yearly seasonality with holidays",
                    why="Prophet decomposes a series into trend, weekly and yearly seasonality and explicit "
                        "holiday windows, with discount and temperature as extra regressors. A strong "
                        "candidate when seasonality or calendar effects dominate.",
                    color="#E8590C"),
    "XGBoost": dict(aspect="Non-linear feature interactions (promo, price, weather, calendar)",
                    why="Gradient-boosted trees learn non-linear interactions between engineered features "
                        "(discounts, price, temperature, day-of-week). A strong candidate when promotions "
                        "or weather drive demand in a non-additive way.",
                    color="#2F9E44"),
    "Random Forest": dict(aspect="Noisy demand, robust to outliers",
                          why="Bagging many decorrelated trees averages out noise and resists overfitting "
                              "on volatile series. A strong candidate when demand is erratic or contains "
                              "outliers.",
                          color="#AE3EC9"),
}

# Candidate algorithms, in evaluation order. Every SKU is tested against all of them.
ALL_MODELS = ["ARIMA", "Prophet", "XGBoost", "Random Forest"]

# Calendar features are always available (derived from the date index).
CAL_FEATS = ["dow", "month", "doy", "weekofyear", "is_weekend", "sin_doy", "cos_doy", "is_holiday"]
# Optional exogenous drivers. Only the ones that actually vary are used, so a
# missing/constant/zero column is dropped rather than fed as a meaningless 0.
EXOG_CANDIDATES = ["discount", "price", "temp", "holiday_lift"]
FEATS = CAL_FEATS + EXOG_CANDIDATES  # documented superset; _features returns the active subset


def _varies(series):
    """True when a column carries usable information (has values and varies)."""
    s = pd.to_numeric(series, errors="coerce")
    if not bool(s.notna().any()):
        return False
    spread = s.std(ddof=0)
    return bool(spread is not None and not np.isnan(spread) and spread > 0)


def available_exog(d):
    """Optional drivers present in `d` that vary (and are therefore worth using)."""
    return [c for c in EXOG_CANDIDATES if c in d.columns and _varies(d[c])]


def _features(d, exog=None):
    """Feature matrix: calendar terms plus the active optional drivers.

    Passing a fixed `exog` list keeps train and predict columns aligned.
    """
    exog = available_exog(d) if exog is None else list(exog)
    x = pd.DataFrame(index=d.index)
    x["dow"] = d.date.dt.dayofweek
    x["month"] = d.date.dt.month
    x["doy"] = d.date.dt.dayofyear
    x["weekofyear"] = d.date.dt.isocalendar().week.astype(int)
    x["is_weekend"] = (x["dow"] >= 5).astype(int)
    angle = 2 * np.pi * x["doy"] / 365.25
    x["sin_doy"] = np.sin(angle)
    x["cos_doy"] = np.cos(angle)
    if "holiday_lift" in d.columns:
        hl = pd.to_numeric(d["holiday_lift"], errors="coerce").fillna(0.0)
    else:
        hl = pd.Series(0.0, index=d.index)
    x["is_holiday"] = (hl > 0).astype(int)
    for c in exog:
        x[c] = d[c]
    return x


def _metrics(y, p, naive):
    """Forecast error metrics for one test window.

    MAPE divides by max(actual, 1) so that zero-demand days do not divide by zero
    (and remain finite). `naive` is the lag-7 "same day last week" baseline,
    evaluated over exactly the same window.
    """
    y, p, naive = np.asarray(y, dtype=float), np.asarray(p, dtype=float), np.asarray(naive, dtype=float)
    mae = float(np.mean(np.abs(y - p))); rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    mape = float(np.mean(np.abs(y - p) / np.maximum(y, 1)) * 100)
    nm = float(np.mean(np.abs(y - naive) / np.maximum(y, 1)) * 100)
    ss_res = float(np.sum((y - p) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 0.0 if ss_tot == 0 else float(1 - ss_res / ss_tot)
    return dict(mae=round(mae, 1), rmse=round(rmse, 1), mape=round(mape, 2), naive_mape=round(nm, 2),
                accuracy=round(100 - mape, 1), r2=round(r2, 3))


def naive_baseline(hist, start, horizon=HOLDOUT):
    """Lag-7 ("same day last week") baseline for the window [start, start+horizon).

    Uses the same history as the ML model so baseline and model are compared on
    identical test periods. Falls back to the window mean where lag-7 is missing
    (only possible if the history is shorter than a week).
    """
    vals = hist["sales"].shift(7).to_numpy(dtype=float)[start:start + horizon]
    if np.isnan(vals).any():
        fill = np.nanmean(vals) if not np.isnan(vals).all() else np.nanmean(hist["sales"].to_numpy(dtype=float))
        vals = np.where(np.isnan(vals), fill, vals)
    return vals


def horizon_error_scale(folds, horizon=HORIZON):
    """Estimate a per-horizon forecast error scale from rolling-origin residuals.

    Each fold has out-of-sample predictions for horizon steps 1..H, so the h-step
    residual is actual[h] - predicted[h]. We take the RMS residual at each step
    (a direct, data-driven estimate of the error scale for that horizon) and then
    apply isotonic regression, the best non-decreasing fit to those points. This
    guarantees the error scale is monotone - uncertainty widens where the model's
    backtest error widens, and stays flat where it does not - without inventing
    arbitrary growth percentages. Returns an array of length `horizon`.
    """
    errors = np.full((len(folds), horizon), np.nan)
    for i, f in enumerate(folds):
        actual = np.asarray(f["actual"], dtype=float)
        pred = np.asarray(f["pred"], dtype=float)
        n = int(min(len(actual), len(pred), horizon))
        errors[i, :n] = actual[:n] - pred[:n]
    if len(folds) == 0 or np.all(np.isnan(errors)):
        return np.zeros(horizon)
    with np.errstate(invalid="ignore"):
        rms = np.sqrt(np.nanmean(errors ** 2, axis=0))
    rms = np.nan_to_num(rms, nan=0.0)
    if np.allclose(rms, 0):
        return np.zeros(horizon)
    h = np.arange(1, horizon + 1, dtype=float)
    scale = IsotonicRegression(increasing=True, out_of_bounds="clip").fit_transform(h, rms)
    return np.maximum(scale, 0.0)


def prediction_interval(preds, error_scale, z=INTERVAL_Z):
    """Two-sided prediction interval for point forecasts.

    Returns (lower, upper) arrays. lower <= forecast <= upper is guaranteed.
    """
    preds = np.asarray(preds, dtype=float)
    scale = np.asarray(error_scale, dtype=float)
    if scale.ndim == 0:
        scale = np.full(preds.shape, float(scale))
    half = np.maximum(z * scale[:len(preds)], 0.0)
    return np.maximum(0.0, preds - half), preds + half


# ---- individual algorithms: fit(train_df, predict_df) -> (pred, extras) --------------------------------------
def fit_arima(tr, te):
    exog = ["discount"] if ("discount" in tr.columns and _varies(tr["discount"])) else []
    kwargs = dict(order=(2, 1, 1), seasonal_order=(1, 0, 1, 7),
                  enforce_stationarity=False, enforce_invertibility=False)
    if exog:
        m = SARIMAX(tr.sales.values, exog=tr[exog].values, **kwargs).fit(disp=False, maxiter=80)
        f = m.get_forecast(len(te), exog=te[exog].values)
    else:
        m = SARIMAX(tr.sales.values, **kwargs).fit(disp=False, maxiter=80)
        f = m.get_forecast(len(te))
    resid = m.resid[14:]
    return f.predicted_mean, dict(
        order="(2,1,1)", seasonal_order="(1,0,1)[7]", aic=round(float(m.aic), 1), bic=round(float(m.bic), 1),
        params={k: round(float(v), 3) for k, v in zip(m.param_names, m.params)}, features_used=exog,
        acf=[round(float(v), 3) for v in acf(tr.sales.diff().dropna().values, nlags=21)],
        resid_acf=[round(float(v), 3) for v in acf(resid, nlags=21)],
        resid_std=round(float(np.std(resid)), 1))


def fit_prophet(tr, te, hol):
    from prophet import Prophet
    # With no yearly cycle to identify, the yearly Fourier fit is unstable and, in
    # multiplicative mode, inflates the trend into wildly wrong forecasts. Only
    # enable it once the training history spans at least two years.
    span_days = int((tr["date"].max() - tr["date"].min()).days)
    yearly = span_days >= PROPHET_YEARLY_MIN_DAYS
    m = Prophet(yearly_seasonality=yearly, weekly_seasonality=True, daily_seasonality=False,
                seasonality_mode="multiplicative", holidays=hol.assign(lower_window=-7, upper_window=1),
                changepoint_prior_scale=.02, holidays_prior_scale=3, seasonality_prior_scale=5)
    regressors = [c for c in ("discount", "temp") if c in tr.columns and _varies(tr[c])]
    for c in regressors:
        m.add_regressor(c)
    cols = ["ds", "y"] + regressors
    m.fit(tr.rename(columns={"date": "ds", "sales": "y"})[cols])
    allx = pd.concat([tr, te]).rename(columns={"date": "ds"})[["ds"] + regressors]
    fc = m.predict(allx)
    comp = fc.iloc[:len(tr)]
    # reindex so short histories (missing weekdays/months) still produce a full component vector.
    # A component column may be absent when that seasonality is disabled (e.g. yearly on short data).
    if "weekly" in comp.columns:
        dow = comp.assign(d=comp.ds.dt.dayofweek).groupby("d").weekly.mean().reindex(range(7)).fillna(0.0)
    else:
        dow = pd.Series(0.0, index=range(7))
    if "yearly" in comp.columns:
        mon = comp.assign(mo=comp.ds.dt.month).groupby("mo").yearly.mean().reindex(range(1, 13)).fillna(0.0)
    else:
        mon = pd.Series(0.0, index=range(1, 13))
    tail = fc.iloc[len(tr):]
    return tail.yhat.values, dict(
        trend=[dict(date=str(r.ds.date()), v=round(float(r.trend), 1)) for r in comp.iloc[::21].itertuples()],
        weekly=[round(float(dow[i]) * 100, 1) for i in range(7)],
        yearly=[round(float(mon[i]) * 100, 1) for i in range(1, 13)],
        holiday_effect=round(float(comp.holidays.abs().max()) * 100, 1),
        changepoints=int(len(m.changepoints)), mode="multiplicative", features_used=regressors)


def fit_xgb(tr, te):
    exog = available_exog(tr)
    x_tr, x_te = _features(tr, exog), _features(te, exog)
    m = XGBRegressor(n_estimators=450, learning_rate=.04, max_depth=5, subsample=.85, colsample_bytree=.85,
                     min_child_weight=3, reg_lambda=2, random_state=1, n_jobs=2)
    m.fit(x_tr, tr.sales)
    imp = sorted(zip(list(x_tr.columns), m.feature_importances_), key=lambda t: -t[1])
    return m.predict(x_te), dict(
        params=dict(n_estimators=450, learning_rate=.04, max_depth=5, subsample=.85, colsample_bytree=.85),
        features_used=exog, importance=[dict(feature=k, value=round(float(v) * 100, 1)) for k, v in imp])


def fit_rf(tr, te):
    exog = available_exog(tr)
    x_tr, x_te = _features(tr, exog), _features(te, exog)
    m = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, max_features=.7, random_state=1, n_jobs=2)
    m.fit(x_tr, tr.sales)
    imp = sorted(zip(list(x_tr.columns), m.feature_importances_), key=lambda t: -t[1])
    # spread across the individual trees = model uncertainty
    per_tree = np.stack([t.predict(x_te.values) for t in m.estimators_])
    return m.predict(x_te), dict(
        params=dict(n_estimators=300, min_samples_leaf=3, max_features=.7), features_used=exog,
        importance=[dict(feature=k, value=round(float(v) * 100, 1)) for k, v in imp],
        tree_spread=round(float(per_tree.std(axis=0).mean()), 1))


def _run(algo, tr, te, hol):
    if algo == "ARIMA": return fit_arima(tr, te)
    if algo == "Prophet": return fit_prophet(tr, te, hol)
    if algo == "XGBoost": return fit_xgb(tr, te)
    return fit_rf(tr, te)


def forecast_product(algo, hist, fut, hol, folds=None):
    """Backtest on the last HOLDOUT days, then refit on all history to forecast the future.

    `folds` (optional) are the rolling-origin validation folds for this model.
    When supplied, the prediction interval is horizon-aware: the per-step error
    scale comes from the folds' out-of-sample residuals and widens with horizon.
    Otherwise a flat RMSE band is used as a fallback.
    """
    split = len(hist) - HOLDOUT
    train, val = hist.iloc[:split], hist.iloc[split:]
    vp, _ = _run(algo, train, val, hol)
    vp = np.maximum(np.asarray(vp, dtype=float), 0)
    naive = naive_baseline(hist, split, HOLDOUT)
    met = _metrics(val.sales.values, vp, naive)
    fp, extras = _run(algo, hist, fut, hol)
    fp = np.maximum(np.asarray(fp, dtype=float), 0)
    horizon = len(fut)
    if folds:
        scale = horizon_error_scale(folds, horizon)
        method, level, z = "horizon_error", FORECAST_INTERVAL_LEVEL, INTERVAL_Z
    else:
        scale = np.full(horizon, float(met["rmse"]))
        method, level, z = "flat_rmse", 0.80, 1.28  # legacy fallback
    lo, hi = prediction_interval(fp, scale, z)
    # Reconstruct a holdout interval from training-only residual scale; avoid
    # using the same holdout outcomes to set their own interval width.
    train_rmse = float(np.sqrt(np.mean((val.sales.to_numpy(dtype=float) - vp) ** 2)))
    holdout_scale = np.full(len(val), train_rmse)
    holdout_lo, holdout_hi = prediction_interval(vp, holdout_scale, z)
    covered = (val.sales.to_numpy(dtype=float) >= holdout_lo) & (val.sales.to_numpy(dtype=float) <= holdout_hi)
    recent = hist.iloc[-90:]
    return dict(
        metrics=met, extras=extras,
        feature_usage=dict(calendar=list(CAL_FEATS), exog=available_exog(hist)),
        interval_coverage=dict(covered=int(covered.sum()), total=int(len(covered)),
                               pct=round(float(covered.mean() * 100), 1), level=level),
        uncertainty=dict(method=method, level=level, z=z,
                         error_scale=[round(float(s), 2) for s in scale]),
        backtest=[dict(date=str(d.date()), actual=float(a), pred=round(float(p), 1)) for d, a, p in zip(val.date, val.sales, vp)],
        history=[dict(date=str(d.date()), actual=float(a)) for d, a in zip(recent.date, recent.sales)],
        forecast=[dict(date=str(d.date()), pred=round(float(p), 1),
                       lo=round(float(l), 1), hi=round(float(u), 1),
                       lower_bound=round(float(l), 1), upper_bound=round(float(u), 1),
                       horizon=i + 1)
                  for i, (d, p, l, u) in enumerate(zip(fut.date, fp, lo, hi))],
        residuals=[round(float(a - p), 1) for a, p in zip(val.sales, vp)])
