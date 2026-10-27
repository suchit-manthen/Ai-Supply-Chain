"""Central configuration for the forecasting backend.

Single source of truth for values that were previously hard-coded in several
modules (data.py, models.py, app.py). Import from here instead of re-defining.
"""
import os
from statistics import NormalDist

# --- Persistence ------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DB_PATH = os.path.join(DATA_DIR, "stocksense.db")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")

# --- Uploaded datasets ------------------------------------------------------
UPLOAD_MIN_TRAIN_DAYS = 90   # minimum history per SKU required for walk-forward on uploaded data
MAX_SKUS = 25                # safety cap so an accidentally huge CSV cannot freeze the demo
MAX_UPLOAD_ROWS = 200000     # safety cap on uploaded file size (rows)
MAX_FORECAST_ROWS = 200000   # maximum normalized records passed into model training

# --- Forecast setup ---------------------------------------------------------
HORIZON = 28          # days to forecast
HOLDOUT = 28          # validation days used by the single-split backtest
# Prophet only identifies a yearly seasonal cycle with roughly two years of
# history; on a shorter series the yearly Fourier terms are unstable and inflate
# the trend (Prophet's own "less than 730 days" warning). Below this span we fit
# without yearly seasonality.
PROPHET_YEARLY_MIN_DAYS = 730

# --- Reproducibility --------------------------------------------------------
SEED = 42             # seed for the synthetic data generator

# --- Forecast uncertainty ---------------------------------------------------
# Two-sided prediction interval coverage shown on the dashboards. The per-horizon
# error scale is estimated from rolling-origin backtest residuals (see models.py).
FORECAST_INTERVAL_LEVEL = 0.80
INTERVAL_Z = round(NormalDist().inv_cdf(0.5 + FORECAST_INTERVAL_LEVEL / 2), 3)  # ~1.282 for 80%

# --- Inventory policy -------------------------------------------------------
SERVICE_LEVEL = 0.95  # target cycle service level used for safety stock
Z = round(NormalDist().inv_cdf(SERVICE_LEVEL), 3)  # == 1.645 for 95%
ORDERING_COST = 50.0  # fixed cost per replenishment order ($), for the EOQ reference
HOLDING_RATE = 0.20   # annual holding cost as a fraction of unit cost ($/$/yr), for EOQ

# --- Walk-forward (rolling-origin) validation -------------------------------
N_FOLDS = 3           # number of evaluation folds
MIN_TRAIN_DAYS = 365  # minimum history required before the first fold's test window

# --- Automatic model selection ----------------------------------------------
SELECTION_METRIC = "mape"           # primary metric: lowest mean validation value wins
SELECTION_FALLBACK_METRIC = "rmse"  # used when MAPE is unreliable (near-zero demand)
LOW_DEMAND_THRESHOLD = 5.0          # mean units/day below which MAPE is considered unreliable

# --- Monitoring (validation drift between runs) -----------------------------
# A SKU is flagged as "deteriorated" if its selected-model validation MAPE rises
# versus the previous run by at least EITHER threshold (relative OR absolute).
DRIFT_REL_THRESHOLD = 0.15          # +15% relative MAPE increase
DRIFT_ABS_POINTS = 2.0              # +2.0 MAPE points
