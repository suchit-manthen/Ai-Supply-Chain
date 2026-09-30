# StockSense – AI Supply Chain Demand Forecasting (demo)

React + Vite frontend, Flask backend, synthetic supermarket data, four ML models.

## Run it (two terminals)
Backend (Python 3.9+):
    cd backend
    python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    python train.py                                      # one training run -> saves to SQLite (~60-90 s)
    python app.py                                        # http://localhost:5000 (serves the saved run, no retrain)

Frontend (Node 18+):
    cd frontend
    npm install
    npm run dev                                          # open http://localhost:5173

## Training vs serving
Training is separated from the API. `python train.py` (or `POST /api/retrain`)
runs the full pipeline and persists a run (timestamped run ID, forecasts,
metrics, selected models, inventory results) to `backend/data/stocksense.db`
(SQLite). `python app.py` only loads the latest completed run, so restarting the
backend is fast and does not retrain. If no run exists yet, the dashboard shows
a "Train models" button.

## Data & inventory
Data comes through `datasource.load_data`: `None` uses the synthetic generator;
a `.csv` path loads observed sales and synthesizes the 28-day future window.
The canonical schema is `date, product, sales, discount, price, temp,
holiday_lift` (plus an optional `stockout` flag). `datasource.validate_sales`
checks for missing dates, missing/unknown SKUs, duplicate records, invalid
prices, negative sales and missing demand observations; error-severity problems
abort training with a clear message. Inventory planning uses a configurable
service level (`SERVICE_LEVEL`, or `--service-level` on `train.py`), per-SKU lead
times, horizon-aware safety stock, a reorder point and an order quantity, with a
classic EOQ reference (`ORDERING_COST`, `HOLDING_RATE`).

### Dataset support

- **Synthetic demo (default):** the built-in eight-product supermarket simulator
  remains the default and can be selected at any time from the dashboard.
- **StockSense CSV:** the existing upload accepts `date, product, sales` and
  optional `category, price, discount, temp, holiday_lift` columns.
- **Favorita:** the first real-data adapter accepts the Kaggle
  `Corporación Favorita Grocery Sales Forecasting` `train.csv` columns
  `date, store_nbr, item_nbr, unit_sales` and optional `onpromotion`.

To try Favorita, choose **Upload CSV → Favorita train.csv**, select `train.csv`,
then select one store, up to eight items, and a date range. The adapter scans the
source in 100,000-row chunks, so it does not build a full in-memory DataFrame.
There is no raw row-count limit on Favorita uploads (the upload still has a
2 GB file-size ceiling). After filtering and daily duplicate aggregation, the
normalized dataset is capped at 200,000 rows; larger selections ask the user to
narrow the store, items, or date range.
It reports file row count, date range, store/item counts, available columns and
data-quality counts before selection. Preparation filters to the selected
store/items/dates and sums duplicate `date + store_nbr + item_nbr` sales rows;
promotion flags are combined with logical OR. The normalized subset is handed
to the same model selection, forecasting, inventory and dashboard pipeline.

The current Favorita integration is deliberately small: the UI permits up to
eight selected items and requires each selected SKU to have at least 118
observed days (90 minimum training days plus the 28-day forecast horizon) with
no missing calendar dates.
Missing sales observations are excluded and never assumed to be zero; missing
calendar days block training because the existing forecasting pipeline expects
daily observations. Negative aggregated sales (returns) are reported as an
actionable error and must be resolved explicitly. `onpromotion` is used as a
binary signal in the existing discount regressor; it does not encode discount
depth. `train.csv` has no unit price, weather, or stock-on-hand fields, so the
forecast uses neutral placeholder values for absent model inputs and the user
must configure inventory assumptions, including a positive unit cost per item.
The suggested opening stock is an editable estimate from the last 14 observed
days, not a measured stock quantity. The adapter does not implement store/item
metadata joins, transactions, stockout correction, or full-catalog forecasting.

## Monitoring
Each run stores its timestamp, dataset info, per-SKU selected model and validation
MAPE/RMSE/MAE plus the naive baseline. `GET /api/runs` returns training history,
`GET /api/runs/<id>` a single run, and `GET /api/monitoring` compares the latest
run's validation error against the naive baseline and the previous run, flagging
deterioration above the configurable thresholds (`DRIFT_REL_THRESHOLD`,
`DRIFT_ABS_POINTS`). The analyst dashboard's "Model runs" tab shows this.

## Model selection (data-driven, not hard-coded)
Every SKU is evaluated against all four algorithms — ARIMA (SARIMAX), Prophet, XGBoost
and Random Forest — using walk-forward / rolling-origin validation (3 folds). The model
with the lowest mean validation MAPE is selected, refit on all history, and used to
forecast the next 28 days. The naive "same day last week" (lag-7) baseline is evaluated
on identical windows and shown for reference. No algorithm is assumed to suit any SKU.

## Forecast uncertainty & inventory
Prediction intervals are horizon-aware: the error scale for each of the 28 days is
estimated from the selected model's walk-forward backtest residuals (a non-decreasing
isotonic fit), so the band widens where the model's backtest error widens. Safety stock
uses that per-horizon error over the lead time, and stockout risk is derived from the
upper interval versus current stock.
