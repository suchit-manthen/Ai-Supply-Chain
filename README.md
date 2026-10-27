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

## Demo sign-in

The frontend opens on a sign-in page with two demo workspaces. The credentials
are also shown on the role cards and are prefilled when a role is selected:

| Workspace | Email | Password |
|---|---|---|
| Store Owner | `owner@stocksense.demo` | `owner123` |
| ML Analyst | `analyst@stocksense.demo` | `analyst123` |

The Store Owner workspace opens the inventory dashboard and simulator. The ML
Analyst workspace opens the model workbench and simulator. The session is stored
in `localStorage` so a refresh keeps you signed in; use **Sign out** in the top
bar to return to the login page. This is a client-side demo gate, not production
authentication — the Flask API itself is unchanged.

## Testing the app

Synthetic quick start: run the two terminals above, open http://localhost:5173
and use **Upload CSV** (or the "Train models" button for the synthetic demo).

Ready-made files are in `data/samples/` (regenerate with
`python scripts/fetch_test_data.py`):

| File | Schema | Wizard path |
|---|---|---|
| `store_family_sample.csv` | `date,store_nbr,family,sales,onpromotion` (real Favorita "Store Sales" trim) | store + family |
| `generic_long_sample.csv` | arbitrary headers over the same data | generic mapping |
| `wide_sample.csv` | `date` + product columns (matrix) | generic wide |
| `favorita_item_sample.csv` | `date,store_nbr,item_nbr,unit_sales,onpromotion` | Favorita |

Each sample has one store and >=150 daily rows per series (above the 118-day
minimum). Upload one, walk the 7 steps (upload -> understand -> map -> quality ->
select -> inventory -> forecast) and the dashboard loads when training finishes.
The script trims the real Favorita file with a partial HTTP Range download, so it
stays a few MB.

## Real-world datasets

Further datasets that match the adapters (schemas shown), verified on the hubs:

- **Favorita store + family** (Kaggle "Store Sales - Time Series Forecasting"):
  Hugging Face `mrcksggcfc/store-sales-time-series-forecasting` and
  `t4tiana/store-sales-time-series-forecasting` -> `train.csv`
  (`date, store_nbr, family, sales, onpromotion`), with `stores.csv`,
  `holidays_events.csv`, `oil.csv`, `transactions.csv`.
- **M5 / Walmart**: Hugging Face `denephew/M5_Forecasting` ->
  `sales_train_validation.csv` / `sales_train_evaluation.csv`
  (wide: `id,item_id,dept_id,cat_id,store_id,state_id,d_1..`), plus
  `calendar.csv` and `sell_prices.csv`; use the generic **wide** path.
  `snchakri/m5-retail-demand-forecasting-benchmarks` has processed features.
- **Kaggle CLI** (needs `~/.kaggle/kaggle.json`):
  `kaggle competitions download -c favorita-grocery-sales-forecasting`
  (item-level), `-c store-sales-time-series-forecasting`,
  `-c rossmann-store-sales`, `-c walmart-recruiting-store-sales-forecasting`,
  `-c m5-forecasting-accuracy`.

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

### Upload wizard

Every CSV goes through one guided flow (frontend `Upload.jsx` → `GenericUpload.jsx`
or `FavoritaUpload.jsx`, stepper in `Wizard.jsx`):

1. **Upload CSV** — a single entry point; the file is streamed to the backend and
   its format detected automatically (Favorita / store+family / arbitrary).
2. **Understand data** — layout (long/wide), date format, columns and the
   auto-detected role mapping.
3. **Column mapping** — confirm or change each role; for wide files pick the
   value columns and optional id column.
4. **Data quality** — the full-file, chunked quality report.
5. **Store / SKU / date selection** — filters applied before the forecasting cap.
6. **Inventory assumptions** — per-SKU stock, lead time, unit cost, service level.
7. **Start forecasting** — review and launch.

### Dataset support

- **Synthetic demo (default):** the built-in eight-product supermarket simulator
  remains the default and can be selected at any time from the dashboard.
- **StockSense CSV:** the existing upload accepts `date, product, sales` and
  optional `category, price, discount, temp, holiday_lift` columns.
- **Favorita:** the first real-data adapter accepts the Kaggle
  `Corporación Favorita Grocery Sales Forecasting` `train.csv` columns
  `date, store_nbr, item_nbr, unit_sales` and optional `onpromotion`.
- **Store + family:** a companion adapter accepts aggregated exports with columns
  `date, store_nbr, family, sales` and optional `onpromotion` (for example the
  `data/train.csv` store/category file). It streams the file in chunks, asks for
  one store and up to eight product families, and normalizes the selection into
  the same pipeline format. Missing sales rows are excluded and missing calendar
  dates block training rather than being filled with zero.
- **Generic CSV (auto-detect):** any other sales export is profiled column by
  column. Each column is scored against the semantic roles `date`, `sales/demand`,
  `product/entity`, `store` and the optional drivers (category, price, discount,
  promotion, temperature, holiday, stock-out), and returned with a confidence
  score and suggested mapping. Only `date + product/entity + sales` are required;
  every other role is optional. The user confirms or overrides the mapping in the
  UI before the file is normalized to the canonical schema, so a completely
  different header set can feed the same model-selection and inventory pipeline.
  The profiler also detects **wide (matrix) vs long** layouts and melds a wide
  file into long rows; it parses many date formats (ISO, `d/m/Y` vs `m/d/Y`,
  `YYYYMMDD` integers, textual months), aggregates duplicate `product+date`
  rows, and reports a normalization summary (date format used, duplicates
  collapsed, rows dropped for missing required values, wide cells filled with 0).
  Missing optional fields fall back to neutral defaults.
- **Large files:** the generic adapter streams the raw file in chunks with **no
  raw row cap**. Store, SKU/product and date filters are applied per chunk so a
  huge source is reduced *before* the 200,000-row forecasting limit; only the
  filtered, aggregated set is capped (narrow the filters if it is exceeded). The
  same chunked scan produces a **dataset quality report** (rows scanned, invalid
  dates, non-numeric/missing/negative sales, missing ids, entity/store counts and
  date range), shown on upload and after preparation.

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

## Feature enrichment & graceful degradation

- **Automatic feature use:** the model layer detects which optional drivers
  (`discount`, `price`, `temp`, `holiday_lift`) are present *and vary* per SKU and
  only feeds those to ARIMA/Prophet/XGBoost/Random Forest. A missing or constant
  column (e.g. no price, or an all-zero temperature) is dropped rather than fed as
  a meaningless zero, so all-zero placeholders cannot distort the models.
- **Calendar / holiday enrichment:** a US holiday calendar (federal holidays plus
  retail events such as Black Friday and Cyber Monday) drives the holiday-lift for
  both synthetic and uploaded data automatically, and clear calendar terms
  (day-of-week, month, day-of-year, ISO week, weekend flag, annual Fourier
  sin/cos, holiday flag) are always available.
- **Category / family support:** the canonical dataset carries the `category`
  column end to end (including the future window), so the dashboard shows a
  per-category breakdown and SKUs keep their family label. A file whose only
  grouping column is a family/category is treated as the entity.
- **Missing optional features:** absent drivers are filled with neutral values for
  storage (`discount` 0, `temp` last-observed, `price` per-SKU median) and then
  excluded from modelling when they carry no signal; the analyst view lists only
  the features actually used.

## Model selection (data-driven, not hard-coded)
Every SKU is evaluated against all four algorithms — ARIMA (SARIMAX), Prophet, XGBoost
and Random Forest — using walk-forward / rolling-origin validation (3 folds). The model
with the lowest mean validation MAPE is selected, refit on all history, and used to
forecast the next 28 days. The naive "same day last week" (lag-7) baseline is evaluated
on identical windows and shown for reference. No algorithm is assumed to suit any SKU.
Prophet's yearly seasonality is enabled only when the training history spans at
least two years (`PROPHET_YEARLY_MIN_DAYS`); on shorter series the yearly Fourier
fit is unidentified and inflates the trend into runaway forecasts, so it is
disabled there. All four models and the baseline share the same folds and metric.

## Forecast uncertainty & inventory
Prediction intervals are horizon-aware: the error scale for each of the 28 days is
estimated from the selected model's walk-forward backtest residuals (a non-decreasing
isotonic fit), so the band widens where the model's backtest error widens. Safety stock
uses that per-horizon error over the lead time, and stockout risk is derived from the
upper interval versus current stock.
