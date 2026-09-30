# Implementation Plan: AI-Based Supply Chain Demand Forecasting and Inventory Optimization System

## 1. Scope and planning constraints

This plan is derived solely from `PROJECT_SPEC.md`. The repository currently contains only that specification; no application code, dataset files, database schema, or frontend structure exists.

The core delivery path is:

Historical M5 data -> clean forecasting data -> demand features -> evaluated forecasts -> inventory analysis -> reorder recommendation -> API -> web dashboard.

The following decisions must be evidence-based during implementation, rather than assumed now:

- the forecasting horizon;
- the final forecasting model;
- which of safety stock, reorder point, and EOQ are appropriate for the available data;
- whether the optional LSTM is worth implementing;
- any inventory or supplier data derived or simulated in addition to M5 data.

## 2. Recommended architecture

Use four clearly separated layers:

1. **Data and ML workspace (Python):** reproducible dataset preparation, feature generation, training, time-series evaluation, and persisted forecast results. It owns training data and experimental comparison; it does not serve the web UI directly.
2. **Django + Django REST Framework backend:** the system boundary for the React client. It stores product/store/catalog context, inventory and lead-time inputs, forecast results, alert state, and recommendations in PostgreSQL, then exposes them through versioned REST endpoints.
3. **PostgreSQL:** the authoritative operational store for data presented in the application. Dataset-derived and simulated values must retain source/provenance metadata so they are not mistaken for observed operational data.
4. **React + Vite + Tailwind CSS dashboard:** consumes only backend APIs and renders forecast, inventory status, alerts, and recommendations using a charting library.

The ML pipeline should write validated forecast outputs to a defined import/service boundary. Inventory analysis should consume forecast output plus current inventory, lead time, and demand variability, then produce persisted/retrievable results. This keeps the model-selection work independent from UI development.

## 3. Independently implementable phases

### Phase 0 — Repository and reproducibility foundation

**Purpose:** establish a documented, repeatable starting point without building product features.

**Work:**

- Define the planned top-level boundaries for ML/data work, Django backend, React frontend, documentation, and tests.
- Record environment setup and dependency-management choices when implementation begins.
- Define configuration handling for non-versioned database credentials and local dataset locations.
- Define experiment conventions: dataset version/source, split dates, feature set identifier, model parameters, metrics, random seed where applicable, and output location.
- Record that the primary data source is M5 and that any inventory/supplier values not from it are derived or simulated.

**Depends on:** none.

**Exit criteria before Phase 1:** a contributor can identify where each layer will live, how data source/provenance will be recorded, and how an experiment will be reproduced. No model or UI is required.

### Phase 1 — M5 dataset understanding and data contract

**Purpose:** turn the available M5 files into a documented, inspectable input contract.

**Work:**

- Obtain and register the exact M5 dataset files used, without placing assumptions about their local path into application code.
- Profile historical sales, calendar, price, product, and store information available in the dataset.
- Document identifiers, date representation, granularity, relationships, missingness, duplicate expectations, price coverage, and usable date range.
- Define the canonical clean demand-data shape and a data dictionary based on the observed fields.
- Identify which required inventory inputs are absent from M5: current inventory and supplier lead time at minimum. Specify the provenance label and documentation required if they are later supplied, derived, or simulated.
- Define validation checks for the raw-to-clean transformation.

**Depends on:** Phase 0.

**Exit criteria:** data dictionary and source/provenance record completed; raw data passes or fails explicit validation checks; the team can state what data is available versus absent. Do not build inventory logic before this is known.

### Phase 2 — Clean forecasting dataset

**Purpose:** create a validated, repeatable forecasting-ready dataset from M5 inputs.

**Work:**

- Implement preprocessing only after Phase 1 confirms the real M5 schema.
- Join/reshape historical sales with the calendar, selling-price, product, and store context that is actually available.
- Handle missing values and invalid/duplicate records according to documented rules; retain audit counts.
- Keep chronological ordering and prevent use of future observations in an earlier row.
- Produce dataset-quality checks and a reproducible transformation record.

**Depends on:** Phase 1's data contract.

**Exit criteria:** clean output conforms to the canonical shape; validation checks pass; source-to-output row/series reconciliation is documented; a small time-series sample can be manually inspected for chronological correctness.

### Phase 3 — Feature engineering and leakage-safe evaluation design

**Purpose:** produce candidate predictors and a valid experimental protocol before comparing models.

**Work:**

- Implement only features confirmed feasible by the clean data: previous sales, lag features, rolling averages, seasonal/time features, price changes, discounts, holidays, day of week, and month.
- Define feature availability at forecast time; each feature must use only information available by that point.
- Select chronological training/validation/test partitions and, once chosen, document the forecasting horizon.
- Define a common prediction target, feature set versioning, and evaluation data slices for every candidate model.
- Calculate MAE, RMSE, and MAPE consistently; record any later metric additions separately.

**Depends on:** Phase 2.

**Exit criteria:** a leakage review passes; all model candidates can use the same time-aware evaluation protocol; feature-generation tests verify lags/rolling values never read future dates.

### Phase 4 — Baseline demand forecasting and model comparison

**Purpose:** establish empirical performance evidence before selecting a production forecasting approach.

**Work:**

- Implement baseline experiments using applicable candidate models from Random Forest, XGBoost, ARIMA/SARIMA, and LightGBM.
- Compare models with the identical data split, horizon, forecast target, and MAE/RMSE/MAPE calculation defined in Phase 3.
- Record configuration, runtime considerations, metric results, failure cases, and forecast plots/tables for review.
- Select a final model from the evidence. LSTM remains optional and may be evaluated only if it is justified after the core candidates work.
- Define a reproducible training-and-forecast execution flow for the selected model and its output contract: item/store scope as supported by data, forecast period, predicted demand, model/experiment version, and generation timestamp.

**Depends on:** Phase 3.

**Exit criteria:** comparison report complete; final model explicitly selected using measured results; selected-model output passes schema and non-negative/chronological plausibility checks; rerunning the experiment reproduces the recorded result within documented expectations.

### Phase 5 — Inventory inputs and inventory-analysis rules

**Purpose:** safely connect demand forecasts with the inventory data required by the specification.

**Work:**

- Define and validate current-inventory and supplier-lead-time input records, including their source/provenance.
- If values are simulated or derived because M5 lacks them, document the generation methodology, assumptions, scope, and clear simulated/derived label.
- Define demand-variability calculation from historical or forecast-related demand without leaking future actuals.
- Implement stockout and overstock classification rules only after inputs and their units/granularity are validated.
- Evaluate safety stock, reorder point, and EOQ individually against available inputs. Implement only concepts whose data requirements and meaning are satisfied.
- Produce inventory status, alerts, recommended reorder timing, and recommended additional quantity.

**Depends on:** Phase 4 selected forecast contract and Phase 1 availability/provenance findings.

**Exit criteria:** sample scenarios covering adequate stock, stockout risk, and overstock are validated by hand; every output can trace to a forecast and inventory/lead-time input; unsupported inventory formulas are documented as excluded rather than guessed.

### Phase 6 — PostgreSQL schema and Django domain foundation

**Purpose:** establish the operational data model that can persist and query the validated pipeline outputs.

**Work:**

- Initialize the Django project and PostgreSQL connection configuration when implementation begins.
- Implement migrations for the approved domain entities below, using actual M5 identifiers/granularity confirmed in Phase 1.
- Enforce foreign keys, uniqueness, suitable indexes for dashboard query paths, timestamps, and provenance fields.
- Define the import boundary from the ML/inventory pipeline: validate and persist only outputs that match the Phase 4 and 5 contracts.
- Seed only controlled development data where required, marking it derived/simulated when applicable.

**Depends on:** Phases 1, 4, and 5. Schema should not be finalized before the clean-data and output contracts exist.

**Exit criteria:** migrations apply to a fresh PostgreSQL database; constraint and relation tests pass; a representative forecast, inventory input, alert, and recommendation can be stored and retrieved without ambiguity.

### Phase 7 — REST API

**Purpose:** expose persisted dashboard data through a stable, testable backend interface.

**Work:**

- Establish a versioned API namespace (for example, `/api/v1/`) and consistent response/error conventions.
- Implement read endpoints for the dashboard resources below, with filters/pagination only where the actual UI query needs established during frontend integration require them.
- Implement validated write/import operations only for data-management workflows actually chosen for inventory and supplier inputs; do not invent user-management or approval workflows.
- Provide endpoint-level validation, database transaction behavior for multi-record imports, and API documentation/examples.
- Ensure responses distinguish observed M5-derived values from simulated/derived operational inputs where relevant.

**Depends on:** Phase 6. It may begin after a representative persisted data slice exists; it must not bypass the database with ad hoc ML output.

**Exit criteria:** API tests cover success, invalid input, absent resource, filtering/query validation where implemented, and provenance representation; frontend can consume a stable documented response from a populated development database.

### Phase 8 — React dashboard

**Purpose:** deliver the specified web-based presentation of results.

**Work:**

- Initialize React with Vite and Tailwind CSS.
- Build dashboard views from the Phase 7 API contract, beginning with a small end-to-end slice rather than all visualizations at once.
- Present future demand forecasts, inventory status, stockout/overstock alerts, reorder recommendations, and recommended additional quantity.
- Add charts only for data and interactions supported by implemented API responses; include clear units, period, and provenance indicators where data is simulated/derived.
- Handle API loading, empty, and error states.

**Depends on:** Phase 7's stable API and representative data. Styling may start earlier with static layout work, but API integration waits for the contract.

**Exit criteria:** dashboard displays correct API data for a representative product/store scope supported by the dataset; UI states and chart values are verified against backend responses; all required outputs are accessible in the web UI.

### Phase 9 — End-to-end verification and deployment readiness

**Purpose:** validate the integrated core system and create a deployment plan only after its components work together.

**Work:**

- Run an end-to-end flow: validated data -> features -> selected model -> forecast -> inventory analysis -> database -> API -> dashboard.
- Verify provenance and reproducibility records across the flow.
- Document environment variables, PostgreSQL migration/backup expectations, static frontend build/hosting approach, Django production configuration requirements, and ML artifact/data storage requirements.
- Decide the concrete deployment platform only when project constraints are supplied; the specification does not name one.
- Prepare deployment checks, rollback approach, operational logs, and health checks appropriate to the chosen platform.

**Depends on:** Phases 4–8.

**Exit criteria:** one documented repeatable end-to-end run succeeds; deployment checklist is complete for the chosen platform; test suite passes in the target-like environment.

## 4. PostgreSQL schema plan

The exact columns must be confirmed against Phase 1's observed M5 fields. The proposed relational responsibilities are:

| Entity | Purpose | Key relationships / constraints |
|---|---|---|
| Product | Product context available from M5 | Stable source identifier; links to sales, prices, inventory, forecasts |
| Store | Store context available from M5 | Stable source identifier; links to sales, prices, inventory, forecasts |
| Calendar date | Calendar/seasonal context used by forecasting | Unique date/source key; supports holiday/date attributes actually present |
| Historical demand | Cleaned historical sales at the confirmed product/store/date granularity | Unique natural grain; source/provenance and ingestion metadata |
| Selling price | Product/store price over the confirmed M5 time grain | Unique natural grain; no invented discount field if unavailable |
| Inventory record | Current inventory input | Product/store scope matching forecast scope; observed/derived/simulated provenance |
| Supplier lead time | Lead-time input | Scope matching inventory/forecast; observed/derived/simulated provenance |
| Forecast run / model experiment | Reproducibility record for a model execution | References model/version, feature version, horizon, data split/period, metrics |
| Demand forecast | Predicted demand by forecast run and forecast period | Unique forecast grain; references product/store and run |
| Inventory analysis | Computed status and demand-variability inputs/result | References relevant forecast and inventory/lead-time records |
| Alert | Stockout or overstock result | References inventory analysis; status/lifecycle only if later required by approved UI/API behavior |
| Reorder recommendation | Recommended timing and additional quantity | References inventory analysis and any inventory method used |

The schema should avoid storing untraceable calculations. Computed output records must identify the forecast run and input versions used. EOQ-specific fields should be added only if Phase 5 shows the necessary data exists.

## 5. API design plan

The API should be RESTful and versioned. Resource groups, not final URLs, are planned now:

- **Catalog/context:** products, stores, and supported calendar/context metadata.
- **Historical/forecast data:** historical demand views where required by the dashboard; forecast runs/metadata; forecasts for a selected supported product/store/period scope.
- **Inventory:** current inventory and supplier lead-time inputs, with provenance.
- **Decision outputs:** inventory analyses, stockout/overstock alerts, and reorder recommendations.

Each response should expose the scope/granularity, relevant date or forecast period, units when defined during implementation, and provenance for non-observed inputs. API details that are not specified—authentication, roles, user accounts, CRUD permissions, alert acknowledgement, and asynchronous training triggers—must remain out of scope until requirements are supplied.

## 6. Testing strategy

Testing proceeds with each phase; no later phase substitutes for earlier data/ML validation.

| Layer | Tests required before advancing |
|---|---|
| Raw and clean data | Schema, required identifiers, chronology, duplicates, missing-value treatment, source-to-output reconciliation |
| Features and evaluation | Lag/rolling boundary tests, no future-data leakage, chronological split verification, metric calculation checks |
| Models | Reproducible training configuration, forecast output contract, MAE/RMSE/MAPE comparison integrity, plausibility review |
| Inventory logic | Hand-calculated scenarios for adequate, stockout-risk, and overstock states; quantity/timing traceability; simulation provenance |
| Django/database | Migration-on-empty-database test, model constraints, relation integrity, import validation, query/index review |
| API | Contract tests, validation/error cases, database integration, stable response/provenance fields |
| React | Component/view tests for loading, empty, error, and populated states; chart/dashboard values checked against API fixtures |
| System | End-to-end test with a controlled representative slice and a manual acceptance check against known results |

## 7. Implementation sequence and major milestones

1. **Milestone A — Data understood:** Phases 0–2 complete. The M5 input contract and clean dataset are validated.
2. **Milestone B — Forecasting proven:** Phases 3–4 complete. Leakage-safe comparison selects the model using MAE, RMSE, and MAPE.
3. **Milestone C — Inventory recommendations verified:** Phase 5 complete. Forecasts are combined with documented inventory/lead-time inputs and validated decision rules.
4. **Milestone D — Backend operational:** Phases 6–7 complete. PostgreSQL persistence and a tested API expose the core result data.
5. **Milestone E — Dashboard complete:** Phase 8 complete. The web UI displays all required project outputs.
6. **Milestone F — Ready to deploy:** Phase 9 complete. A repeatable integrated run and deployment checklist are verified.

## 8. Key dependencies and risks

| Dependency or risk | Impact | Planned control |
|---|---|---|
| M5 may not contain current inventory or supplier lead time | Inventory optimization cannot be represented as observed operational data | Confirm absence in Phase 1; document all separately supplied, derived, or simulated values and their method |
| Forecast horizon is undecided | Splits, labels, model output, API, and dashboard periods depend on it | Finalize once the data contract and evaluation design are understood in Phase 3 |
| Time-series leakage | Invalidly optimistic model selection | Chronological partitions and feature-availability tests are exit criteria before modeling |
| Candidate models behave differently by data scale/granularity | A complex model may not be appropriate | Compare applicable baseline candidates under one protocol; select from metrics, not expectation |
| MAPE may be unstable when actual demand is zero | Model comparison can be misleading | Detect and document the observed zero-demand behavior; define consistent handling before comparison |
| Inventory formulas require inputs not present in scope | Incorrect recommendations | Evaluate safety stock/reorder point/EOQ only after their required data is verified |
| Unclear data granularity | Broken joins and misleading dashboard aggregation | Establish and enforce the natural grain in Phase 1 and database constraints in Phase 6 |
| Premature frontend/backend coupling | Rework and fragile contracts | Stabilize forecast and database/API contracts before integration |
| Deployment target is unspecified | No platform-specific production design can be finalized | Keep Phase 9 platform-neutral until a target is selected |

## 9. Exact next step — do not implement yet

**Implement Phase 0, beginning with a repository foundation and reproducibility/documentation convention, then proceed to the Phase 1 M5 dataset inventory and data dictionary.**

The first concrete deliverable should be a documented M5 data inventory and schema/data-quality profile. This is first because it determines the actual forecasting grain, feasible features, missing inventory/supplier fields, and therefore every later ML, database, API, and dashboard contract.
