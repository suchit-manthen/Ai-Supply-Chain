import pytest

import simulator


@pytest.fixture
def item():
    return {"id": "sku-a", "lead": 2, "stock": 10, "on_order": 0,
            "safety": 0, "max_cover": 14, "price": 5, "cost": 2}


@pytest.fixture
def forecast():
    return [10.0] * 28


def test_demand_multiplier_and_one_time_spike(item, forecast):
    result = simulator.simulate(forecast=forecast, item=item, horizon=30,
                                demand_multiplier=2, spike_pct=100, spike_day=3)
    rows = result["scenario"]["daily"]
    assert [row["demand"] for row in rows[:4]] == [20, 20, 40, 20]
    assert result["baseline"]["daily"][2]["demand"] == 10


def test_stock_is_consumed_without_negative_inventory_and_lost_demand_is_recorded(forecast):
    result = simulator.simulate(forecast=forecast, item={"lead": 2, "stock": 5}, horizon=30)
    first = result["scenario"]["daily"][0]
    assert first["fulfilled"] == 5
    assert first["lost"] == 5
    assert first["inventory"] == 0
    assert all(row["inventory"] >= 0 for row in result["scenario"]["daily"])


def test_supplier_delay_applies_to_order_arrival(item, forecast):
    result = simulator.simulate(forecast=forecast, item=item, horizon=30,
                                supplier_delay=3)
    baseline_order = result["baseline"]["orders"][0]
    delayed_order = result["scenario"]["orders"][0]
    assert baseline_order["arrival_day"] - baseline_order["placed_day"] == 2
    assert delayed_order["arrival_day"] - delayed_order["placed_day"] == 5
    assert result["scenario"]["summary"]["lost_units"] > result["baseline"]["summary"]["lost_units"]


def test_reorder_policy_places_positive_orders_and_accounts_for_pending(item, forecast):
    result = simulator.simulate(forecast=forecast, item=item, horizon=30)
    orders = result["scenario"]["orders"]
    assert orders
    assert all(order["units"] > 0 for order in orders)
    assert len({order["placed_day"] for order in orders}) == len(orders)
    first = result["scenario"]["daily"][0]
    assert first["purchase_order_units"] == orders[0]["units"]
    assert first["inventory_position"] >= first["inventory"]


def test_unmet_sales_do_not_create_duplicate_orders_while_replenishment_is_pending(forecast):
    result = simulator.simulate(forecast=forecast,
                                item={"lead": 2, "stock": 0, "on_order": 0},
                                horizon=30, supplier_delay=30)
    orders = result["scenario"]["orders"]
    assert len(orders) == 1
    assert orders[0]["arrival_day"] == orders[0]["placed_day"] + 32


def test_service_and_financial_metrics_are_consistent(forecast):
    result = simulator.simulate(forecast=forecast,
                                item={"lead": 2, "stock": 1000, "price": 5, "cost": 2},
                                horizon=30, annual_holding_rate=.2, order_cost=50)
    summary = result["scenario"]["summary"]
    assert summary["service_level"] == 100
    assert summary["lost_units"] == 0
    assert summary["lost_revenue"] == 0
    assert summary["purchase_orders"] == 0
    expected_holding = sum(row["inventory"] for row in result["scenario"]["daily"]) * 2 * .2 / 365
    assert summary["holding_cost"] == pytest.approx(expected_holding, abs=.01)
    assert summary["total_estimated_cost"] == pytest.approx(summary["holding_cost"])


def test_baseline_and_unchanged_scenario_match(item, forecast):
    result = simulator.simulate(forecast=forecast, item=item, horizon=30,
                                demand_multiplier=1, spike_pct=0,
                                supplier_delay=0, initial_stock_multiplier=1)
    assert result["baseline"]["summary"] == result["scenario"]["summary"]
    assert result["daily"][0]["baseline_inventory"] == result["daily"][0]["scenario_inventory"]
    assert result["daily"][0]["baseline_demand"] == result["daily"][0]["scenario_demand"]


def test_stock_multiplier_changes_scenario_initial_inventory(item, forecast):
    result = simulator.simulate(forecast=forecast, item=item, horizon=30,
                                initial_stock_multiplier=2)
    assert result["daily"][0]["scenario_inventory"] > result["daily"][0]["baseline_inventory"]


def test_forecast_extension_repeats_last_week_without_retraining(forecast):
    extended = simulator.extend_weekly(forecast, 30)
    assert len(extended) == 30
    assert extended[28:] == forecast[-7:-5]


@pytest.mark.parametrize("kwargs", [
    {"horizon": 45}, {"demand_multiplier": 3.1}, {"spike_pct": 201},
    {"spike_day": 31}, {"supplier_delay": 31},
    {"initial_stock_multiplier": 2.1},
])
def test_invalid_scenario_controls_are_rejected(item, forecast, kwargs):
    with pytest.raises(ValueError):
        simulator.simulate(forecast=forecast, item=item, **kwargs)
