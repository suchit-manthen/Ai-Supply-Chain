import numpy as np
import pytest

from config import Z
import inventory


def test_z_matches_95_percent_service_level():
    assert round(Z, 3) == 1.645


def test_safety_stock_formula():
    # z * rmse * sqrt(lead)
    assert inventory.safety_stock(10.0, 4) == pytest.approx(1.645 * 10.0 * 2.0)


def test_reorder_point_is_sum():
    assert inventory.reorder_point(100, 15) == 115


def test_days_of_cover_interpolates_within_the_crossing_day():
    # cumulative demand 100, 200, 300; stock 150 -> crosses on day index 1
    assert inventory.days_of_cover(150, [100, 100, 100]) == pytest.approx(1.5)


def test_days_of_cover_when_stock_outlasts_horizon():
    assert inventory.days_of_cover(1000, [100, 100, 100]) == pytest.approx(10.0)


def test_days_of_cover_with_zero_stock():
    assert inventory.days_of_cover(0, [100, 100, 100]) == 0.0


def test_order_quantity_rounds_up_to_step_of_ten():
    # target = sum(first lead+7 = 12 days) = 120, position 0
    assert inventory.order_quantity([10] * 30, lead=5, safety=0, position=0) == 120


def test_order_quantity_never_negative():
    assert inventory.order_quantity([10] * 30, lead=5, safety=0, position=10000) == 0


def test_excess_units():
    assert inventory.excess_units(500, [10, 10, 10], 3) == 470


def test_classify_all_branches():
    assert inventory.classify(5, 10, 5, 20, 1, 10) == "Critical"
    assert inventory.classify(50, 10, 10, 20, 5, 10) == "Reorder"
    assert inventory.classify(500, 10, 500, 20, 50, 10) == "Overstock"
    assert inventory.classify(50, 10, 50, 20, 5, 10) == "Healthy"


def test_zero_inventory_is_critical():
    assert inventory.classify(0, 10, 0, 20, 0, 10) == "Critical"


# --- Phase 3: horizon-aware safety stock and risk helpers -------------------

def test_safety_stock_scalar_matches_classic_formula():
    assert inventory.safety_stock(10.0, 4) == pytest.approx(1.645 * 10.0 * 2.0)


def test_safety_stock_vector_uses_lead_time_errors():
    # per-horizon scale all ones over a 4-day lead -> sqrt(4) = 2
    assert inventory.safety_stock([1.0, 1.0, 1.0, 1.0], 4) == pytest.approx(1.645 * 2.0)


def test_safety_stock_vector_uses_only_the_lead_time_window():
    assert inventory.safety_stock([1.0, 1.0, 99.0, 99.0], 2) == pytest.approx(1.645 * np.sqrt(2.0))


def test_safety_stock_zero_lead_is_zero():
    assert inventory.safety_stock([5.0, 5.0, 5.0], 0) == 0.0


def test_lead_time_demand_sums_point_forecast():
    assert inventory.lead_time_demand([10, 20, 30, 40], 2) == 30.0
    assert inventory.lead_time_demand([10, 20, 30], 0) == 0.0


def test_lead_time_upper_sums_upper_bounds():
    assert inventory.lead_time_upper([15, 25, 35, 45], 2) == 40.0


def test_stockout_risk_levels():
    assert inventory.stockout_risk(5, 10, 20) == "High"       # below expected
    assert inventory.stockout_risk(15, 10, 20) == "Elevated"  # covers expected, not upper
    assert inventory.stockout_risk(25, 10, 20) == "Low"       # covers upper


def test_stockout_risk_low_demand_is_low():
    # stock covers even the upper bound -> Low; equal-to-expected but below upper is Elevated
    assert inventory.stockout_risk(2, 1, 2) == "Low"
    assert inventory.stockout_risk(1, 1, 2) == "Elevated"


def test_near_reorder_true_when_upside_demand_enters_buffer():
    assert inventory.near_reorder(position=100, rop=90, demand_lead_upper=95, safety=10) is True


def test_near_reorder_false_when_comfortable():
    assert inventory.near_reorder(position=200, rop=90, demand_lead_upper=95, safety=10) is False


def test_near_reorder_false_below_threshold():
    assert inventory.near_reorder(position=80, rop=90, demand_lead_upper=95, safety=10) is False


def test_zero_demand_edge_cases():
    assert inventory.lead_time_demand([0, 0, 0], 3) == 0.0
    assert inventory.safety_stock([0.0, 0.0, 0.0], 3) == 0.0
    assert inventory.days_of_cover(0, [0, 0, 0]) == 3.0
    assert np.isfinite(inventory.days_of_cover(100, [0, 0, 0]))


def test_intermittent_demand_days_of_cover():
    # sparse demand: stock 25, forecast 0,0,40,10 -> crosses on day index 2
    assert inventory.days_of_cover(25, [0, 0, 40, 10]) == pytest.approx(2.6)


def test_large_and_short_lead_times():
    fc = [10.0] * 28
    assert inventory.lead_time_demand(fc, 20) == 200.0
    assert inventory.lead_time_demand(fc, 1) == 10.0
    assert inventory.safety_stock([1.0] * 28, 20) == pytest.approx(1.645 * np.sqrt(20.0))


# --- Phase 4C: service-level z and EOQ reference ----------------------------

def test_service_z_matches_config():
    assert inventory.service_z(0.95) == pytest.approx(1.645)
    assert inventory.service_z(0.99) > inventory.service_z(0.95)


def test_eoq_classic_formula():
    # D=1000, S=50, H=2 -> sqrt(2*1000*50/2) = sqrt(50000) ~= 224
    assert inventory.eoq(1000, 50, 2) == 224


def test_eoq_zero_when_inputs_non_positive():
    assert inventory.eoq(0, 50, 2) == 0
    assert inventory.eoq(1000, 0, 2) == 0
    assert inventory.eoq(1000, 50, 0) == 0
