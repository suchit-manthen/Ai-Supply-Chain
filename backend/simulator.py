"""Deterministic, isolated what-if inventory simulation using the saved forecast."""
from __future__ import annotations

import math

import inventory

HORIZONS = (30, 60, 90)


def extend_weekly(forecast, horizon):
    """Extend the persisted 28-day forecast with its final weekly pattern."""
    values = [max(0.0, float(x)) for x in forecast]
    if not values:
        raise ValueError("No saved daily forecast is available for this SKU.")
    weekly = values[-7:] if len(values) >= 7 else values
    while len(values) < horizon:
        values.append(weekly[(len(values) - len(forecast) - 1) % len(weekly)])
    return values[:horizon]


def _integer_demand(value):
    return max(0, int(math.floor(float(value) + 0.5)))


def _simulate_path(*, forecast, item, horizon, demand_multiplier, spike_pct,
                   spike_day, supplier_delay, initial_stock_multiplier,
                   unit_price, unit_cost, annual_holding_rate, order_cost):
    demand_plan = extend_weekly(forecast, horizon)
    base_lead = max(1, int(item.get("lead", 1)))
    lead = base_lead + supplier_delay
    safety_base = max(0.0, float(item.get("safety", 0) or 0))
    max_cover = max(1, int(item.get("max_cover", 14) or 14))
    stock = _integer_demand(max(0, float(item.get("stock", 0) or 0) * initial_stock_multiplier))
    initial_on_order = _integer_demand(max(0, float(item.get("on_order", 0) or 0)))
    pending = []
    orders = []
    if initial_on_order:
        initial_order = {"placed_day": 0, "arrival_day": max(1, lead),
                         "units": initial_on_order, "initial": True}
        pending.append(dict(initial_order))
        orders.append(initial_order)

    rows = []
    total_demand = total_fulfilled = total_lost = 0
    inventory_sum = holding_cost = overstock_unit_days = 0.0
    procurement_cost = 0.0

    regular_plan = [_integer_demand(value * demand_multiplier) for value in demand_plan]
    adjusted_plan = list(regular_plan)
    adjusted_plan[spike_day - 1] = _integer_demand(
        demand_plan[spike_day - 1] * demand_multiplier * (1 + spike_pct / 100))

    for day in range(1, horizon + 1):
        arrivals = sum(order["units"] for order in pending if order["arrival_day"] == day)
        pending = [order for order in pending if order["arrival_day"] != day]
        stock += arrivals

        demand = adjusted_plan[day - 1]
        fulfilled = min(stock, demand)
        lost = demand - fulfilled
        stock = max(0, stock - fulfilled)
        total_demand += demand
        total_fulfilled += fulfilled
        total_lost += lost

        # The same reorder-point and order-up-to policy applies to both paths.
        policy_forecast = adjusted_plan[day:]
        if len(policy_forecast) < lead + 7:
            tail = regular_plan[-7:] if len(regular_plan) >= 7 else regular_plan
            while len(policy_forecast) < lead + 7:
                policy_forecast.append(tail[(len(policy_forecast) - (horizon - day)) % len(tail)])
        lead_demand = inventory.lead_time_demand(policy_forecast, lead)
        safety = safety_base * demand_multiplier * math.sqrt(lead / base_lead)
        reorder_point = inventory.reorder_point(lead_demand, safety)
        position = stock + sum(order["units"] for order in pending)
        placed = 0
        if position <= reorder_point:
            placed = inventory.order_quantity(policy_forecast, lead, safety, position)
            if placed > 0:
                arrival_day = day + lead
                pending.append({"arrival_day": arrival_day, "units": placed, "initial": False})
                orders.append({"placed_day": day, "arrival_day": arrival_day, "units": placed})
                procurement_cost += placed * unit_cost

        overstock_forecast = adjusted_plan[day:day + max_cover]
        tail = regular_plan[-7:] if len(regular_plan) >= 7 else regular_plan
        while len(overstock_forecast) < max_cover:
            overstock_forecast.append(tail[(len(overstock_forecast) - (horizon - day)) % len(tail)])
        next_demand = sum(overstock_forecast)
        overstock = max(0.0, stock - next_demand)
        inventory_sum += stock
        holding_cost += stock * unit_cost * annual_holding_rate / 365
        overstock_unit_days += overstock
        rows.append({
            "day": day, "demand": demand, "fulfilled": fulfilled, "lost": lost,
            "inventory": stock, "inventory_position": stock + sum(o["units"] for o in pending),
            "arrivals": arrivals, "purchase_order_units": placed,
            "purchase_order_count": int(placed > 0), "reorder_point": round(reorder_point, 2),
            "stockout": lost > 0, "overstock_units": round(overstock, 2),
            "lost_revenue": round(lost * unit_price, 2),
            "holding_cost": round(stock * unit_cost * annual_holding_rate / 365, 2),
        })

    service_level = 100.0 * total_fulfilled / total_demand if total_demand else 100.0
    lost_revenue = total_lost * unit_price
    new_orders = [order for order in orders if not order.get("initial")]
    ordering_cost = len(new_orders) * order_cost
    total_cost = lost_revenue + holding_cost + procurement_cost + ordering_cost
    return {
        "daily": rows,
        "orders": orders,
        "summary": {
            "total_demand": total_demand, "fulfilled_units": total_fulfilled,
            "lost_units": total_lost, "lost_revenue": round(lost_revenue, 2),
            "stockout_days": sum(row["stockout"] for row in rows),
            "service_level": round(service_level, 2),
            "ending_inventory": stock,
            "average_inventory": round(inventory_sum / horizon, 2),
            "holding_cost": round(holding_cost, 2),
            "purchase_cost": round(procurement_cost, 2),
            "ordering_cost": round(ordering_cost, 2),
            "total_estimated_cost": round(total_cost, 2),
            "purchase_orders": len(new_orders),
            "overstock_unit_days": round(overstock_unit_days, 2),
        },
    }


def simulate(*, forecast, item, horizon=30, demand_multiplier=1.0, spike_pct=0,
             spike_day=1, supplier_delay=0, initial_stock_multiplier=1.0,
             unit_price=None, unit_cost=None, annual_holding_rate=0.20,
             order_cost=50.0):
    """Return baseline and scenario paths plus merged chart-ready daily rows."""
    if horizon not in HORIZONS:
        raise ValueError("Choose a simulation horizon of 30, 60, or 90 days.")
    if not 0.5 <= demand_multiplier <= 3.0:
        raise ValueError("Demand multiplier must be between 0.5x and 3.0x.")
    if not 0 <= spike_pct <= 200:
        raise ValueError("Demand spike must be between 0% and 200%.")
    if not 1 <= spike_day <= horizon:
        raise ValueError("Simulation day must be within the selected horizon.")
    if not 0 <= supplier_delay <= 30:
        raise ValueError("Supplier delay must be between 0 and 30 days.")
    if not 0.25 <= initial_stock_multiplier <= 2.0:
        raise ValueError("Initial stock multiplier must be between 0.25x and 2.0x.")
    price = float(unit_price if unit_price is not None else item.get("price", 1.0))
    cost = float(unit_cost if unit_cost is not None else item.get("cost", price * 0.7))
    if price <= 0 or cost < 0:
        raise ValueError("Unit price must be positive and unit cost cannot be negative.")
    if not 0 <= annual_holding_rate <= 1:
        raise ValueError("Annual holding rate must be between 0% and 100%.")
    if order_cost < 0:
        raise ValueError("Order cost cannot be negative.")

    # Baseline uses the same SKU and saved forecast with default conditions.
    baseline = _simulate_path(
        forecast=forecast, item=item, horizon=horizon, demand_multiplier=1.0,
        spike_pct=0, spike_day=spike_day, supplier_delay=0,
        initial_stock_multiplier=1.0, unit_price=price, unit_cost=cost,
        annual_holding_rate=annual_holding_rate, order_cost=order_cost)
    scenario = _simulate_path(
        forecast=forecast, item=item, horizon=horizon,
        demand_multiplier=demand_multiplier, spike_pct=spike_pct,
        spike_day=spike_day, supplier_delay=supplier_delay,
        initial_stock_multiplier=initial_stock_multiplier, unit_price=price,
        unit_cost=cost, annual_holding_rate=annual_holding_rate,
        order_cost=order_cost)
    daily = []
    cumulative_lost_revenue_baseline = cumulative_lost_revenue_scenario = 0.0
    cumulative_cost_baseline = cumulative_cost_scenario = 0.0
    for b, s in zip(baseline["daily"], scenario["daily"]):
        cumulative_lost_revenue_baseline += b["lost_revenue"]
        cumulative_lost_revenue_scenario += s["lost_revenue"]
        cumulative_cost_baseline += b["holding_cost"] + b["lost_revenue"] + b["purchase_order_units"] * cost + b["purchase_order_count"] * order_cost
        cumulative_cost_scenario += s["holding_cost"] + s["lost_revenue"] + s["purchase_order_units"] * cost + s["purchase_order_count"] * order_cost
        daily.append({
            "day": b["day"], "baseline_inventory": b["inventory"],
            "scenario_inventory": s["inventory"], "baseline_reorder_point": b["reorder_point"],
            "scenario_reorder_point": s["reorder_point"],
            "baseline_demand": b["demand"], "baseline_fulfilled": b["fulfilled"],
            "scenario_demand": s["demand"], "scenario_fulfilled": s["fulfilled"],
            "baseline_stockout": b["stockout"], "scenario_stockout": s["stockout"],
            "baseline_lost_revenue_cumulative": round(cumulative_lost_revenue_baseline, 2),
            "scenario_lost_revenue_cumulative": round(cumulative_lost_revenue_scenario, 2),
            "baseline_total_cost_cumulative": round(cumulative_cost_baseline, 2),
            "scenario_total_cost_cumulative": round(cumulative_cost_scenario, 2),
        })
    return {"baseline": baseline, "scenario": scenario, "daily": daily}
