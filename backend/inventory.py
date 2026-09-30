"""Inventory optimization helpers.

Pure functions extracted from the Flask view layer so the reorder-point logic
can be unit-tested independently. Safety stock now uses the horizon-aware
forecast error scale (when available) instead of a single pooled RMSE.
"""
import math
from statistics import NormalDist

import numpy as np

from config import Z


def service_z(service_level):
    """Z value for a target cycle service level (e.g. 0.95 -> 1.645)."""
    return round(NormalDist().inv_cdf(service_level), 3)


def safety_stock(error_scale, lead, z=Z):
    """Safety stock over the lead time.

    `error_scale` is either a scalar error std or a per-horizon error scale
    (index h-1 = std of the h-step forecast error). Per-horizon errors are
    treated as independent, so the lead-time error std is sqrt(sum(scale_h^2)).
    The scalar form reproduces the classic z * sigma * sqrt(lead) formula.
    """
    lead = max(int(lead), 0)
    if np.ndim(error_scale) == 0:
        return z * float(error_scale) * math.sqrt(lead)
    scale = np.asarray(error_scale, dtype=float)[:lead]
    return z * math.sqrt(float(np.sum(scale ** 2)))


def reorder_point(demand_lead, safety):
    """Reorder point = expected demand during lead time + safety stock."""
    return demand_lead + safety


def lead_time_demand(forecast, lead):
    """Expected demand over the lead time (sum of the point forecast)."""
    fc = np.asarray(forecast, dtype=float)
    return float(np.sum(fc[:max(int(lead), 0)]))


def lead_time_upper(forecast_upper, lead):
    """Upper-bound demand over the lead time (sum of the upper prediction bounds)."""
    fc = np.asarray(forecast_upper, dtype=float)
    return float(np.sum(fc[:max(int(lead), 0)]))


def stockout_risk(stock, demand_lead, demand_lead_upper):
    """Plain-language stockout risk derived from expected vs upper lead-time demand.

    High     -> stock is below expected lead-time demand
    Elevated -> expected demand is covered, but the upper forecast range is not
    Low      -> stock covers even the upper forecast range
    """
    if stock < demand_lead:
        return "High"
    if stock < demand_lead_upper:
        return "Elevated"
    return "Low"


def near_reorder(position, rop, demand_lead_upper, safety):
    """True when stock is still above the reorder point but close to it.

    "Close" means the upside of lead-time demand would push the position into
    the safety-stock buffer: the buffer above the reorder point is smaller than
    one safety-stock margin. Uses only forecast bounds and safety stock.
    """
    return position >= rop and (position - demand_lead_upper) < safety


def days_of_cover(stock, forecast):
    """Days until cumulative forecast demand exceeds current stock."""
    fc = np.asarray(forecast, dtype=float)
    cum = np.cumsum(fc)
    if cum[-1] > stock:
        out_idx = int(np.argmax(cum > stock))
        prev = cum[out_idx - 1] if out_idx else 0
        return round(float(out_idx + (stock - prev) / max(fc[out_idx], 1)), 1)
    mean = float(fc.mean())
    if mean <= 0:
        return float(len(fc))  # no demand forecast: cover bounded by the horizon
    return round(float(stock / mean), 1)


def order_quantity(forecast, lead, safety, position, step=10):
    """Order enough to cover lead time + one week, net of stock already on hand/on order."""
    fc = np.asarray(forecast, dtype=float)
    target = float(fc[:lead + 7].sum()) + safety
    return max(0, int(math.ceil((target - position) / step) * step))


def eoq(annual_demand, ordering_cost, holding_cost_per_unit_year):
    """Economic Order Quantity (classic Wilson formula) - a reference, not a rule.

    EOQ = sqrt(2 * D * S / H), where D = annual demand, S = cost per order and
    H = annual holding cost per unit. Returns 0 when any input is non-positive.
    """
    if annual_demand <= 0 or ordering_cost <= 0 or holding_cost_per_unit_year <= 0:
        return 0
    return int(round(math.sqrt(2.0 * annual_demand * ordering_cost / holding_cost_per_unit_year)))


def excess_units(stock, forecast, max_cover):
    """Units above the next `max_cover` days of demand (overstock amount)."""
    fc = np.asarray(forecast, dtype=float)
    return max(0, int(stock - fc[:max_cover].sum()))


def classify(stock, demand_lead, position, rop, cover, max_cover):
    """Map a SKU's inventory state to one of four actions."""
    if stock < demand_lead:
        return "Critical"
    if position < rop:
        return "Reorder"
    if cover > max_cover:
        return "Overstock"
    return "Healthy"
