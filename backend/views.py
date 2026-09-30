"""Pure payload builders for the two dashboards.

Extracted from app.py so the training pipeline can produce the owner/analyst
JSON without importing the Flask app. `products` defaults to the synthetic
catalog but can be any list of SKU specs (e.g. discovered from an uploaded CSV).
These functions have no side effects and no Flask dependency.
"""
import numpy as np

from data import PRODUCTS
from config import (HORIZON, HOLDOUT, N_FOLDS, SELECTION_METRIC, SELECTION_FALLBACK_METRIC,
                    LOW_DEMAND_THRESHOLD, SERVICE_LEVEL, Z, ORDERING_COST, HOLDING_RATE)
import inventory
from models import MODEL_INFO, ALL_MODELS


def owner_view(res, hist, end, policy=None, products=None, data_source="synthetic"):
    products = products or PRODUCTS
    policy = policy or {}
    z = float(policy.get("z", Z))
    service_level = float(policy.get("service_level", SERVICE_LEVEL))
    ordering_cost = float(policy.get("ordering_cost", ORDERING_COST))
    holding_rate = float(policy.get("holding_rate", HOLDING_RATE))
    items, tot_hist, tot_fc = [], {}, {}
    for p in products:
        r = res[p["id"]]; L = int(p.get("lead", 3))
        price = float(p.get("price", 1.0)); cost = float(p.get("cost", round(price * 0.7, 2)))
        shelf = int(p.get("shelf", 30)); max_cover = int(p.get("max_cover", 14))
        fc = np.array([x["pred"] for x in r["forecast"]])
        fc_hi = np.array([x["upper_bound"] for x in r["forecast"]])
        scale = r["uncertainty"]["error_scale"]
        recent = hist[hist["product"] == p["id"]].sales.iloc[-14:].mean()
        if p.get("stock") is not None:
            stock = int(round(float(p["stock"])))
        else:
            stock = int(round(float(p.get("stock_days", 0)) * recent))
        pos = stock + int(p.get("on_order", 0))
        pz = inventory.service_z(float(p["service_level"])) if p.get("service_level") else z
        dem_L = inventory.lead_time_demand(fc, L); dem_L_hi = inventory.lead_time_upper(fc_hi, L)
        ss = inventory.safety_stock(scale, L, z=pz); rop = inventory.reorder_point(dem_L, ss)
        cover = inventory.days_of_cover(stock, fc)
        qty = inventory.order_quantity(fc, L, ss, pos)
        eoq_units = inventory.eoq(float(fc.mean()) * 365.0, ordering_cost, holding_rate * cost)
        excess = inventory.excess_units(stock, fc, max_cover)
        status = inventory.classify(stock, dem_L, pos, rop, cover, max_cover)
        risk = inventory.stockout_risk(stock, dem_L, dem_L_hi)
        near = inventory.near_reorder(pos, rop, dem_L_hi, ss)
        if status == "Critical": msg = f"Will run out in ~{cover} days, before a {L}-day delivery can arrive. Order now / request express delivery."
        elif status == "Reorder": msg = f"Stock is below the reorder point ({int(rop)}). Order {qty} units today."
        elif status == "Overstock": msg = f"~{excess} units more than the next {max_cover} days of demand. Skip ordering; consider markdown{' – short shelf life!' if shelf <= 5 else '.'}"
        elif risk == "Elevated": msg = f"Expected demand is covered, but if demand runs at the top of the forecast range (~{int(round(dem_L_hi))} units before delivery arrives) you could run short. Consider ordering soon."
        elif near: msg = f"Inventory is close to the reorder point ({int(rop)}). Plan the next order in about {max(0, int(cover - L))} days."
        else: msg = f"Covered for ~{cover} days. Next order in about {max(0, int(cover - L))} days."
        if status in ("Healthy", "Overstock"): qty = 0
        items.append(dict(id=p["id"], name=p.get("name", p["id"]), category=p.get("category", "Unknown"), model=r["selected_model"],
                          price=price, lead=L, shelf=shelf,
                          stock=stock, on_order=int(p.get("on_order", 0)), cost=cost, service_level=float(p.get("service_level") or service_level),
                          cover_days=cover, forecast_7=int(fc[:7].sum()), forecast_28=int(fc.sum()), demand_lead=int(dem_L),
                          demand_lead_upper=int(round(dem_L_hi)), stockout_risk=risk, near_reorder=near,
                          inventory_position=pos, observed_recent=int(round(recent)), eoq=eoq_units,
                          safety=int(ss), rop=int(rop), order_qty=qty, order_cost=round(qty * cost, 2), status=status, message=msg,
                          lost_revenue=round(max(0, dem_L - stock) * price, 2), excess=excess if status == "Overstock" else 0,
                          mape=r["metrics"]["mape"], series=r["history"][-35:] and
                          [dict(date=x["date"], actual=x["actual"]) for x in r["history"][-35:]] +
                          [dict(date=x["date"], pred=x["pred"], lo=x["lo"], hi=x["hi"], horizon=x["horizon"]) for x in r["forecast"]]))
        for x in r["history"][-35:]: tot_hist[x["date"]] = tot_hist.get(x["date"], 0) + x["actual"]
        for x in r["forecast"]: tot_fc[x["date"]] = tot_fc.get(x["date"], 0) + x["pred"]
    order = {"Critical": 0, "Reorder": 1, "Overstock": 2, "Healthy": 3}
    items.sort(key=lambda i: order[i["status"]])
    rev = sum(i["forecast_7"] * i["price"] for i in items)
    return dict(
        as_of=str(end.date()), horizon=HORIZON, data_source=data_source, skus=len(products),
        service_level=service_level, safety_stock_z=z,
        inventory_policy=dict(service_level=service_level, safety_stock_z=z, ordering_cost=ordering_cost, holding_rate=holding_rate),
        kpis=dict(critical=sum(i["status"] == "Critical" for i in items), reorder=sum(i["status"] == "Reorder" for i in items),
                  overstock=sum(i["status"] == "Overstock" for i in items), healthy=sum(i["status"] == "Healthy" for i in items),
                  order_units=sum(i["order_qty"] for i in items), order_cost=round(sum(i["order_cost"] for i in items)),
                  revenue_7=round(rev), revenue_at_risk=round(sum(i["lost_revenue"] for i in items)),
                  units_28=sum(i["forecast_28"] for i in items)),
        items=items,
        total=[dict(date=d, actual=v) for d, v in sorted(tot_hist.items())] + [dict(date=d, pred=round(v)) for d, v in sorted(tot_fc.items())])


def analyst_view(res, hist, end, hol, data_quality=None, products=None, data_source="synthetic", n_folds=None):
    products = products or PRODUCTS
    folds_count = n_folds or N_FOLDS
    algorithms = [dict(name=a, color=MODEL_INFO[a]["color"], aspect=MODEL_INFO[a]["aspect"], why=MODEL_INFO[a]["why"])
                  for a in ALL_MODELS]
    win_counts = {a: 0 for a in ALL_MODELS}
    acc = {a: dict(mape=[], rmse=[], naive=[]) for a in ALL_MODELS}
    names = {p["id"]: p.get("name", p["id"]) for p in products}
    skus = []
    for p in products:
        r = res[p["id"]]; ev = r["evaluation"]; sel = r["selection"]
        win_counts[sel["model"]] += 1
        comparison = []
        for a in ALL_MODELS:
            m = ev["models"][a]["mean"]; acc[a]["mape"].append(m["mape"])
            acc[a]["rmse"].append(m["rmse"]); acc[a]["naive"].append(m["naive_mape"])
            comparison.append(dict(model=a, color=MODEL_INFO[a]["color"], mae=m["mae"], rmse=m["rmse"],
                                   mape=m["mape"], naive_mape=m["naive_mape"], r2=m["r2"], selected=(a == sel["model"])))
        b = ev["baseline"]["mean"]
        comparison.append(dict(model="Naive lag-7", color="#c9d1c6", mae=b["mae"], rmse=b["rmse"],
                               mape=b["mape"], naive_mape=b["naive_mape"], r2=b["r2"], selected=False, baseline=True))
        folds = {a: [dict(fold=f["fold"], train_end=f["train_end"], test_start=f["test_start"],
                          test_end=f["test_end"], **f["metrics"]) for f in ev["models"][a]["folds"]] for a in ALL_MODELS}
        skus.append(dict(
            id=p["id"], name=names[p["id"]], category=p.get("category", "Unknown"), legacy_model=p.get("legacy_model"),
            selected_model=sel["model"], selected_color=MODEL_INFO[sel["model"]]["color"], reason=sel["reason"],
            metric_used=sel["metric"], selection_score=sel["score"], baseline_score=sel["baseline_score"],
            improvement=sel["improvement"], fallback_used=sel["fallback_used"], mean_demand=sel["mean_demand"],
            ranking=sel["ranking"], comparison=comparison, folds=folds,
            result=dict(metrics=r["metrics"], extras=r["extras"], uncertainty=r["uncertainty"],
                        interval_coverage=r.get("interval_coverage"), backtest=r["backtest"], history=r["history"], forecast=r["forecast"], residuals=r["residuals"])))
    leaderboard = [dict(name=a, color=MODEL_INFO[a]["color"], aspect=MODEL_INFO[a]["aspect"], wins=win_counts[a],
                        avg_mape=round(float(np.mean(acc[a]["mape"])), 2), avg_rmse=round(float(np.mean(acc[a]["rmse"])), 1),
                        avg_naive=round(float(np.mean(acc[a]["naive"])), 2)) for a in ALL_MODELS]
    return dict(selection=dict(metric=SELECTION_METRIC, fallback=SELECTION_FALLBACK_METRIC,
                               low_demand_threshold=LOW_DEMAND_THRESHOLD, n_folds=folds_count),
                algorithms=algorithms, leaderboard=leaderboard, win_counts=win_counts, skus=skus,
                data_quality=data_quality or {}, data_source=data_source, holdout=HOLDOUT, horizon=HORIZON,
                dataset=dict(rows=int(len(hist)), skus=len(products), start=str(hist.date.min().date()), end=str(end.date()),
                             data_source=data_source,
                             features=["day of week", "month / day-of-year", "weekend flag", "discount %", "shelf price", "temperature", "holiday lift"],
                             sample=hist.tail(8).assign(date=lambda d: d.date.dt.strftime("%Y-%m-%d")).round(2).to_dict("records"),
                             monthly=[dict(month=str(m), units=int(u)) for m, u in hist.groupby(hist.date.dt.to_period("M")).sales.sum().tail(24).items()]))
