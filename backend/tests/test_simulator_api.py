import app as api


def test_simulator_catalog_and_run_use_saved_item_forecast(monkeypatch):
    item = {"id": "sku-a", "name": "Test SKU", "category": "Test",
            "forecast": [10.0] * 28, "price": 5.0, "cost": 2.0,
            "price_assumed": False, "cost_assumed": False,
            "stock": 50, "on_order": 0, "lead": 2, "safety": 0,
            "max_cover": 14}
    monkeypatch.setattr(api, "_simulator_items", lambda: ([item], "synthetic"))
    client = api.app.test_client()

    catalog = client.get("/api/simulator/catalog")
    assert catalog.status_code == 200
    assert catalog.get_json()["items"][0]["id"] == "sku-a"
    assert "forecast" not in catalog.get_json()["items"][0]

    response = client.post("/api/simulator/run", json={
        "sku": "sku-a", "horizon": 30, "demand_multiplier": 1.5,
        "spike_pct": 25, "spike_day": 10, "supplier_delay": 3,
        "initial_stock_multiplier": 1.25, "unit_price": 5, "unit_cost": 2,
    })
    payload = response.get_json()
    assert response.status_code == 200
    assert payload["ok"] is True
    assert len(payload["daily"]) == 30
    assert payload["daily"][9]["scenario_demand"] > payload["daily"][8]["scenario_demand"]
    assert payload["assumptions"]["forecast_note"]


def test_simulator_run_rejects_unknown_sku(monkeypatch):
    monkeypatch.setattr(api, "_simulator_items", lambda: ([], "synthetic"))
    response = api.app.test_client().post("/api/simulator/run", json={"sku": "missing"})
    assert response.status_code == 400
    assert response.get_json()["ok"] is False
