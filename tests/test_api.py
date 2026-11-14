"""Smoke tests de la API (se ejecutan en el CI)."""
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)

VALID_CAR = {
    "brand": "Maruti",
    "year": 2015,
    "km_driven": 50000,
    "fuel": "Petrol",
    "seller_type": "Individual",
    "transmission": "Manual",
    "owner": "First Owner",
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["model_loaded"] is True


def test_predict_ok():
    r = client.post("/predict", json=VALID_CAR)
    assert r.status_code == 200
    assert r.json()["predicted_price"] > 0


def test_predict_negative_km():
    r = client.post("/predict", json={**VALID_CAR, "km_driven": -10})
    assert r.status_code == 422


def test_predict_unknown_fuel():
    r = client.post("/predict", json={**VALID_CAR, "fuel": "Gasolina"})
    assert r.status_code == 422


def test_predict_missing_field():
    car = dict(VALID_CAR)
    del car["brand"]
    r = client.post("/predict", json=car)
    assert r.status_code == 422


def test_swagger_served():
    assert client.get("/docs").status_code == 200


def test_version():
    r = client.get("/version")
    assert r.status_code == 200
    assert r.json()["model"]["test_mae"] > 0
