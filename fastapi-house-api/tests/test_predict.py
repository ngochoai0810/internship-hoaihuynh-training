from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "gr_liv_area": 1500,
    "overall_qual": 7,
    "year_built": 2005,
    "total_bsmt_sf": 800,
    "garage_cars": 2,
}

def test_predict_success_returns_contract_shape() -> None:
    response = client.post("/api/v1/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"predicted_price", "currency"}
    assert isinstance(payload["predicted_price"], int | float)
    assert payload["predicted_price"] > 0
    assert payload["currency"] == "USD"


def test_predict_rejects_invalid_business_rule() -> None:
    response = client.post(
        "/api/v1/predict",
        json={**VALID_PAYLOAD, "gr_liv_area": 50000},
    )

    assert response.status_code == 400
    assert "gr_liv_area" in response.json()["detail"]


def test_predict_rejects_invalid_schema_type() -> None:
    response = client.post(
        "/api/v1/predict",
        json={**VALID_PAYLOAD, "gr_liv_area": "not-a-number"},
    )

    assert response.status_code == 422
