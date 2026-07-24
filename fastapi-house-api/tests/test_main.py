from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)

VALID_PREDICT_PAYLOAD = {
    "gr_liv_area": 1500,
    "overall_qual": 7,
    "year_built": 2005,
    "total_bsmt_sf": 800,
    "garage_cars": 2,
}


def test_health_check_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_cors_allows_streamlit_origin() -> None:
    response = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:8501"


def test_unversioned_house_aliases_work() -> None:
    list_response = client.get("/houses?limit=1")
    get_response = client.get("/houses/1")
    create_response = client.post(
        "/houses",
        json={"area": 95.5, "rooms": 2, "location": "Thu Duc"},
    )

    assert list_response.status_code == 200
    assert list_response.json()[0]["id"] == 1
    assert get_response.status_code == 200
    assert get_response.json()["id"] == 1
    assert create_response.status_code == 201
    assert create_response.json()["location"] == "Thu Duc"


def test_unversioned_predict_alias_works() -> None:
    response = client.post("/predict", json=VALID_PREDICT_PAYLOAD)

    assert response.status_code == 200
    assert set(response.json()) == {"predicted_price", "currency"}
