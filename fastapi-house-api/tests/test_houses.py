from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_list_houses_returns_mock_payload() -> None:
    response = client.get("/api/v1/houses/?limit=1")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["id"] == 1
    assert payload[0]["area"] > 0
    assert payload[0]["rooms"] > 0
    assert "price" in payload[0]


def test_create_house_returns_created_house() -> None:
    response = client.post(
        "/api/v1/houses/",
        json={"area": 95.5, "rooms": 2, "location": "Thu Duc"},
    )

    assert response.status_code == 201
    assert response.json() == {
        "id": 3,
        "area": 95.5,
        "rooms": 2,
        "location": "Thu Duc",
        "price": 0.0,
    }


def test_get_house_rejects_non_integer_path_param() -> None:
    response = client.get("/api/v1/houses/abc")

    assert response.status_code == 422
    detail = response.json()["detail"][0]
    assert detail["loc"] == ["path", "house_id"]
    assert "int" in detail["type"]
