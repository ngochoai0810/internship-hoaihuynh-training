from fastapi.testclient import TestClient


def test_list_houses_returns_database_payload(client: TestClient) -> None:
    response = client.get("/api/v1/houses/?limit=1")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["id"] == 1
    assert payload[0]["area"] > 0
    assert payload[0]["rooms"] > 0
    assert "price" in payload[0]


def test_create_house_returns_created_house(client: TestClient) -> None:
    response = client.post(
        "/api/v1/houses/",
        json={"area": 95.5, "rooms": 2, "location": "Thu Duc"},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["id"] > 2
    assert payload == {
        "id": 3,
        "area": 95.5,
        "rooms": 2,
        "location": "Thu Duc",
        "price": 0.0,
    }

    list_response = client.get("/api/v1/houses/")
    assert list_response.status_code == 200
    assert any(house["id"] == payload["id"] for house in list_response.json())


def test_list_houses_filters_by_query_params(client: TestClient) -> None:
    response = client.get("/api/v1/houses/?min_price=200000&max_rooms=3")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["id"] == 1
    assert payload[0]["price"] >= 200000
    assert payload[0]["rooms"] <= 3


def test_get_house_returns_row_by_id(client: TestClient) -> None:
    response = client.get("/api/v1/houses/2")

    assert response.status_code == 200
    assert response.json() == {
        "id": 2,
        "area": 85.0,
        "rooms": 2,
        "location": "District 7",
        "price": 180000.0,
    }


def test_get_house_returns_404_when_missing(client: TestClient) -> None:
    response = client.get("/api/v1/houses/999")

    assert response.status_code == 404
    assert response.json() == {"detail": "House not found"}


def test_update_house_only_changes_sent_fields(client: TestClient) -> None:
    response = client.patch("/api/v1/houses/1", json={"price": 260000.0})

    assert response.status_code == 200
    assert response.json() == {
        "id": 1,
        "area": 120.5,
        "rooms": 3,
        "location": "District 1",
        "price": 260000.0,
    }


def test_update_house_returns_404_when_missing(client: TestClient) -> None:
    response = client.patch("/api/v1/houses/999", json={"price": 260000.0})

    assert response.status_code == 404
    assert response.json() == {"detail": "House not found"}


def test_get_house_rejects_non_integer_path_param(client: TestClient) -> None:
    response = client.get("/api/v1/houses/abc")

    assert response.status_code == 422
    detail = response.json()["detail"][0]
    assert detail["loc"] == ["path", "house_id"]
    assert "int" in detail["type"]
