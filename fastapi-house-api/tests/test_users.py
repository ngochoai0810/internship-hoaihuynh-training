from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_create_user_filters_password_fields_from_response() -> None:
    response = client.post(
        "/api/v1/users/",
        json={
            "email": "a@test.com",
            "full_name": "Hoai Huynh",
            "password": "12345678",
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload == {
        "id": 2,
        "email": "a@test.com",
        "full_name": "Hoai Huynh",
        "is_active": True,
    }
    assert "password" not in payload
    assert "hashed_password" not in payload
