from fastapi.testclient import TestClient
from app.main import app
from app.config import settings

client = TestClient(app)

# 1. Test missing API key -> Should return 401 Unauthorized
def test_missing_api_key():
    response = client.post(
        "/api/v1/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
    )
    assert response.status_code == 401


# 2. Test invalid API key -> Should return 401 Unauthorized
def test_invalid_api_key():
    response = client.post(
        "/api/v1/predict",
        headers={"X-API-Key": "invalid_secret_key"},
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
    )
    assert response.status_code == 401


# 3. Test extra unexpected payload field -> Should return 422 Unprocessable Entity
def test_extra_field_forbidden():
    response = client.post(
        "/api/v1/predict",
        headers={"X-API-Key": settings.API_KEY},
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
            "unexpected_field": "hacker_payload"
        }
    )
    assert response.status_code == 422    