import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.config import settings

@pytest.fixture
def client():
    test_client = TestClient(app)
    # By default, the API Key Header is added to all requests
    test_client.headers = {"X-API-Key": settings.API_KEY}
    return test_client