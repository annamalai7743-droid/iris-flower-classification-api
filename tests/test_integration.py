import requests

BASE_URL = "http://127.0.0.1:8000"
API_KEY = "my_super_secret_api_key_123"
HEADERS = {"X-API-Key": API_KEY}

def test_health_integration():
    res = requests.get(f"{BASE_URL}/api/v1/health", headers=HEADERS)
    assert res.status_code == 200

def test_predict_integration():
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    res = requests.post(f"{BASE_URL}/api/v1/predict", json=payload, headers=HEADERS)
    assert res.status_code == 200
    assert "prediction" in res.json()

def test_metrics_integration():
    res = requests.get(f"{BASE_URL}/metrics")
    assert res.status_code == 200