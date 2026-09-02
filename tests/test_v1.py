from app.config import settings

def test_health_check(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK", "model_loaded": True}

def test_predict_success(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert data["status"] == "success"

def test_predict_validation_error(client):
    payload = {
        "sepal_length": 5.1
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422

def test_predict_batch_success(client):
    payload = {
        "inputs": [
            {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
            {"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3}
        ]
    }
    response = client.post("/api/v1/predict-batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["batch_size"] == 2
    assert len(data["predictions"]) == 2

def test_predict_batch_exceeds_limit(client):
    oversized_batch = [
        {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    ] * (settings.MAX_BATCH_SIZE + 1)

    payload = {"inputs": oversized_batch}
    response = client.post("/api/v1/predict-batch", json=payload)
    assert response.status_code == 400

def test_model_info(client):
    response = client.get("/api/v1/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "model_version" in data
    assert "features" in data
def test_v1_and_v2_schema_isolation(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    
    res_v1 = client.post("/api/v1/predict", json=payload)
    res_v2 = client.post("/api/v2/predict", json=payload)

    assert res_v1.status_code == 200
    assert res_v2.status_code == 200

    data_v1 = res_v1.json()
    data_v2 = res_v2.json()

    # V1 checks
    assert "confidence" in data_v1
    assert "class_probabilities" not in data_v1

    # V2 checks
    assert "class_probabilities" in data_v2
    assert "predicted_class_name" in data_v2
    assert "confidence" not in data_v2