from locust import HttpUser, task, between

class IrisApiUser(HttpUser):
    wait_time = between(1, 2)
    headers = {"X-API-Key": "my_super_secret_api_key_123"}

    @task(3)
    def predict(self):
        payload = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        self.client.post("/api/v1/predict", json=payload, headers=self.headers)

    @task(1)
    def health_check(self):
        self.client.get("/api/v1/health", headers=self.headers)