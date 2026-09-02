from fastapi import APIRouter, Request
from app.models.schemas import IrisInput, PredictionOutputV2
from app.config import settings

router = APIRouter(prefix="/v2", tags=["v2 Endpoints"])

TARGET_CLASSES = ["setosa", "versicolor", "virginica"]

@router.post("/predict", response_model=PredictionOutputV2)
def predict_v2(request: Request, input_data: IrisInput):
    request_id = getattr(request.state, "request_id", "N/A")
    model = getattr(request.app.state, "model", None)

    features = [[
        input_data.sepal_length,
        input_data.sepal_width,
        input_data.petal_length,
        input_data.petal_width
    ]]
    
    if model:
        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]
    else:
        prediction = 0
        probabilities = [0.95, 0.03, 0.02]

    class_probs = {
        TARGET_CLASSES[i]: round(float(probabilities[i]), 4)
        for i in range(len(TARGET_CLASSES))
    }

    return {
        "request_id": request_id,
        "prediction": prediction,
        "predicted_class_name": TARGET_CLASSES[prediction],
        "class_probabilities": class_probs,
        "model_version": settings.API_VERSION,
        "status": "success"
    }