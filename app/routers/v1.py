import uuid
import numpy as np
import joblib
from fastapi import APIRouter, HTTPException, Request, status
from app.config import settings
from app.logging_config import logger
from app.models.schemas import (
    IrisInput,
    IrisBatchInput,
    PredictionOutput,
    IrisBatchOutput,
    ModelInfoOutput
)

router = APIRouter(prefix="/v1", tags=["V1 - Prediction"])

model = joblib.load(settings.MODEL_PATH)
features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET_CLASSES = ["setosa", "versicolor", "virginica"]

@router.get("/health")
def health_check():
    return {
        "status": "OK",
        "model_loaded": model is not None
    }

@router.post("/predict", response_model=PredictionOutput)
def predict(data: IrisInput):
    input_data = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])
    
    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)
    confidence = np.max(probabilities)
    
    return {
    "prediction": int(prediction[0]),
    "predicted_class_name": TARGET_CLASSES[prediction[0]],
    "confidence": float(confidence),
    "model_version": settings.API_VERSION,
    "status": "success"
}

@router.post("/predict-batch", response_model=IrisBatchOutput)
def predict_batch(request: Request, batch_data: IrisBatchInput):
    request_id = str(uuid.uuid4())
    batch_size = len(batch_data.inputs)
    
    if batch_size > settings.MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size exceeds maximum limit of {settings.MAX_BATCH_SIZE}"
        )
        
    input_list = [
        [item.sepal_length, item.sepal_width, item.petal_length, item.petal_width]
        for item in batch_data.inputs
    ]
    
    predictions = model.predict(input_list)
    probabilities = model.predict_proba(input_list)
    
    results = []
    for idx, input_item in enumerate(batch_data.inputs):
        pred_class = int(predictions[idx])
        conf = float(np.max(probabilities[idx]))
        results.append({
            "request_id": request_id,
            "prediction": pred_class,
            "predicted_class_name": TARGET_CLASSES[pred_class],
            "confidence": round(conf, 4),
            "model_version": settings.API_VERSION,
            "status": "success"
        })

    logger.info(f"[{request_id}] Successfully processed batch size of {batch_size}.")

    return {
        "predictions": results,
        "batch_size": batch_size,
        "status": "success"
    }

@router.get("/model-info", response_model=ModelInfoOutput)
def get_model_info():
    return {
        "model_version": settings.API_VERSION,
        "model_type": settings.MODEL_TYPE,
        "features": features,
        "target_classes": TARGET_CLASSES
    }