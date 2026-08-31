import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import joblib
import numpy as np

from app.models.schemas import IrisInput, PredictionOutput
from app.logging_config import logger

ml_models = {}

class ModelInferenceError(Exception):
    def __init__(self, message: str):
        self.message = message

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("App starting: Loading Iris model into memory...")
    try:
        ml_models["iris_model"] = joblib.load("ml/saved_model/model.joblib")
        logger.info("SUCCESS: Iris model loaded successfully into memory!")
    except Exception as e:
        logger.error(f"ERROR: Model load error: {e}")
    
    yield
    
    logger.info("App stopping: Cleaning up resources...")
    ml_models.clear()

app = FastAPI(title="Iris Classification API", lifespan=lifespan)

# Middleware: Logs incoming request, execution time, and attaches unique request_id
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    start_time = time.time()
    response = await call_next(request)
    process_time = round((time.time() - start_time) * 1000, 2)  # Duration in ms
    
    logger.info(
        f"[{request_id}] Method={request.method} Path={request.url.path} "
        f"Status={response.status_code} Duration={process_time}ms"
    )
    return response

# Custom Exception Handler
@app.exception_handler(ModelInferenceError)
async def model_inference_exception_handler(request: Request, exc: ModelInferenceError):
    request_id = getattr(request.state, "request_id", "N/A")
    logger.error(f"[{request_id}] ModelInferenceError: {exc.message}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal prediction failure. Please try again later.",
            "detail": exc.message
        }
    )

@app.get("/")
def root():
    return {"message": "ML API is alive"}

@app.get("/health")
def health_check():
    is_loaded = "iris_model" in ml_models and ml_models["iris_model"] is not None
    return {
        "status": "ok",
        "model_loaded": is_loaded
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(request: Request, input_data: IrisInput):
    request_id = request.state.request_id
    model = ml_models.get("iris_model")
    
    if not model:
        logger.error(f"[{request_id}] Prediction failed: Model not loaded in memory")
        raise HTTPException(status_code=500, detail="Model is not loaded into memory")

    try:
        features = [
            input_data.sepal_length,
            input_data.sepal_width,
            input_data.petal_length,
            input_data.petal_width
        ]
        
        input_array = np.array(features).reshape(1, -1)
        
        prediction = model.predict(input_array)[0]
        probabilities = model.predict_proba(input_array)[0]
        confidence = float(np.max(probabilities))
        
        logger.info(f"[{request_id}] Prediction Successful: Result={prediction}, Confidence={confidence:.4f}")
        
        return PredictionOutput(
            request_id=request_id,
            prediction=int(prediction),
            confidence=round(confidence, 4),
            model_version="v1.0",
            status="success"
        )
    except Exception as e:
        logger.error(f"[{request_id}] Prediction Exception: {str(e)}")
        raise ModelInferenceError(message=f"Model failed to process inputs: {str(e)}")