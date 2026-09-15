import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, status, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter  # Custom Metric Import

from app.config import settings
from app.logging_config import logger
from app.routers import v1, v2


# --- Task 18 Custom Metric Creation ---
PREDICTION_COUNTER = Counter(
    "iris_prediction_count_total",
    "Total number of Iris model predictions made",
    ["predicted_class"]
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("App starting: loading Iris model into memory...")
    try:
        app.state.model = joblib.load(settings.MODEL_PATH)
        # Prediction metric-ah router or services-la use panna app state-la store panrom
        app.state.prediction_counter = PREDICTION_COUNTER
        logger.info(f"SUCCESS: Iris model loaded successfully from {settings.MODEL_PATH}")
    except Exception as e:
        logger.error(f"ERROR: Failed to load model from {settings.MODEL_PATH}: {e}")
        app.state.model = None
        app.state.prediction_counter = None
    yield
    logger.info("App shutting down...")


app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    lifespan=lifespan,
)

# Instrument Prometheus metrics (/metrics endpoint created)
Instrumentator().instrument(app).expose(app)

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup API Key Security Dependency
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key == settings.API_KEY:
        return api_key
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key",
    )

# Include Routers
app.include_router(
    v1.router,
    prefix="/api",
    dependencies=[Depends(verify_api_key)]
)
app.include_router(
    v2.router,
    prefix="/api",
    dependencies=[Depends(verify_api_key)]
)


@app.get("/")
def read_root():
    return {"message": "Welcome to Iris Classification API"}