import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.config import settings
from app.logging_config import logger
from app.routers import v1, v2


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("App starting: loading Iris model into memory...")
    try:
        app.state.model = joblib.load(settings.MODEL_PATH)
        logger.info(f"SUCCESS: Iris model loaded successfully from {settings.MODEL_PATH}")
    except Exception as e:
        logger.error(f"ERROR: Failed to load model from {settings.MODEL_PATH}: {e}")
        app.state.model = None
    yield
    logger.info("App shutting down...")


app = FastAPI(
    title="Iris Classification API",
    version=settings.API_VERSION,
    lifespan=lifespan,
)

app.include_router(v1.router, prefix="/api")
app.include_router(v2.router, prefix="/api")


@app.get("/")
def read_root():
    return {"message": "Welcome to Iris Classification API"}