from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    API_TITLE: str = "Iris Classification API"
    API_VERSION: str = "v1.0"
    LOG_LEVEL: str = "INFO"
    MODEL_PATH: str = "ml/saved_model/model.pkl"
    MODEL_TYPE: str = "random_forest"
    MAX_BATCH_SIZE: int = 100
    
    # Task 17 Security Settings
    API_KEY: str = "my_super_secret_api_key_123"
    ALLOWED_ORIGINS: list[str] = ["http://localhost:8000", "http://127.0.0.1:8000"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()