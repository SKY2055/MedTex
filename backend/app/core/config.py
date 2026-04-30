from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "MedTex API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # The specific Med7 model we are using
    MODEL_NAME: str = "en_core_med7_lg"
    
    # Environment configuration
    ENABLE_HEAVY_MODELS: bool = False
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = True
    
    class Config:
        env_file = ".env"
        extra = "ignore"  # Allow extra fields in .env without validation errors

settings = Settings()