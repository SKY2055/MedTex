from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "MedTex API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # The specific Med7 model we are using
    MODEL_NAME: str = "en_core_med7_lg"
    
    class Config:
        env_file = ".env"

settings = Settings()