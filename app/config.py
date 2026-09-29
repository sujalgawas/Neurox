from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    ALPACA_API_KEY : str
    ALPACA_SECRET_KEY : str
    ALPACA_BASE_URL : str
    JEV_API_KEY : str 
    JEV_API_URL: str | None = None
    JEV_MODEL: str = "jev-latest"
    huggingface_key : str | None = None
    jepa_checkpoint: str | None = None

    jepa_hf_repo: str = "sujalgawas/jepa-trading-model-direction"
    jepa_hf_filename: str = "model.pt"
    
    class Config: 
        env_file = ".env"
        
@lru_cache
def get_settings():
    return Settings()
