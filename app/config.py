from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    ALPACA_API_KEY : str
    ALPACA_SECRET_KEY : str
    ALPACA_BASE_URL : str
    JEV_API_KEY : str 
    
    class Config: 
        env_file = ".env"
        
@lru_cache
def get_settings():
    return Settings()
