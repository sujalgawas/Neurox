from pydantic import AliasChoices, Field
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
    jepa_checkpoint: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "JEPA_CHECKPOINT",
            "JEPA_DIRECTION_CHECKPOINT",
        ),
    )

    jepa_hf_repo: str = "sujalgawas/jepa-trading-direction-Big"
    jepa_hf_filename: str = "model.pt"
    jepa_hf_revision: str = "858439ff868c21a15d1766dac84f378c40081ba3"
    combined_signal_log_path: str = "data/combined/signals.jsonl"
    jepa_train_cutoff: str | None = None

    COMBINED_PERCENTILE_WINDOW: int = 500
    COMBINED_JEPA_WEIGHT: float = 0.5
    COMBINED_JEV_WEIGHT: float = 0.5
    COMBINED_ENTRY_THRESHOLD: float = 0.6
    COMBINED_SOLO_THRESHOLD: float = 0.85
    COMBINED_MAX_SIZE: float = 1.0
    COMBINED_WARMUP_PERIOD: int = 2
    
    class Config: 
        env_file = ".env"
        
@lru_cache
def get_settings():
    return Settings()
