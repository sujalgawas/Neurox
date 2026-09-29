from functools import lru_cache
from datetime import datetime, timedelta, timezone
import random
import time
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, Field
from app.config import Settings, get_settings
from app.Trader.history import get_market_observation
from app.Trader.portfolio import get_trading_data
from app.Trader.auth import get_data_client_key, get_trading_client_key
from app.Trader.trading import run_trading_pipeline_jepa
from app.services.JEPA import JEPAInference, download_checkpoint
from app.services.JEV import JEVInference

data_client = get_data_client_key()
trading_client = get_trading_client_key()

app = FastAPI()


class TradingPipelineRequest(BaseModel):
    symbol: str = Field(default="AAPL", min_length=1)
    quantity: float = Field(default=1, gt=0)
    lookback_minutes: int = Field(default=60, ge=10)
    history: Any = None


@lru_cache(maxsize=1)
def get_jepa_inference() -> JEPAInference:
    settings = get_settings()
    checkpoint_path = settings.jepa_checkpoint
    if not checkpoint_path:
        checkpoint_path = download_checkpoint(
            repo_id=settings.jepa_hf_repo,
            filename=settings.jepa_hf_filename,
            token=settings.huggingface_key,
        )
    return JEPAInference(checkpoint_path)


def get_dummy_market_observation() -> dict[str, Any]:
    """Create a deterministic 60-step feature window for local testing."""
    rng = np.random.default_rng(42)
    features = rng.normal(loc=0.0, scale=0.001, size=(60, 4)).astype(np.float32)
    features[:, 1] = np.abs(features[:, 1])
    return {
        "raw_records": [],
        "jepa_features": features.tolist(),
        "latest_price": 100.0,
    }


def get_dummy_jev_market_observation(
    symbol: str,
    lookback_minutes: int,
    data_client: Any,
) -> dict[str, Any]:
    """Fetch a random historical market window for local or off-hours testing."""
    market_timezone = ZoneInfo("America/New_York")
    historical_date = datetime.now(market_timezone) - timedelta(days=random.randint(1, 5))
    if historical_date.weekday() == 5:
        historical_date -= timedelta(days=1)
    elif historical_date.weekday() == 6:
        historical_date -= timedelta(days=2)

    historical_end = historical_date.replace(
        hour=15,
        minute=random.randint(0, 59),
        second=0,
        microsecond=0,
    )
    return get_market_observation(
        symbol=symbol,
        lookback_minutes=lookback_minutes,
        data_client=data_client,
        end=historical_end.astimezone(timezone.utc),
    )


@lru_cache(maxsize=1)
def get_jev_inference() -> JEVInference:
    settings = get_settings()
    if not settings.JEV_API_URL:
        raise ValueError("JEV_API_URL is not configured")
    return JEVInference(
        api_url=settings.JEV_API_URL,
        api_key=settings.JEV_API_KEY,
        model=settings.JEV_MODEL,
    )

@app.get("/")
def health():
    return {"message":"backend is working"}

@app.get("/test_env")
def test_env(env: Settings = Depends(get_settings)):
    return {"history": get_trading_data(trading_client=trading_client)}


@app.post("/jepa_observation")
def jepa_observation(request: TradingPipelineRequest):
    """Predict down, flat, or up from the latest JEPA context."""
    data_source = "alpaca"
    try:
        observation = get_market_observation(
            symbol=request.symbol,
            lookback_minutes=request.lookback_minutes + 1,
            data_client=data_client,
        )
    except Exception:
        # Temporary fallback while market-data access is unavailable locally.
        observation = get_dummy_market_observation()
        data_source = "dummy"
        print("we using dummy?")

    try:
        result = get_jepa_inference().encode_observation(observation)
        result["data_source"] = data_source
        return result
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/run_trading_pipeline_jepa")
def run_pipeline(request: TradingPipelineRequest):
    results = []

    for iteration in range(30):
        try:
            observation = get_market_observation(
                symbol=request.symbol,
                lookback_minutes=request.lookback_minutes,
                data_client=data_client,
            )
            result = run_trading_pipeline_jepa(
                history=request.history,
                observation=observation,
                predict_signal=get_jepa_inference().predict_signal,
                trading_client=trading_client,
                symbol=request.symbol,
                quantity=request.quantity,
            )
        except Exception as error:
            result = {"action": "error", "detail": str(error)}

        results.append({"iteration": iteration + 1, "result": result})

        if iteration < 9:
            time.sleep(50)

    return {"runs": results}


@app.post("/jev_observation")
def jev_observation(request: TradingPipelineRequest):
    data_source = "alpaca"
    try:
        observation = get_market_observation(
            symbol=request.symbol,
            lookback_minutes=min(request.lookback_minutes, 10),
            data_client=data_client,
        )
    except Exception:
        observation = get_dummy_jev_market_observation(
            symbol=request.symbol,
            lookback_minutes=60,
            data_client=data_client,
        )
        data_source = "alpaca_historical"
    result = get_jev_inference().predict_direction(observation)
    result["data_source"] = data_source
    return result


@app.post("/run_pipeline_jev")
def run_pipeline_jev(request: TradingPipelineRequest):
    results = []
    iterations = 1
    for iteration in range(iterations):
        try:
            observation = get_market_observation(
                symbol=request.symbol,
                lookback_minutes=min(request.lookback_minutes, 10),
                data_client=data_client,
            )
        except Exception:
            observation = get_dummy_jev_market_observation(
                symbol=request.symbol,
                lookback_minutes=60,
                data_client=data_client,
            )

        prediction = {}

        def predict_jev_signal(history: Any, current_observation: dict[str, Any]) -> float:
            prediction.update(get_jev_inference().predict_direction(current_observation))
            return float(prediction["direction"])

        try:
            result = run_trading_pipeline_jepa(
                history=request.history,
                observation=observation,
                predict_signal=predict_jev_signal,
                trading_client=trading_client,
                symbol=request.symbol,
                quantity=request.quantity,
            )
        except Exception as error:
            result = {"action": "error", "detail": str(error)}

        if prediction:
            result["direction_label"] = prediction["direction_label"]
            result["confidence"] = prediction["confidence"]

        results.append({"iteration": iteration + 1, "result": result})

        if iteration < iterations - 1:
            time.sleep(60)

    return {"runs": results}