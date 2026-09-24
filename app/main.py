from typing import Any

from fastapi import FastAPI, Depends
from pydantic import BaseModel, Field
from app.config import Settings, get_settings
from app.Trader.history import get_market_observation
from app.Trader.portfolio import get_trading_data
from app.Trader.auth import get_data_client_key, get_trading_client_key
from app.Trader.trading import run_trading_pipeline, dummy_predict_signal

data_client = get_data_client_key()
trading_client = get_trading_client_key()

app = FastAPI()


class TradingPipelineRequest(BaseModel):
    symbol: str = Field(default="AAPL", min_length=1)
    quantity: float = Field(default=1, gt=0)
    lookback_minutes: int = Field(default=60, ge=10)
    history: Any = None

@app.get("/")
def health():
    return {"message":"backend is working"}

@app.get("/test_env")
def test_env(env: Settings = Depends(get_settings)):
    return {"history": get_trading_data(trading_client=trading_client)}


@app.post("/run_trading_pipeline")
def run_pipeline(request: TradingPipelineRequest):
    observation = get_market_observation(
        symbol=request.symbol,
        lookback_minutes=request.lookback_minutes,
        data_client=data_client,
    )

    return run_trading_pipeline(
        history=request.history,
        observation=observation,
        predict_signal=dummy_predict_signal,
        trading_client=trading_client,
        symbol=request.symbol,
        quantity=request.quantity,
    )