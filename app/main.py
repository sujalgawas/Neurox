from fastapi import FastAPI, Depends
from app.config import Settings,get_settings
from app.Trader.history import get_market_observation
from app.Trader.auth import get_data_client

data_client = get_data_client()

app = FastAPI()

@app.get("/")
def health():
    return {"message":"backend is working"}

@app.get("/test_env")
def test_env(env: Settings = Depends(get_settings)):
    return {"history": get_market_observation(symbol="AAPL",lookback_minutes=60, data_client=data_client)}