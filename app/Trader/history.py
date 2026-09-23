import json
from datetime import datetime, timedelta, timezone

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.enums import DataFeed
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import numpy as np

def get_market_observation(
                            symbol: str = "AAPL", 
                            lookback_minutes: int = 60,
                            data_client: StockHistoricalDataClient = None
                        ) -> dict:
    
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=lookback_minutes)

    request = StockBarsRequest(
        symbol_or_symbols=symbol.upper(),
        timeframe=TimeFrame(1, TimeFrameUnit.Minute),
        start=start,
        end=end,
        feed=DataFeed.IEX,
    )

    bars = data_client.get_stock_bars(request)
    df = bars.df.reset_index()

    if df.empty or len(df) < 10:
        raise ValueError("Insufficient market data returned from provider.")

    # --- Feature Engineering (Stationary Features for JEPA) ---
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["bar_range"] = (df["high"] - df["low"]) / df["close"]
    df["vwap_dist"] = (df["close"] - df["vwap"]) / df["vwap"]
    df["log_vol_change"] = np.log(df["volume"] + 1) - np.log(df["volume"].shift(1) + 1)

    # Drop initial NaN rows created by shifting
    df_clean = df.dropna().copy()

    # Select the stationary features matrix for JEPA input
    feature_cols = ["log_return", "bar_range", "vwap_dist", "log_vol_change"]
    jepa_input_tensor = df_clean[feature_cols].values # Shape: [sequence_length, 4]

    return {
        "raw_records": json.loads(df.to_json(orient="records", date_format="iso")),
        "jepa_features": jepa_input_tensor.tolist(),
        "latest_price": float(df_clean["close"].iloc[-1]),
    }