from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from app.config import get_settings

API_KEY, SECRET_KEY = get_settings().ALPACA_API_KEY, get_settings().ALPACA_SECRET_KEY

def get_data_client():
    # Market data
    data_client = StockHistoricalDataClient(
        API_KEY,
        SECRET_KEY
    )

    return data_client

def get_trading_client():
    
    # Paper trading account
    trading_client = TradingClient(
        API_KEY,
        SECRET_KEY,
        paper=True
    )
    
    return trading_client