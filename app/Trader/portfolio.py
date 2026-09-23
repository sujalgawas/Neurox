from alpaca.trading.client import TradingClient

def get_trading_data(trading_client: TradingClient) -> dict:
    account = trading_client.get_account()
    
    return {
        "Cash": account.chash,
        "Equity": account.equity,
        "Buying_power" : account.buying_power,
        "Trading_block" : account.trading_block
    }