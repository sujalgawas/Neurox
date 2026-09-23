"""Prototype decision and order functions for one stock symbol."""

import math
from typing import Any, Callable

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest


def execute_order(
    trading_client: TradingClient,
    symbol: str,
    side: str,
    quantity: float,
) -> dict[str, Any]:
    """Submit one market order. This function does not decide whether to trade."""
    action = side.strip().lower()
    if action not in {"buy", "sell"}:
        raise ValueError("side must be 'buy' or 'sell'")
    if quantity <= 0:
        raise ValueError("quantity must be greater than zero")

    order = trading_client.submit_order(
        order_data=MarketOrderRequest(
            symbol=symbol.upper(),
            qty=quantity,
            side=OrderSide.BUY if action == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
    )
    return {
        "action": action,
        "symbol": symbol.upper(),
        "quantity": quantity,
        "order_id": str(order.id),
        "status": str(order.status),
    }


def dummy_predict_next_price(history: Any, observation: dict[str, Any]) -> float:
    """Placeholder predictor that returns the latest observed price."""
    current_price = float(observation.get("latest_price", 0))
    if not math.isfinite(current_price) or current_price <= 0:
        raise ValueError("invalid_current_price")
    return current_price + 10


def run_trading_pipeline(
    history: Any,
    observation: dict[str, Any],
    predict_next_price: Callable[[Any, dict[str, Any]], float],
    trading_client: TradingClient,
    symbol: str,
    quantity: float = 1,
) -> dict[str, Any]:
    """Predict the next-minute price, check the portfolio, and decide what to do.

    The prediction callable receives (history, observation) and returns a price.
    A buy is sized by `quantity`; a sell closes the available long position.
    """
    symbol = symbol.strip().upper()
    current_price = observation.get("latest_price")
    if not symbol:
        return {"action": "hold", "reason": "missing_symbol"}
    if current_price is None:
        return {"action": "hold", "symbol": symbol, "reason": "invalid_current_price"}
    try:
        current_price = float(current_price)
    except (TypeError, ValueError):
        return {"action": "hold", "symbol": symbol, "reason": "invalid_current_price"}
    if not math.isfinite(current_price) or current_price <= 0:
        return {"action": "hold", "symbol": symbol, "reason": "invalid_current_price"}
    if quantity <= 0:
        return {"action": "hold", "symbol": symbol, "reason": "invalid_quantity"}

    try:
        predicted_price = float(predict_next_price(history, observation))
    except Exception as error:
        return {"action": "hold", "symbol": symbol, "reason": "prediction_failed", "detail": str(error)}
    if not math.isfinite(predicted_price) or predicted_price <= 0:
        return {"action": "hold", "symbol": symbol, "reason": "invalid_prediction"}

    if predicted_price == current_price:
        return {"action": "hold", "symbol": symbol, "current_price": current_price,
                "predicted_price": predicted_price, "reason": "no_predicted_change"}

    # Do not place orders if Alpaca reports the account is blocked from trading.
    account = trading_client.get_account()
    if bool(account.trading_blocked):
        return {"action": "hold", "symbol": symbol, "reason": "account_trading_blocked"}

    positions = trading_client.get_all_positions()
    position = next((p for p in positions if p.symbol.upper() == symbol), None)

    if predicted_price > current_price:
        required_funds = current_price * quantity
        buying_power = float(account.buying_power)
        if buying_power < required_funds:
            return {"action": "hold", "symbol": symbol, "reason": "insufficient_buying_power",
                    "required_funds": required_funds, "buying_power": buying_power}
        result = execute_order(trading_client, symbol, "buy", quantity)
        return {**result, "current_price": current_price, "predicted_price": predicted_price,
                "reason": "predicted_up"}

    # On a down prediction, only sell this symbol when its average entry price
    # is above the predicted price. Never send a sell order without a holding.
    if position is None:
        return {"action": "hold", "symbol": symbol, "current_price": current_price,
                "predicted_price": predicted_price, "reason": "no_position_to_sell"}
    if float(position.avg_entry_price) <= predicted_price:
        return {"action": "hold", "symbol": symbol, "current_price": current_price,
                "predicted_price": predicted_price, "reason": "entry_price_not_above_prediction"}

    sell_quantity = float(position.qty)
    if sell_quantity <= 0:
        return {"action": "hold", "symbol": symbol, "reason": "no_sellable_quantity"}
    result = execute_order(trading_client, symbol, "sell", sell_quantity)
    return {**result, "current_price": current_price, "predicted_price": predicted_price,
            "reason": "predicted_down_and_entry_above_prediction"}
