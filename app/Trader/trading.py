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


def dummy_predict_signal(history: Any, observation: dict[str, Any]) -> float:
    """Placeholder predictor that returns the latest observed price."""
    current_price = float(observation.get("latest_price", 0))
    if not math.isfinite(current_price) or current_price <= 0:
        raise ValueError("invalid_current_price")
    return 1 #returning 1 to trigger buy


def run_trading_pipeline(
    history: Any,
    observation: dict[str, Any],
    predict_signal: Callable[[Any, dict[str, Any]], float],
    trading_client: Any,
    symbol: str,
    quantity: float = 1,
) -> dict[str, Any]:
    """
    Evaluates trading signals (+1 UP, -1 DOWN, 0 NEUTRAL or predicted prices) 
    against current portfolio holdings and executes appropriate position management.
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

    # -------------------------------------------------------------------
    # 1. Fetch Model Prediction & Normalize Signal (+1, -1, 0)
    # -------------------------------------------------------------------
    try:
        raw_pred = float(predict_signal(history, observation))
    except Exception as error:
        return {"action": "hold", "symbol": symbol, "reason": "prediction_failed", "detail": str(error)}

    if not math.isfinite(raw_pred):
        return {"action": "hold", "symbol": symbol, "reason": "invalid_prediction"}

    # Signal Normalization (Handles +1/-1/0 signals, return %, or raw price)
    # NOTE: raw_pred == current_price (exact "no move" prediction) is treated as
    # NEUTRAL, not UP -- it used to fall through to the `raw_pred > 0` branch.
    if raw_pred == current_price:
        signal = 0
        predicted_price = current_price
    elif raw_pred > current_price: # Absolute price mode (UP)
        signal = 1
        predicted_price = raw_pred
    elif raw_pred < current_price and raw_pred > 2.0: # Absolute price mode (DOWN)
        # CAVEAT: this threshold is a heuristic, not a guarantee. It will
        # misread an absolute-price prediction below $2 as a directional/%
        # signal, and can misread a large % return (e.g. raw_pred=2.5 meaning
        # +250%) as an absolute-price DOWN call. Only a real fix is to make
        # predict_signal self-describing (mode + value) instead of a bare float.
        signal = -1
        predicted_price = raw_pred
    elif raw_pred > 0: # Directional +1 signal or positive % return
        signal = 1
        predicted_price = current_price * (1.0 + raw_pred if raw_pred < 1.0 else 1.01)
    elif raw_pred < 0: # Directional -1 signal or negative % return
        signal = -1
        predicted_price = current_price * (1.0 + raw_pred if raw_pred > -1.0 else 0.99)
    else:
        signal = 0
        predicted_price = current_price

    if signal == 0:
        return {
            "action": "hold", 
            "symbol": symbol, 
            "current_price": current_price, 
            "predicted_price": predicted_price, 
            "reason": "neutral_signal"
        }

    # -------------------------------------------------------------------
    # 2. Query Account & Portfolio Positions
    # -------------------------------------------------------------------
    account = trading_client.get_account()
    if bool(account.trading_blocked):
        return {"action": "hold", "symbol": symbol, "reason": "account_trading_blocked"}

    positions = trading_client.get_all_positions()
    position = next((p for p in positions if p.symbol.upper() == symbol), None)

    # -------------------------------------------------------------------
    # 3. Decision Logic (Position-Aware)
    # -------------------------------------------------------------------
    
    # CASE A: PREDICTED UP (+1)
    if signal > 0:
        if position is not None:
            try:
                avg_entry = float(position.avg_entry_price)
            except (TypeError, ValueError):
                return {"action": "hold", "symbol": symbol, "reason": "invalid_position_data"}
            return {
                "action": "hold",
                "symbol": symbol,
                "current_price": current_price,
                "predicted_price": predicted_price,
                "avg_entry_price": avg_entry,
                "reason": "holding_existing_position_on_up_signal",
            }
        
        # If no position exists, enter a NEW long position
        try:
            buying_power = float(account.buying_power)
        except (TypeError, ValueError):
            return {"action": "hold", "symbol": symbol, "reason": "invalid_account_data"}

        required_funds = current_price * quantity
        if buying_power < required_funds:
            return {
                "action": "hold",
                "symbol": symbol,
                "reason": "insufficient_buying_power",
                "required_funds": required_funds,
                "buying_power": buying_power,
            }
        
        result = execute_order(trading_client, symbol, "buy", quantity)
        return {
            **result,
            "current_price": current_price,
            "predicted_price": predicted_price,
            "reason": "predicted_up_opened_new_position",
        }

    # CASE B: PREDICTED DOWN (-1)
    if signal < 0:
        if position is None:
            return {
                "action": "hold",
                "symbol": symbol,
                "current_price": current_price,
                "predicted_price": predicted_price,
                "reason": "no_position_to_sell",
            }
        
        try:
            sell_quantity = float(position.qty)
            avg_entry = float(position.avg_entry_price)
        except (TypeError, ValueError):
            return {"action": "hold", "symbol": symbol, "reason": "invalid_position_data"}

        if sell_quantity <= 0:
            return {"action": "hold", "symbol": symbol, "reason": "no_sellable_quantity"}

        # Close position when model predicts down trend
        result = execute_order(trading_client, symbol, "sell", sell_quantity)
        return {
            **result,
            "current_price": current_price,
            "predicted_price": predicted_price,
            "avg_entry_price": avg_entry,
            "reason": "predicted_down_closed_position",
        }
                
"""
#calculating the direction

import torch
import torch.nn as nn

# Calculate cumulative future return direction (+1 or -1) from y_future batch [32, 15, 4]
def get_future_direction(y_future_batch, threshold=0.0005):
    # Log returns are at feature index 0 across all 15 future steps
    future_log_returns = y_future_batch[:, :, 0] # Shape: [32, 15]
    
    # Sum log returns over the 15-minute horizon
    cum_returns = torch.sum(future_log_returns, dim=1) # Shape: [32]
    
    # Classify direction: +1 (Up), -1 (Down), 0 (Flat)
    direction = torch.zeros_like(cum_returns)
    direction[cum_returns > threshold] = 1.0   # + Signal
    direction[cum_returns < -threshold] = -1.0 # - Signal
    
    return direction

# Example Usage
# y_future_batch shape: [32, 15, 4]
target_signals = get_future_direction(y_future_batch)
"""