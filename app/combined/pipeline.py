"""Application service that adapts the existing model clients to the combiner."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import lru_cache
import logging
import random
from typing import Any
from zoneinfo import ZoneInfo

from app.config import get_settings
from app.Trader.auth import get_data_client_key
from app.Trader.auth import get_trading_client_key
from app.Trader.history import get_market_observation
from app.Trader.trading import execute_order
from app.services.JEPA import JEPAInference, download_checkpoint
from app.services.JEV import JEVInference

from .combiner import CombinedSignal, combine_predictions, histories_for_symbol, signal_config_from_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_jepa_inference() -> JEPAInference:
    """Load the existing JEPA client once."""
    settings = get_settings()
    checkpoint = settings.jepa_checkpoint or download_checkpoint(
        repo_id=settings.jepa_hf_repo,
        filename=settings.jepa_hf_filename,
        token=settings.huggingface_key,
        revision=settings.jepa_hf_revision,
    )
    return JEPAInference(checkpoint)


@lru_cache(maxsize=1)
def get_jev_inference() -> JEVInference:
    """Load the existing JEV client once."""
    settings = get_settings()
    if not settings.JEV_API_URL:
        raise ValueError("JEV_API_URL is not configured")
    return JEVInference(settings.JEV_API_URL, settings.JEV_API_KEY, settings.JEV_MODEL)


class CombinedPipeline:
    """Maintain per-symbol history and evaluate combined predictions."""

    # The feature builder drops the first bar and Alpaca may omit an
    # incomplete/current minute, so request a buffer beyond JEPA's 60 rows.
    _observation_lookback_minutes = 120

    def __init__(self):
        self.settings = get_settings()
        self.config = signal_config_from_settings(self.settings)
        self.histories: dict[str, dict[str, list[float]]] = {}
        self.latest: dict[str, dict[str, object]] = {}

    def _history(self, symbol: str) -> dict[str, list[float]]:
        return histories_for_symbol(self.histories, symbol)

    def _current_observation(self, symbol: str) -> dict[str, Any]:
        """Fetch the latest market bars for a real prediction."""
        return get_market_observation(
            symbol=symbol,
            lookback_minutes=self._observation_lookback_minutes,
            data_client=get_data_client_key(),
        )

    def _prior_day_observation(self, symbol: str) -> dict[str, Any]:
        """Fetch a prior weekday market window for a non-trading test."""
        data_client = get_data_client_key()
        market_timezone = ZoneInfo("America/New_York")
        historical_date = datetime.now(market_timezone) - timedelta(
            days=random.randint(1, 5)
        )
        while historical_date.weekday() >= 5:
            historical_date -= timedelta(days=1)
        historical_end = historical_date.replace(
            hour=15,
            minute=random.randint(0, 59),
            second=0,
            microsecond=0,
        )
        return get_market_observation(
            symbol=symbol,
            lookback_minutes=self._observation_lookback_minutes,
            data_client=data_client,
            end=historical_end.astimezone(timezone.utc),
        )

    def predict(self, symbol: str, observation: dict[str, Any] | None = None) -> dict[str, object]:
        """Fetch one market window, call both existing clients, and combine them."""
        symbol = symbol.strip().upper()
        data_source = "provided"
        if observation is None:
            observation = self._current_observation(symbol)
            data_source = "current"
        observation["symbol"] = symbol
        jepa_output: dict[str, Any] = {}
        jev_output: dict[str, Any] = {}
        try:
            jepa_output = get_jepa_inference().encode_observation(observation)
        except Exception as error:
            logger.exception("JEPA prediction failed for %s", symbol)
        try:
            jev_output = get_jev_inference().predict_direction(observation)
        except Exception as error:
            logger.warning("JEV prediction failed for %s: %s", symbol, error)

        jev_label = {"BUY": "up", "SELL": "down", "HOLD": "flat"}.get(
            str(jev_output.get("direction_label", "")).upper()
        )
        history = self._history(symbol)
        signal = combine_predictions(
            jepa_output or None,
            jev_label,
            jev_output.get("confidence"),
            history["jepa"],
            history["jev"],
            self.config,
        )
        result = {
            "symbol": symbol,
            "latest_price": observation.get("latest_price"),
            "data_source": data_source,
            "jepa": jepa_output or None,
            "jev": jev_output or None,
            **signal.as_dict(),
        }
        self.latest[symbol] = result
        return result

    def predict_dummy(self, symbol: str) -> dict[str, object]:
        """Run both models on prior-day data without executing an order."""
        result = self.predict(symbol, self._prior_day_observation(symbol))
        result["data_source"] = "prior_day"
        return result

    def predict_and_trade(
        self,
        symbol: str,
        quantity: float = 1.0,
        observation: dict[str, Any] | None = None,
        execute: bool = True,
    ) -> dict[str, object]:
        """Predict once and submit an eligible paper Alpaca order."""
        result = self.predict(symbol, observation)
        if not execute:
            result["order_status"] = "execution_disabled"
            return result
        if result["action"] == "flat":
            result["order_status"] = "no_trade_signal"
            return result

        client = get_trading_client_key()
        positions = client.get_all_positions()
        position = next(
            (item for item in positions if item.symbol.upper() == result["symbol"]),
            None,
        )
        action = str(result["action"])
        order_quantity = min(quantity, float(result["size"]))
        if order_quantity <= 0:
            result["order"] = None
            result["order_status"] = "invalid_order_quantity"
            return result
        if action == "up" and position is None:
            result["order"] = execute_order(
                client, str(result["symbol"]), "buy", order_quantity
            )
            result["order_status"] = "submitted_buy"
        elif action == "down" and position is not None:
            result["order"] = execute_order(
                client, str(result["symbol"]), "sell", float(position.qty)
            )
            result["order_status"] = "submitted_sell"
        elif action == "up":
            result["order"] = None
            result["order_status"] = "position_already_open"
        else:
            result["order"] = None
            result["order_status"] = "no_position_to_sell"
        return result


@lru_cache(maxsize=1)
def get_pipeline() -> CombinedPipeline:
    """Return the process-wide API pipeline."""
    return CombinedPipeline()