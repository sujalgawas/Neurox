from __future__ import annotations

from typing import Any

import httpx


class JEVInference:
    def __init__(self, api_url: str, api_key: str, model: str):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model

    def _request_direction(self, observation: dict[str, Any]) -> tuple[str, Any]:
        records = observation.get("raw_records", [])[-10:]
        state = {
            "symbol": observation.get("symbol"),
            "latest_price": observation.get("latest_price"),
            "latest_minute_bars": records,
        }
        response = httpx.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "state": state,
                "model": self.model,
                "questions": {
                    "direction": {
                        "type": "choice",
                        "instructions": "What should the trading system do next based on these minute bars?",
                        "criteria": {
                            "BUY": "Open or maintain a long position because the near-term direction is up.",
                            "SELL": "Close a long position because the near-term direction is down.",
                            "HOLD": "Do not change the position because the direction is flat or uncertain.",
                        },
                    }
                },
            },
            timeout=30.0,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            detail = response.text[:500]
            raise ValueError(
                f"JEV API returned HTTP {response.status_code}: {detail}"
            ) from error

        payload = response.json()
        direction_answer = payload["answers"]["direction"]
        return str(direction_answer["choice"]), direction_answer["confidence"]

    def predict_direction(self, observation: dict[str, Any]) -> dict[str, Any]:
        direction_label, confidence = self._request_direction(observation)
        direction_label = direction_label.upper()
        direction = {
            "BUY": 1,
            "SELL": -1,
            "HOLD": 0,
        }.get(direction_label)
        if direction is None:
            raise ValueError(f"JEV returned unsupported direction: {direction_label}")
        return {
            "latest_price": observation["latest_price"],
            "direction": direction,
            "direction_label": direction_label,
            "confidence": confidence,
        }

    def predict_signal(self, history: Any, observation: dict[str, Any]) -> float:
        return float(self.predict_direction(observation)["direction"])