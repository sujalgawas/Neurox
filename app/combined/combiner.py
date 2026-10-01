"""Pure confidence-normalizing rules for the combined trading signal."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, MutableMapping, Sequence


@dataclass(frozen=True)
class CombinerConfig:
    """Runtime parameters for signal normalization and rule selection."""

    window: int = 500
    jepa_weight: float = 0.5
    jev_weight: float = 0.5
    entry_threshold: float = 0.6
    solo_threshold: float = 0.85
    max_size: float = 1.0
    warmup_period: int | None = None


@dataclass(frozen=True)
class CombinedSignal:
    """The complete, explainable result for one symbol and one bar."""

    action: str
    normalized_jepa: float
    normalized_jev: float
    combined_score: float
    size: float
    rule: str
    jepa_score: float | None
    jev_score: float | None
    jepa_missing: bool
    jev_missing: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _jepa_score(values: Any) -> float | None:
    if isinstance(values, dict):
        label = str(values.get("direction_label", "")).strip().lower()
        confidence = values.get("direction_confidence", values.get("confidence"))
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            return None
        if label == "up":
            return confidence
        if label == "down":
            return -confidence
        if label in {"flat", "hold"}:
            return 0.0
        return None
    probabilities = _valid_probability_vector(values)
    return None if probabilities is None else probabilities[2] - probabilities[0]


def _valid_probability_vector(values: Sequence[float] | None) -> tuple[float, float, float] | None:
    if values is None or len(values) != 3:
        return None
    try:
        probabilities = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return None
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in probabilities):
        return None
    return probabilities


def _valid_jev_score(decision: str | None, confidence: float | None) -> float | None:
    if decision is None or confidence is None:
        return None
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or not 0 <= value <= 1:
        return None
    label = str(decision).strip().lower()
    if label in {"flat", "hold"}:
        return 0.0
    if label == "up":
        return value
    if label == "down":
        return -value
    return None


def _percentile(value: float, history: Sequence[float]) -> float:
    """Rank a value against prior observations only, returning [0, 1]."""
    if not history:
        return 0.0
    return sum(previous <= value for previous in history) / len(history)


def _normalized(score: float | None, history: list[float], config: CombinerConfig) -> tuple[float, bool]:
    if score is None:
        return 0.0, False
    magnitude = abs(score)
    warmup_period = config.warmup_period or max(1, math.ceil(config.window / 4))
    ready = len(history) >= warmup_period
    normalized = math.copysign(_percentile(magnitude, history), score) if ready and magnitude else 0.0
    history.append(magnitude)
    del history[:-config.window]
    return normalized, ready


def combine_predictions(
    jepa_probabilities: Sequence[float] | dict[str, Any] | None,
    jev_decision: str | None,
    jev_confidence: float | None,
    jepa_history: list[float],
    jev_history: list[float],
    config: CombinerConfig = CombinerConfig(),
) -> CombinedSignal:
    """Combine one bar of model output without observing future predictions."""
    jepa_score = _jepa_score(jepa_probabilities)
    jev_score = _valid_jev_score(jev_decision, jev_confidence)
    jepa_missing = jepa_score is None
    jev_missing = jev_score is None
    normalized_jepa, jepa_ready = _normalized(jepa_score, jepa_history, config)
    normalized_jev, jev_ready = _normalized(jev_score, jev_history, config)
    combined_score = config.jepa_weight * normalized_jepa + config.jev_weight * normalized_jev

    if jepa_missing or jev_missing:
        rule = "missing_model"
    elif not jepa_ready or not jev_ready:
        rule = "warm_up"
    elif normalized_jepa and normalized_jev and normalized_jepa * normalized_jev < 0:
        rule = "opposite_veto"
    elif normalized_jepa and normalized_jev:
        action = "up" if combined_score > 0 else "down"
        if abs(combined_score) >= config.entry_threshold:
            return CombinedSignal(action, normalized_jepa, normalized_jev, combined_score,
                                  config.max_size * abs(combined_score), "both_same", jepa_score,
                                  jev_score, jepa_missing, jev_missing)
        rule = "entry_threshold"
    elif normalized_jepa or normalized_jev:
        directional = normalized_jepa or normalized_jev
        if abs(directional) >= config.solo_threshold:
            action = "up" if directional > 0 else "down"
            return CombinedSignal(action, normalized_jepa, normalized_jev, combined_score,
                                  config.max_size * abs(directional) / 2, "solo", jepa_score,
                                  jev_score, jepa_missing, jev_missing)
        rule = "solo_threshold"
    else:
        rule = "both_flat"
    return CombinedSignal("flat", normalized_jepa, normalized_jev, combined_score, 0.0, rule,
                          jepa_score, jev_score, jepa_missing, jev_missing)


def signal_config_from_settings(settings: object) -> CombinerConfig:
    """Build the pure combiner configuration from the application settings."""
    return CombinerConfig(
        window=settings.COMBINED_PERCENTILE_WINDOW,
        jepa_weight=settings.COMBINED_JEPA_WEIGHT,
        jev_weight=settings.COMBINED_JEV_WEIGHT,
        entry_threshold=settings.COMBINED_ENTRY_THRESHOLD,
        solo_threshold=settings.COMBINED_SOLO_THRESHOLD,
        max_size=settings.COMBINED_MAX_SIZE,
        warmup_period=settings.COMBINED_WARMUP_PERIOD,
    )


def histories_for_symbol(
    histories: MutableMapping[str, dict[str, list[float]]], symbol: str
) -> dict[str, list[float]]:
    """Return mutable per-symbol JEPA and JEV rolling histories."""
    return histories.setdefault(symbol.upper(), {"jepa": [], "jev": []})