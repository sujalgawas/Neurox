from app.combined.combiner import CombinerConfig, combine_predictions


def ready_histories():
    return [0.1], [0.1]


def test_warm_up_stays_flat_and_records_current_values():
    jepa, jev = [], []
    result = combine_predictions([0.0, 0.1, 0.9], "up", 0.9, jepa, jev, CombinerConfig(window=8))
    assert result.action == "flat"
    assert result.rule == "warm_up"
    assert jepa == [0.9]
    assert jev == [0.9]


def test_missing_model_fails_closed():
    jepa, jev = ready_histories()
    result = combine_predictions(None, "up", 1.0, jepa, jev, CombinerConfig(window=4))
    assert result.action == "flat"
    assert result.rule == "missing_model"


def test_opposite_direction_is_vetoed():
    jepa, jev = ready_histories()
    result = combine_predictions([0.9, 0.0, 0.1], "up", 1.0, jepa, jev, CombinerConfig(window=4))
    assert result.action == "flat"
    assert result.rule == "opposite_veto"


def test_same_direction_trades_at_entry_threshold():
    jepa, jev = ready_histories()
    result = combine_predictions([0.0, 0.0, 1.0], "up", 1.0, jepa, jev, CombinerConfig(window=4))
    assert result.action == "up"
    assert result.rule == "both_same"
    assert result.size == 1.0


def test_application_warmup_allows_trading_before_full_percentile_window():
    jepa, jev = [0.1, 0.2], [0.1, 0.2]
    result = combine_predictions(
        [0.0, 0.0, 1.0],
        "up",
        1.0,
        jepa,
        jev,
        CombinerConfig(window=500, warmup_period=2),
    )

    assert result.action == "up"
    assert result.rule == "both_same"


def test_same_direction_below_entry_threshold_stays_flat():
    jepa, jev = ready_histories()
    result = combine_predictions([0.0, 0.0, 0.8], "up", 0.8, jepa, jev,
                                  CombinerConfig(window=4, entry_threshold=1.1))
    assert result.action == "flat"
    assert result.rule == "entry_threshold"


def test_solo_direction_requires_solo_threshold_and_uses_half_size():
    jepa, jev = ready_histories()
    result = combine_predictions([0.0, 1.0, 0.0], "up", 1.0, jepa, jev,
                                  CombinerConfig(window=4, solo_threshold=0.8))
    assert result.action == "up"
    assert result.rule == "solo"
    assert result.size == 0.5


def test_flat_pair_is_flat():
    jepa, jev = ready_histories()
    result = combine_predictions([0.0, 1.0, 0.0], "flat", 0.0, jepa, jev, CombinerConfig(window=4))
    assert result.action == "flat"
    assert result.rule == "both_flat"


def test_percentile_uses_previous_values_not_current_or_future_values():
    jepa, jev = [0.2, 0.4], [0.2, 0.4]
    result = combine_predictions([0.0, 0.0, 0.3], "up", 0.3, jepa, jev, CombinerConfig(window=8))
    assert result.normalized_jepa == 0.5
    assert result.normalized_jev == 0.5
    assert jepa == [0.2, 0.4, 0.3]