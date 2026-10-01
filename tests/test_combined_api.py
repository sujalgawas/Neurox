from app.combined import api


class RecordingPipeline:
    def __init__(self):
        self.calls = []

    def predict_and_trade(self, symbol, *, quantity, execute):
        self.calls.append((symbol, quantity, execute))
        return {"action": "flat"}


def test_predict_uses_trading_pipeline(monkeypatch):
    pipeline = RecordingPipeline()
    sleeps = []
    monkeypatch.setattr(api, "get_pipeline", lambda: pipeline)
    monkeypatch.setattr(api.time, "sleep", sleeps.append)

    response = api.predict(api.CombinedRequest(symbol="AAPL", quantity=2))

    assert len(response["runs"]) == 30
    assert pipeline.calls == [("AAPL", 2.0, True)] * 30
    assert sleeps == [60] * 29