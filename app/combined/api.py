"""FastAPI routes for combined predictions."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import time

from .pipeline import get_pipeline

router = APIRouter(prefix="/combined", tags=["combined"])


class CombinedRequest(BaseModel):
    """Request body for one real combined prediction and paper trade."""

    symbol: str = Field(default="AAPL", min_length=1)
    quantity: float = Field(default=1.0, gt=0)


@router.post("/predict")
def predict(request: CombinedRequest) -> dict[str, object]:
    """Run 30 one-minute combined predictions and execute eligible actions."""
    results: list[dict[str, object]] = []
    pipeline = get_pipeline()
    iterations = 200
    for iteration in range(iterations):
        try:
            result = pipeline.predict_and_trade(
                request.symbol,
                quantity=request.quantity,
                execute=True,
            )
        except Exception as error:
            result = {"action": "error", "detail": str(error)}

        results.append({"iteration": iteration + 1, "result": result})

        if iteration < iterations - 1:
            time.sleep(60)

    return {"runs": results}


@router.post("/dummy")
def dummy(request: CombinedRequest) -> dict[str, object]:
    """Test both models on prior-day data without placing an order."""
    try:
        return get_pipeline().predict_dummy(request.symbol)
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error