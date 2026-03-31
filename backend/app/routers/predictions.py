from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.orm import ModelPrediction
from app.schemas.props import PredictionResponse

router = APIRouter()


@router.get("/", response_model=list[PredictionResponse])
async def list_predictions(db: AsyncSession = Depends(get_db)) -> list[ModelPrediction]:
    """List model predictions, most recent first."""
    result = await db.execute(
        select(ModelPrediction).order_by(ModelPrediction.prediction_time.desc()).limit(100)
    )
    return list(result.scalars().all())


@router.get("/top-edges", response_model=list[PredictionResponse])
async def top_edges(
    limit: int = Query(default=10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
) -> list[ModelPrediction]:
    """Return props ranked by edge (model_probability − implied_probability), highest first."""
    result = await db.execute(
        select(ModelPrediction).order_by(ModelPrediction.edge.desc()).limit(limit)
    )
    return list(result.scalars().all())
