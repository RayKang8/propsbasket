from fastapi import APIRouter, Depends, Query
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models.orm import ModelPrediction, PropLine
from app.schemas.props import PredictionResponse, PredictionSummary

router = APIRouter()

_eager = selectinload(ModelPrediction.prop_line).selectinload(PropLine.player)


@router.get("/summary", response_model=PredictionSummary)
async def prediction_summary(db: AsyncSession = Depends(get_db)) -> PredictionSummary:
    """Aggregate stats across all predictions."""
    row = await db.execute(
        select(
            func.count(ModelPrediction.id),
            func.sum(case((ModelPrediction.edge > 0, 1), else_=0)),
            func.avg(ModelPrediction.edge),
            func.max(ModelPrediction.edge),
        )
    )
    total, positive, avg_edge, best_edge = row.one()
    return PredictionSummary(
        total_predictions=total or 0,
        positive_edge_count=positive or 0,
        avg_edge=avg_edge or 0.0,
        best_edge=best_edge or 0.0,
    )


@router.get("/", response_model=list[PredictionResponse])
async def list_predictions(db: AsyncSession = Depends(get_db)) -> list[ModelPrediction]:
    """List model predictions, most recent first."""
    result = await db.execute(
        select(ModelPrediction)
        .options(_eager)
        .order_by(ModelPrediction.prediction_time.desc())
        .limit(100)
    )
    return list(result.scalars().all())


@router.get("/top-edges", response_model=list[PredictionResponse])
async def top_edges(
    limit: int = Query(default=10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
) -> list[ModelPrediction]:
    """Return props ranked by edge (model_probability − implied_probability), highest first."""
    result = await db.execute(
        select(ModelPrediction)
        .options(_eager)
        .order_by(ModelPrediction.edge.desc())
        .limit(limit)
    )
    return list(result.scalars().all())
