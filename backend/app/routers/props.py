from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.orm import PropLine
from app.schemas.props import PropLineResponse

router = APIRouter()


@router.get("/", response_model=list[PropLineResponse])
async def list_props(db: AsyncSession = Depends(get_db)) -> list[PropLine]:
    """List available prop lines, most recent first."""
    result = await db.execute(select(PropLine).order_by(PropLine.timestamp.desc()).limit(100))
    return list(result.scalars().all())


@router.get("/{prop_id}", response_model=PropLineResponse)
async def get_prop(prop_id: int, db: AsyncSession = Depends(get_db)) -> PropLine:
    """Get a single prop line by ID."""
    result = await db.execute(select(PropLine).where(PropLine.id == prop_id))
    prop = result.scalar_one_or_none()
    if prop is None:
        raise HTTPException(status_code=404, detail="Prop not found")
    return prop
