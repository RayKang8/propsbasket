import datetime

from pydantic import BaseModel, computed_field


class PropLineResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    player_id: int
    game_id: int
    sportsbook_id: int
    line_value: float
    odds: int
    market_type: str
    timestamp: datetime.datetime


class PredictionResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    prop_line_id: int
    model_probability: float
    implied_probability: float
    edge: float
    prediction_time: datetime.datetime

    @computed_field
    @property
    def edge_pct(self) -> str:
        return f"{self.edge * 100:+.1f}%"
