import datetime

from pydantic import BaseModel, computed_field


class PredictionSummary(BaseModel):
    total_predictions: int
    positive_edge_count: int
    avg_edge: float
    best_edge: float


class PlayerResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    name: str
    team: str | None


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
    player: PlayerResponse


class PredictionResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    prop_line_id: int
    model_probability: float
    implied_probability: float
    edge: float
    prediction_time: datetime.datetime
    prop_line: PropLineResponse

    @computed_field
    @property
    def edge_pct(self) -> str:
        return f"{self.edge * 100:+.1f}%"

    @computed_field
    @property
    def player_name(self) -> str:
        return self.prop_line.player.name

    @computed_field
    @property
    def player_team(self) -> str | None:
        return self.prop_line.player.team

    @computed_field
    @property
    def market_type(self) -> str:
        return self.prop_line.market_type

    @computed_field
    @property
    def line_value(self) -> float:
        return self.prop_line.line_value

    @computed_field
    @property
    def odds(self) -> int:
        return self.prop_line.odds
