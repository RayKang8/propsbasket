import datetime

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Player(Base):
    __tablename__ = "players"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    team: Mapped[str | None] = mapped_column(String)
    position: Mapped[str | None] = mapped_column(String)

    game_logs: Mapped[list["PlayerGameLog"]] = relationship(back_populates="player")
    prop_lines: Mapped[list["PropLine"]] = relationship(back_populates="player")


class Game(Base):
    __tablename__ = "games"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    home_team: Mapped[str | None] = mapped_column(String)
    away_team: Mapped[str | None] = mapped_column(String)


class PlayerGameLog(Base):
    __tablename__ = "player_game_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"))
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"))
    minutes: Mapped[float | None] = mapped_column(Float)
    points: Mapped[int | None] = mapped_column(Integer)
    rebounds: Mapped[int | None] = mapped_column(Integer)
    assists: Mapped[int | None] = mapped_column(Integer)
    usage_rate: Mapped[float | None] = mapped_column(Float)
    shots_attempted: Mapped[int | None] = mapped_column(Integer)
    shots_made: Mapped[int | None] = mapped_column(Integer)

    player: Mapped["Player"] = relationship(back_populates="game_logs")


class Sportsbook(Base):
    __tablename__ = "sportsbooks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)


class PropLine(Base):
    __tablename__ = "prop_lines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"))
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"))
    sportsbook_id: Mapped[int] = mapped_column(ForeignKey("sportsbooks.id"))
    line_value: Mapped[float] = mapped_column(Float, nullable=False)
    odds: Mapped[int] = mapped_column(Integer, nullable=False)
    market_type: Mapped[str] = mapped_column(String, nullable=False)
    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime)

    player: Mapped["Player"] = relationship(back_populates="prop_lines")
    outcome: Mapped["PropOutcome | None"] = relationship(back_populates="prop_line")
    predictions: Mapped[list["ModelPrediction"]] = relationship(back_populates="prop_line")


class PropOutcome(Base):
    __tablename__ = "prop_outcomes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prop_line_id: Mapped[int] = mapped_column(ForeignKey("prop_lines.id"), unique=True)
    actual_stat: Mapped[float | None] = mapped_column(Float)
    did_hit: Mapped[bool | None] = mapped_column(Boolean)

    prop_line: Mapped["PropLine"] = relationship(back_populates="outcome")


class ModelPrediction(Base):
    __tablename__ = "model_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prop_line_id: Mapped[int] = mapped_column(ForeignKey("prop_lines.id"))
    model_probability: Mapped[float] = mapped_column(Float, nullable=False)
    implied_probability: Mapped[float] = mapped_column(Float, nullable=False)
    edge: Mapped[float] = mapped_column(Float, nullable=False)
    prediction_time: Mapped[datetime.datetime] = mapped_column(DateTime, nullable=False)

    prop_line: Mapped["PropLine"] = relationship(back_populates="predictions")


class EvaluationMetric(Base):
    __tablename__ = "evaluation_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    log_loss: Mapped[float | None] = mapped_column(Float)
    brier_score: Mapped[float | None] = mapped_column(Float)
    accuracy: Mapped[float | None] = mapped_column(Float)
