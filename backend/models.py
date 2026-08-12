from typing import Literal

from pydantic import BaseModel, Field


class TrainStop(BaseModel):
    train_number: str
    train_name: str

    station_code: str
    station_name: str

    arrival_time: str | None = None
    departure_time: str | None = None

    sequence: int
    distance: float | None = None


class TrainSearchResult(BaseModel):
    train_number: str
    train_name: str

    origin_code: str
    origin_name: str

    destination_code: str
    destination_name: str

    departure_time: str | None = None
    arrival_time: str | None = None

    duration_minutes: int | None = None

    source: str = "static_schedule_2017"


class TravelIntent(BaseModel):
    origin: str = Field(
        description="Origin city or station"
    )

    destination: str = Field(
        description="Destination city or station"
    )

    preference: Literal[
        "balanced",
        "fastest",
        "shortest_distance",
    ] = Field(
        default="balanced",
        description="User's preferred train ranking"
    )

from enum import Enum

class ChatIntent(str, Enum):
    TRAIN_SEARCH = "TRAIN_SEARCH"
    TRAIN_DETAILS = "TRAIN_DETAILS"
    STATION_SEARCH = "STATION_SEARCH"
    RAILWAY_RULES = "RAILWAY_RULES"
    GENERAL_CHAT = "GENERAL_CHAT"

class UserIntent(BaseModel):
    intent: ChatIntent = Field(
        description="The classified intent of the user's latest query."
    )
    preference: str | None = Field(
        default=None,
        description="Optional ranking preference if mentioned (e.g., fastest, earliest, shortest, balanced)."
    )