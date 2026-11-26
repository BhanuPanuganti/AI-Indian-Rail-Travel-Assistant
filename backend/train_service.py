from typing import Any

try:
    from backend.railway_repository import RailwayRepository
except ModuleNotFoundError:
    # Allow running backend scripts directly (e.g. python backend/test_local_data.py).
    from railway_repository import RailwayRepository

try:
    from backend.train_ranker import rank_trains
except ModuleNotFoundError:
    from train_ranker import rank_trains


try:
    from backend.format_utils import format_duration
except ModuleNotFoundError:
    from format_utils import format_duration


repository = RailwayRepository()


def resolve_station(
    query: str,
) -> list[dict]:

    return repository.find_stations(
        query
    )

def search_by_stations(
    origin_stations: list[dict],
    destination_stations: list[dict],
    preference: str = "balanced",
) -> dict[str, Any]:

    trains = repository.search_trains(
        origin=[station["Station Code"] for station in origin_stations],
        destination=[station["Station Code"] for station in destination_stations],
    )

    for train in trains:

        train["duration"] = format_duration(
            train.get("duration_minutes")
        )

    return rank_trains(
        trains,
        preference=preference,
    )

def search_by_city(
    origin: str,
    destination: str,
    preference: str = "balanced",
) -> dict[str, Any]:

    origin_stations = resolve_station(
        origin
    )

    destination_stations = resolve_station(
        destination
    )

    return search_by_stations(
        origin_stations=origin_stations,
        destination_stations=destination_stations,
        preference=preference,
    )

def get_train_route(
    train_number: str,
) -> list[dict]:

    return repository.find_train_route(
        train_number
    )


def search_trains(
    origin: str,
    destination: str,
    preference: str = "balanced",
) -> dict[str, Any]:

    trains = repository.search_trains(
        origin=origin,
        destination=destination,
    )

    for train in trains:

        train["duration"] = format_duration(
            train.get("duration_minutes")
        )

    return rank_trains(
        trains,
        preference=preference,
    )