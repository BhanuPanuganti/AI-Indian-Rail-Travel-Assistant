from typing import Any

def rank_trains(
    trains: list[dict],
    preference: str = "balanced",
) -> dict[str, Any]:

    if not trains:
        return {
            "trains": [],
            "recommended_train": None,
            "reason": "No trains found.",
            "ranking_preference": preference,
        }

    valid_trains = [
        train
        for train in trains
        if train.get("duration_minutes")
        is not None
    ]

    reason = ""

    if preference == "fastest":
        valid_trains.sort(key=lambda train: train["duration_minutes"])
        reason = "Fastest option."
    elif preference == "shortest_distance":
        valid_trains.sort(key=lambda train: train.get("distance_km", float("inf")))
        reason = "Shortest distance option."
    elif preference == "earliest":
        valid_trains.sort(key=lambda train: train.get("departure_time", "23:59:59"))
        reason = "Earliest departure."
    else:
        valid_trains.sort(key=lambda train: (train["duration_minutes"], train.get("distance_km", float("inf"))))
        reason = "Best overall balance of speed and distance."
        preference = "balanced"

    recommended = valid_trains[0] if valid_trains else None

    return {
        "trains": valid_trains,
        "recommended_train": recommended,
        "reason": reason,
        "ranking_preference": preference,
    }