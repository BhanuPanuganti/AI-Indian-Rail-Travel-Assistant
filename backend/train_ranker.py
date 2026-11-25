from typing import Any


VALID_PREFERENCES = ("balanced", "fastest", "shortest_distance", "earliest")

_PREFERENCE_ALIASES = {
    "fast": "fastest",
    "quickest": "fastest",
    "shortest": "shortest_distance",
    "distance": "shortest_distance",
    "early": "earliest",
    "earliest_departure": "earliest",
}

_INF = float("inf")


def normalize_preference(preference: str | None) -> str:
    """Map free-form preference text to one of VALID_PREFERENCES."""

    if not preference:
        return "balanced"

    value = preference.strip().lower().replace(" ", "_").replace("-", "_")
    value = _PREFERENCE_ALIASES.get(value, value)

    return value if value in VALID_PREFERENCES else "balanced"


def _num(value) -> float:
    return _INF if value is None else value


def rank_trains(
    trains: list[dict],
    preference: str = "balanced",
) -> dict[str, Any]:

    preference = normalize_preference(preference)

    if not trains:
        return {
            "trains": [],
            "recommended_train": None,
            "reason": "No trains found.",
            "ranking_preference": preference,
        }

    # Trains whose duration could not be computed are kept, but they sort
    # after trains with known values.
    ranked = list(trains)

    if preference == "fastest":
        ranked.sort(key=lambda t: _num(t.get("duration_minutes")))
        reason = "Fastest option."
    elif preference == "shortest_distance":
        ranked.sort(key=lambda t: _num(t.get("distance_km")))
        reason = "Shortest distance option."
    elif preference == "earliest":
        ranked.sort(key=lambda t: t.get("departure_time") or "99:99:99")
        reason = "Earliest departure."
    else:
        ranked.sort(
            key=lambda t: (_num(t.get("duration_minutes")), _num(t.get("distance_km")))
        )
        reason = "Best overall balance of speed and distance."

    return {
        "trains": ranked,
        "recommended_train": ranked[0],
        "reason": reason,
        "ranking_preference": preference,
    }
