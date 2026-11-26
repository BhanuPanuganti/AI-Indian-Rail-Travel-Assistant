import json
import sys
from pathlib import Path

from langchain_core.tools import tool

for candidate in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
    backend_dir = candidate / "backend"
    if backend_dir.exists():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        if str(backend_dir) not in sys.path:
            sys.path.insert(0, str(backend_dir))
        break

from rag.rag_service import answer_question

try:
    from backend.railradar_client import get_live_train_status
except ModuleNotFoundError:
    from railradar_client import get_live_train_status

try:
    from backend.train_service import (
        resolve_station,
        get_train_route,
        search_by_city,
    )
except ModuleNotFoundError:
    from train_service import (
        resolve_station,
        get_train_route,
        search_by_city,
    )

# Maximum number of station matches before we flag as ambiguous
_STATION_AMBIGUITY_THRESHOLD = 10


@tool
def station_search(query: str) -> str:
    """Find railway stations matching a city, station name, or station code."""

    results = resolve_station(query)

    if not results:
        return json.dumps(
            {
                "error": "no_stations_found",
                "query": query,
                "message": (
                    f"No stations found matching '{query}'. "
                    "Please check the spelling or try a different city name or station code."
                ),
            },
            ensure_ascii=False,
        )

    if len(results) > _STATION_AMBIGUITY_THRESHOLD:
        top = results[:_STATION_AMBIGUITY_THRESHOLD]
        return json.dumps(
            {
                "ambiguous": True,
                "query": query,
                "total_matches": len(results),
                "shown": top,
                "message": (
                    f"'{query}' matched {len(results)} stations. "
                    "Showing the top 10. Please ask the user to be more specific "
                    "(e.g. provide the exact station code or full city name)."
                ),
            },
            ensure_ascii=False,
        )

    return json.dumps(results, ensure_ascii=False)


@tool
def railway_knowledge_search(
    question: str,
) -> str:
    """
    Search official Indian Railways and IRCTC documents
    for rules, policies, Tatkal, cancellation, refunds,
    luggage, concessions, e-ticketing, and related
    railway information.
    """

    result = answer_question(question)
    answer = result["answer"]
    sources = result.get("sources", [])

    if not answer or not answer.strip():
        return (
            "I couldn't find relevant information about this in the "
            "official railway documents I have."
        )

    if sources:
        answer += "\n\n📚 Sources\n"
        for s in sources:
            answer += f"• {s['source']} — Page {s['page']}\n"

    return answer


@tool
def train_search(
    origin: str,
    destination: str,
    preference: str = "balanced",
) -> str:
    """Search direct trains between two cities or stations.

    preference must be one of: "balanced", "fastest",
    "shortest_distance", "earliest".
    """

    print("TRAIN SEARCH START")

    try:
        results = search_by_city(
            origin=origin,
            destination=destination,
            preference=preference,
        )
    except Exception as exc:
        print(f"TRAIN SEARCH ERROR: {exc}")
        return json.dumps(
            {
                "error": "search_failed",
                "message": (
                    "An unexpected error occurred while searching for trains. "
                    "Please try again."
                ),
            },
            ensure_ascii=False,
        )

    print("TRAIN SEARCH END")

    trains = results.get("trains", [])

    if not trains:
        return json.dumps(
            {
                "error": "no_trains_found",
                "origin": origin,
                "destination": destination,
                "message": (
                    f"No trains found between '{origin}' and '{destination}'. "
                    "This could be because: (1) the station codes were not resolved — "
                    "try using exact station codes; (2) there are no direct trains on "
                    "this route in the current schedule data."
                ),
            },
            ensure_ascii=False,
        )

    return json.dumps(results, ensure_ascii=False)


@tool
def train_details(
    train_number: str,
) -> str:
    """Get the complete route and stops of a train."""

    results = get_train_route(train_number)

    if not results:
        return json.dumps(
            {
                "error": "train_not_found",
                "train_number": train_number,
                "message": (
                    f"No route information found for train number '{train_number}'. "
                    "Please verify the train number and try again."
                ),
            },
            ensure_ascii=False,
        )

    return json.dumps(results, ensure_ascii=False)


@tool
def live_train_status(
    train_number: str,
    date: str = "",
) -> str:
    """Get the LIVE running status of one train: where it is now, its delay,
    and its previous/next halt. Use only when the user asks about current
    position, delay or whether a train is running late, not for timetables.

    train_number: the train number, e.g. "12302".
    date: optional journey start date (YYYY-MM-DD) when the train departed its
    source station; leave empty for the current run.
    """

    return json.dumps(
        get_live_train_status(train_number, date or None),
        ensure_ascii=False,
    )


railway_tools = [
    station_search,
    train_search,
    train_details,
    live_train_status,
    railway_knowledge_search,
]