from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

try:
    from backend.intent_service import extract_travel_intent
    from backend.train_service import (
        resolve_station,
        search_by_stations,
    )
    from backend.llm_service import llm
except ModuleNotFoundError:
    from intent_service import extract_travel_intent
    from train_service import (
        resolve_station,
        search_by_stations,
    )

    from llm_service import llm

class TravelState(TypedDict, total=False):
    user_message: str

    origin: str
    destination: str
    preference: str

    origin_stations: list[dict]
    destination_stations: list[dict]

    trains: list[dict]

    response: str


def extract_intent_node(
    state: TravelState,
) -> dict[str, Any]:

    intent = extract_travel_intent(
        state["user_message"]
    )

    return {
        "origin": intent.origin,
        "destination": intent.destination,
        "preference": intent.preference,
    }


def resolve_stations_node(
    state: TravelState,
) -> dict[str, Any]:

    origin_stations = resolve_station(
        state["origin"]
    )

    destination_stations = resolve_station(
        state["destination"]
    )

    return {
        "origin_stations": origin_stations,
        "destination_stations": destination_stations,
    }


def search_trains_node(
    state: TravelState,
) -> dict[str, Any]:

    trains = search_by_stations(
        origin_stations=state["origin_stations"],
        destination_stations=state["destination_stations"],
        preference=state["preference"],
    )

    return {
        "trains": trains,
    }

def build_response_node(
    state: TravelState,
) -> dict[str, Any]:

    trains = state.get("trains", [])

    if not trains:

        return {
            "response": (
                f"I couldn't find any trains from "
                f"{state['origin']} to "
                f"{state['destination']}."
            )
        }

    train_text = "\n".join(
        [
            (
                f"- Train {train['train_number']}: "
                f"{train['train_name']}, "
                f"departure {train['departure_time']}, "
                f"arrival {train['arrival_time']}, "
                f"duration {train.get('duration', 'N/A')}, "
                f"distance {train.get('distance_km', 'N/A')} km"
            )
            for train in trains[:10]
        ]
    )

    prompt = f"""
You are an Indian railway travel assistant.

The user asked:
{state['user_message']}

Extracted origin:
{state['origin']}

Extracted destination:
{state['destination']}

Preference:
{state['preference']}

Train search results:
{train_text}

Answer the user's request using ONLY the train information
provided above.

Rules:
- Do not invent trains or information.
- Mention the best matching train first.
- Keep the answer concise.
- Mention that the schedule data is from 2017 when presenting results.
- If there are multiple trains, briefly compare the best options.
"""

    response = llm.invoke(prompt)

    return {
        "response": response.content,
    }


builder = StateGraph(TravelState)

builder.add_node(
    "extract_intent",
    extract_intent_node,
)

builder.add_node(
    "resolve_stations",
    resolve_stations_node,
)

builder.add_node(
    "search_trains",
    search_trains_node,
)

builder.add_node(
    "build_response",
    build_response_node,
)

builder.add_edge(
    START,
    "extract_intent",
)

builder.add_edge(
    "extract_intent",
    "resolve_stations",
)

builder.add_edge(
    "resolve_stations",
    "search_trains",
)

builder.add_edge(
    "search_trains",
    "build_response",
)

builder.add_edge(
    "build_response",
    END,
)

travel_graph = builder.compile()