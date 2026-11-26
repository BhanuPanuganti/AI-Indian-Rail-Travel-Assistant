"""Deterministic tests for the train data layer (no LLM or network calls)."""

import json

import pytest

from backend import train_service
from backend.format_utils import format_duration
from backend.time_utils import calculate_route_duration, time_to_minutes
from backend.train_ranker import normalize_preference, rank_trains


# --- station search ---------------------------------------------------------

def test_station_search_treats_query_as_plain_text():
    assert train_service.resolve_station("(") == []
    assert train_service.resolve_station("a.*") == []


def test_empty_and_short_queries_do_not_match_everything():
    assert train_service.resolve_station("") == []
    assert train_service.resolve_station("   ") == []
    assert train_service.resolve_station("a") == []


def test_short_query_matches_exact_station_code():
    assert train_service.resolve_station("be") == [
        {"Station Code": "BE", "Station Name": "BAREILLY JN."}
    ]


def test_city_name_resolves_to_its_stations():
    codes = {s["Station Code"] for s in train_service.resolve_station("secunderabad")}
    assert "SC" in codes


# --- train search -----------------------------------------------------------

def test_known_direct_train_is_found():
    result = train_service.search_trains("NDLS", "HWH", preference="fastest")
    numbers = [t["train_number"] for t in result["trains"]]
    assert "12302" in numbers
    assert result["recommended_train"]["train_number"] == "12302"


def test_city_search_returns_each_train_once():
    result = train_service.search_by_city("delhi", "mumbai")
    numbers = [t["train_number"] for t in result["trains"]]
    assert numbers, "expected at least one Delhi -> Mumbai train"
    assert len(numbers) == len(set(numbers))


def test_wrong_direction_returns_nothing_for_one_way_pair():
    # 107 runs SWV -> MAO only.
    trains = train_service.repository.search_trains("MAO", "SWV")
    assert all(t["train_number"] != "107" for t in trains)


def test_search_results_are_strict_json():
    result = train_service.search_by_city("delhi", "mumbai")
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("train_number", ["6919", "56905", "12723"])
def test_train_route_is_strict_json(train_number):
    route = train_service.get_train_route(train_number)
    assert route
    json.dumps(route, allow_nan=False)


def test_unknown_train_route_is_empty():
    assert train_service.get_train_route("00000") == []


# --- duration maths ---------------------------------------------------------

def test_time_to_minutes():
    assert time_to_minutes("10:25:00") == 625
    assert time_to_minutes("") is None
    assert time_to_minutes("not a time") is None


def test_route_duration_crosses_midnight():
    route = [
        {"Arrival time": "00:00:00", "Departure Time": "22:00:00"},
        {"Arrival time": "23:30:00", "Departure Time": "23:40:00"},
        {"Arrival time": "02:15:00", "Departure Time": "00:00:00"},
    ]
    assert calculate_route_duration(route, 0, 2) == 4 * 60 + 15


def test_route_duration_rejects_reversed_indexes():
    route = [{"Arrival time": "10:00:00", "Departure Time": "10:05:00"}]
    assert calculate_route_duration(route, 0, 0) is None


@pytest.mark.parametrize(
    "minutes, expected",
    [(None, None), (0, "0m"), (45, "45m"), (125, "2h 5m"), (1500, "1d 1h")],
)
def test_format_duration(minutes, expected):
    assert format_duration(minutes) == expected


# --- ranking ----------------------------------------------------------------

TRAINS = [
    {"train_number": "A", "duration_minutes": 300, "distance_km": 500.0, "departure_time": "09:00:00"},
    {"train_number": "B", "duration_minutes": 240, "distance_km": 520.0, "departure_time": "18:00:00"},
    {"train_number": "C", "duration_minutes": None, "distance_km": 480.0, "departure_time": "06:00:00"},
]


@pytest.mark.parametrize(
    "preference, expected_order",
    [
        ("fastest", ["B", "A", "C"]),
        ("shortest_distance", ["C", "A", "B"]),
        ("earliest", ["C", "A", "B"]),
        ("balanced", ["B", "A", "C"]),
    ],
)
def test_rank_trains_orders_by_preference(preference, expected_order):
    ranked = rank_trains([dict(t) for t in TRAINS], preference)
    assert [t["train_number"] for t in ranked["trains"]] == expected_order
    assert ranked["recommended_train"]["train_number"] == expected_order[0]


def test_rank_trains_keeps_trains_without_duration():
    ranked = rank_trains([dict(TRAINS[2])], "fastest")
    assert [t["train_number"] for t in ranked["trains"]] == ["C"]


def test_rank_trains_empty():
    assert rank_trains([], "fastest")["recommended_train"] is None


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Fastest", "fastest"),
        ("quickest", "fastest"),
        ("shortest", "shortest_distance"),
        ("early", "earliest"),
        ("nonsense", "balanced"),
        (None, "balanced"),
    ],
)
def test_normalize_preference(raw, expected):
    assert normalize_preference(raw) == expected


def test_route_keeps_original_column_names_and_order():
    route = train_service.get_train_route("107")
    assert [stop["Station Code"] for stop in route] == ["SWV", "THVM", "KRMI", "MAO"]
    assert route[0]["Train Name"] == "SWV-MAO-VLNK"
    assert route[-1]["Distance"] == 78.0


def test_train_route_with_missing_distance_returns_null():
    route = train_service.get_train_route("6919")
    assert any(stop["Distance"] is None for stop in route)


def test_build_database_from_small_csv(tmp_path):
    from backend.train_db import build_database, connect

    csv_path = tmp_path / "trains.csv"
    csv_path.write_text(
        "Train No,Train Name,SEQ,Station Code,Station Name,Arrival time,Departure Time,"
        "Distance,Source Station,Source Station Name,Destination Station,Destination Station Name\n"
        " 1 ,T1,1, aaa ,Alpha ,00:00:00,10:00:00,0,AAA,Alpha,BBB,Beta\n"
        "1,T1,2,BBB,Beta,12:00:00,00:00:00,,AAA,Alpha,BBB,Beta\n",
        encoding="utf-8",
    )
    db_path = build_database(csv_path, tmp_path / "trains.sqlite")

    with connect(db_path) as conn:
        rows = [dict(r) for r in conn.execute("SELECT train_no, station_code, station_name, distance FROM stops")]
    assert rows == [
        {"train_no": "1", "station_code": "AAA", "station_name": "Alpha", "distance": 0.0},
        {"train_no": "1", "station_code": "BBB", "station_name": "Beta", "distance": None},
    ]
