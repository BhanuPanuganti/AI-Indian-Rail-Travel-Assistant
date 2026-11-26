import threading
from functools import lru_cache
from typing import Any

try:
    from backend.time_utils import calculate_route_duration
    from backend.train_db import DB_FILE, connect, ensure_database
except ModuleNotFoundError:
    # Allow running backend scripts directly from the backend directory.
    from time_utils import calculate_route_duration
    from train_db import DB_FILE, connect, ensure_database


MIN_NAME_QUERY_LENGTH = 3

# Column names returned for route stops (kept identical to the original CSV
# headers so the /trains/{number} API response is unchanged).
_ROUTE_COLUMNS = """
    train_no AS "Train No",
    train_name AS "Train Name",
    seq AS "SEQ",
    station_code AS "Station Code",
    station_name AS "Station Name",
    arrival_time AS "Arrival time",
    departure_time AS "Departure Time",
    distance AS "Distance",
    source_station AS "Source Station",
    source_station_name AS "Source Station Name",
    destination_station AS "Destination Station",
    destination_station_name AS "Destination Station Name"
"""


class RailwayRepository:
    """Read-only access to the timetable stored in SQLite."""

    def __init__(self, db_path=DB_FILE):
        self._db_path = db_path
        self._local = threading.local()
        self._ready = False
        self._ready_lock = threading.Lock()
        self._route_cache = lru_cache(maxsize=256)(self._load_route)

    # -- connection -----------------------------------------------------------

    def _conn(self):
        if not self._ready:
            with self._ready_lock:
                if not self._ready:
                    ensure_database(self._db_path)
                    self._ready = True

        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = connect(self._db_path)
            self._local.connection = connection
        return connection

    # -- routes ---------------------------------------------------------------

    def _load_route(self, train_number: str) -> tuple[dict, ...]:
        rows = self._conn().execute(
            f"SELECT {_ROUTE_COLUMNS} FROM stops WHERE train_no = ? ORDER BY seq",
            (train_number,),
        ).fetchall()
        return tuple(dict(row) for row in rows)

    def find_train_route(
        self,
        train_number: str,
    ) -> list[dict[str, Any]]:

        route = self._route_cache(str(train_number).strip())
        return [dict(stop) for stop in route]

    # -- stations -------------------------------------------------------------

    def find_stations(
        self,
        query: str,
    ) -> list[dict[str, Any]]:

        query = query.strip().lower()

        if not query:
            return []

        # Substring name matching only for queries long enough to be
        # meaningful; very short queries ("a", "ne") would match thousands
        # of stations. Exact station-code matches always apply.
        if len(query) >= MIN_NAME_QUERY_LENGTH:
            sql = """
                SELECT station_code, station_name FROM stations
                WHERE lower(station_code) = ? OR instr(name_lower, ?) > 0
                ORDER BY first_row
            """
            params = (query, query)
        else:
            sql = """
                SELECT station_code, station_name FROM stations
                WHERE lower(station_code) = ?
                ORDER BY first_row
            """
            params = (query,)

        return [
            {"Station Code": row["station_code"], "Station Name": row["station_name"]}
            for row in self._conn().execute(sql, params)
        ]

    # -- search ---------------------------------------------------------------

    def search_trains(
        self,
        origin: str | list[str],
        destination: str | list[str],
    ) -> list[dict[str, Any]]:
        """
        Find direct trains from any of the origin station codes to any of
        the destination station codes. Each train appears at most once; if
        it serves several origin/destination pairs, the quickest is kept.
        """

        origin_codes = _normalize_codes(origin)
        destination_codes = _normalize_codes(destination)

        if not origin_codes or not destination_codes:
            return []

        rows = self._conn().execute(
            f"""
            SELECT
                o.train_no, o.train_name,
                o.seq AS origin_seq, o.station_code AS origin_code,
                o.station_name AS origin_name, o.departure_time,
                o.distance AS origin_distance,
                d.seq AS dest_seq, d.station_code AS dest_code,
                d.station_name AS dest_name, d.arrival_time,
                d.distance AS dest_distance
            FROM stops o
            JOIN stops d ON d.train_no = o.train_no
            WHERE o.station_code IN ({",".join("?" * len(origin_codes))})
              AND d.station_code IN ({",".join("?" * len(destination_codes))})
              AND o.seq < d.seq
              AND o.station_code != d.station_code
            """,
            (*origin_codes, *destination_codes),
        ).fetchall()

        best_by_train: dict[str, dict[str, Any]] = {}

        for row in rows:
            train_number = row["train_no"]
            route = self._route_cache(train_number)

            origin_index = _position(route, row["origin_seq"])
            destination_index = _position(route, row["dest_seq"])
            if origin_index is None or destination_index is None:
                continue

            candidate = {
                "train_number": train_number,
                "train_name": row["train_name"],
                "origin_code": row["origin_code"],
                "origin_name": row["origin_name"],
                "destination_code": row["dest_code"],
                "destination_name": row["dest_name"],
                "departure_time": row["departure_time"],
                "arrival_time": row["arrival_time"],
                "duration_minutes": calculate_route_duration(
                    route=route,
                    origin_index=origin_index,
                    destination_index=destination_index,
                ),
                "distance_km": _safe_distance(row["origin_distance"], row["dest_distance"]),
                "source": "static_schedule_2017",
            }

            existing = best_by_train.get(train_number)
            if existing is None or _duration_key(candidate) < _duration_key(existing):
                best_by_train[train_number] = candidate

        return list(best_by_train.values())


def _position(route: tuple[dict, ...], seq: int) -> int | None:
    for index, stop in enumerate(route):
        if stop["SEQ"] == seq:
            return index
    return None


def _normalize_codes(codes: str | list[str]) -> list[str]:
    if isinstance(codes, str):
        codes = [codes]
    return [c.strip().upper() for c in codes if c and c.strip()]


def _safe_distance(start, end) -> float | None:
    if start is None or end is None:
        return None
    return round(float(end) - float(start), 2)


def _duration_key(train: dict[str, Any]) -> float:
    duration = train.get("duration_minutes")
    return float("inf") if duration is None else duration
