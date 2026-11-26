"""
Build and open the SQLite copy of the train timetable.

The 2017 timetable CSV (~186k rows) is converted once into an indexed SQLite
file. Queries then read from disk instead of holding the whole table in a
pandas DataFrame, which keeps the backend's memory use small.

The Docker build runs `python -m backend.train_db` so the database ships in the
image; locally it is built automatically on first use.
"""

import csv
import os
import sqlite3
import threading
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "railway"
CSV_FILE = DATA_DIR / "Train_details_22122017.csv"
DB_FILE = Path(os.getenv("TRAIN_DB_PATH", str(DATA_DIR / "trains.sqlite")))

SCHEMA = """
CREATE TABLE stops (
    train_no TEXT NOT NULL,
    train_name TEXT,
    seq INTEGER,
    station_code TEXT NOT NULL,
    station_name TEXT,
    arrival_time TEXT,
    departure_time TEXT,
    distance REAL,
    source_station TEXT,
    source_station_name TEXT,
    destination_station TEXT,
    destination_station_name TEXT
);
CREATE INDEX idx_stops_station ON stops (station_code);
CREATE INDEX idx_stops_train ON stops (train_no, seq);

-- One row per distinct (code, name) pair, in order of first appearance.
CREATE TABLE stations (
    station_code TEXT NOT NULL,
    station_name TEXT,
    name_lower TEXT,
    first_row INTEGER
);
"""

_build_lock = threading.Lock()


def _to_float(value: str) -> float | None:
    value = _clean(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number  # NaN -> None


def _to_int(value: str) -> int | None:
    number = _to_float(value)
    return None if number is None else int(number)


# Markers the CSV uses for missing values (pandas' default NA strings), which
# the original pandas loader turned into empty values.
_MISSING = {
    "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
    "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a",
    "nan", "null",
}


def _clean(value: str | None) -> str:
    value = (value or "").strip()
    return "" if value in _MISSING else value


def build_database(csv_path: Path = CSV_FILE, db_path: Path = DB_FILE) -> Path:
    """Convert the timetable CSV into an indexed SQLite file (atomic replace)."""

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = db_path.with_suffix(".tmp")
    tmp_path.unlink(missing_ok=True)

    connection = sqlite3.connect(tmp_path)
    try:
        connection.executescript(SCHEMA)

        with open(csv_path, encoding="utf-8", newline="") as file:
            rows = (
                (
                    _clean(r["Train No"]),
                    _clean(r["Train Name"]),
                    _to_int(r["SEQ"]),
                    _clean(r["Station Code"]).upper(),
                    _clean(r["Station Name"]),
                    _clean(r["Arrival time"]),
                    _clean(r["Departure Time"]),
                    _to_float(r["Distance"]),
                    _clean(r["Source Station"]),
                    _clean(r["Source Station Name"]),
                    _clean(r["Destination Station"]),
                    _clean(r["Destination Station Name"]),
                )
                for r in csv.DictReader(file)
            )
            connection.executemany(
                "INSERT INTO stops VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows
            )

        connection.execute(
            """
            INSERT INTO stations
            SELECT station_code, station_name, lower(station_name), MIN(rowid)
            FROM stops
            WHERE station_code != ''
            GROUP BY station_code, station_name
            ORDER BY MIN(rowid)
            """
        )
        connection.commit()
    finally:
        connection.close()

    os.replace(tmp_path, db_path)
    return db_path


def ensure_database(db_path: Path = DB_FILE, csv_path: Path = CSV_FILE) -> Path:
    """Build the database if it is missing or older than the CSV."""

    with _build_lock:
        stale = (
            not db_path.exists()
            or (csv_path.exists() and csv_path.stat().st_mtime > db_path.stat().st_mtime)
        )
        if stale:
            print(f"Building train database at {db_path} ...")
            build_database(csv_path, db_path)
    return db_path


def connect(db_path: Path = DB_FILE) -> sqlite3.Connection:
    """Open a read-only connection that returns rows as dict-like objects."""

    connection = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection


if __name__ == "__main__":
    path = build_database()
    with connect(path) as conn:
        stops = conn.execute("SELECT COUNT(*) FROM stops").fetchone()[0]
        stations = conn.execute("SELECT COUNT(*) FROM stations").fetchone()[0]
    print(f"Built {path} ({stops} stops, {stations} stations, {path.stat().st_size / 2**20:.1f} MB)")
