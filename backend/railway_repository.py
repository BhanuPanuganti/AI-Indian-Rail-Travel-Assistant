from typing import Any

try:
    from backend.time_utils import calculate_route_duration
except ModuleNotFoundError:
    from time_utils import calculate_route_duration

import pandas as pd

try:
    from backend.data_loader import load_train_data
except ModuleNotFoundError:
    # Allow running backend scripts directly from the backend directory.
    from data_loader import load_train_data


class RailwayRepository:

    def __init__(self):
        self._data: pd.DataFrame | None = None
        self._train_routes: dict[str, pd.DataFrame] | None = None

    @property
    def data(self) -> pd.DataFrame:

        if self._data is None:
            self._data = load_train_data()
            self._data["Station Code"] = self._data["Station Code"].astype(str).str.strip().str.upper()
            self._data["Station Name"] = self._data["Station Name"].astype(str).str.strip()

        return self._data

    @property
    def train_routes(self) -> dict[str, pd.DataFrame]:

        if self._train_routes is None:

            self._train_routes = {
                str(train_number).strip(): group.sort_values(
                    "SEQ"
                ).reset_index(drop=True)
                for train_number, group
                in self.data.groupby(
                    "Train No",
                    sort=False,
                )
            }

        return self._train_routes

    def find_train_route(
        self,
        train_number: str,
    ) -> pd.DataFrame:

        train_number = str(
            train_number
        ).strip()

        route = self.train_routes.get(
            train_number
        )

        if route is None:
            return pd.DataFrame()

        # Replace NaN (e.g. missing distances) with None so the result can be
        # serialised as strict JSON.
        route = route.astype(object)
        return route.where(route.notna(), None)

    def find_stations(
        self,
        query: str,
    ) -> pd.DataFrame:

        query = query.strip().lower()

        empty = pd.DataFrame(columns=["Station Code", "Station Name"])

        if not query:
            return empty

        code_match = (
            self.data["Station Code"]
            .str.lower()
            .eq(query)
        )

        # Substring name matching only for queries long enough to be
        # meaningful; very short queries ("a", "ne") would match thousands
        # of stations. Exact station-code matches always apply.
        if len(query) >= MIN_NAME_QUERY_LENGTH:
            name_match = (
                self.data["Station Name"]
                .str.lower()
                .str.contains(
                    query,
                    na=False,
                    regex=False,
                )
            )
            mask = code_match | name_match
        else:
            mask = code_match

        results = self.data[mask][
            [
                "Station Code",
                "Station Name",
            ]
        ].copy()

        return (
            results
            .drop_duplicates()
            .reset_index(drop=True)
        )

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

        origin_df = self.data[self.data["Station Code"].isin(origin_codes)]
        dest_df = self.data[self.data["Station Code"].isin(destination_codes)]

        if origin_df.empty or dest_df.empty:
            return []

        common_trains = pd.merge(
            origin_df,
            dest_df,
            on="Train No",
            suffixes=("_orig", "_dest")
        )

        valid_trains = common_trains[
            (common_trains["SEQ_orig"] < common_trains["SEQ_dest"])
            & (common_trains["Station Code_orig"] != common_trains["Station Code_dest"])
        ]

        if valid_trains.empty:
            return []

        best_by_train: dict[str, dict[str, Any]] = {}

        for row in valid_trains.to_dict(orient="records"):
            train_number = str(row["Train No"]).strip()

            group = self.train_routes.get(train_number)
            if group is None:
                continue

            origin_positions = group.index[group["SEQ"].eq(row["SEQ_orig"])]
            dest_positions = group.index[group["SEQ"].eq(row["SEQ_dest"])]

            if origin_positions.empty or dest_positions.empty:
                continue

            duration_minutes = calculate_route_duration(
                route=group,
                origin_index=origin_positions[0],
                destination_index=dest_positions[0],
            )

            distance_km = _safe_distance(row["Distance_orig"], row["Distance_dest"])

            candidate = {
                "train_number": train_number,
                "train_name": str(row["Train Name_orig"]).strip(),
                "origin_code": row["Station Code_orig"],
                "origin_name": str(row["Station Name_orig"]).strip(),
                "destination_code": row["Station Code_dest"],
                "destination_name": str(row["Station Name_dest"]).strip(),
                "departure_time": str(row["Departure Time_orig"]).strip(),
                "arrival_time": str(row["Arrival time_dest"]).strip(),
                "duration_minutes": duration_minutes,
                "distance_km": distance_km,
                "source": "static_schedule_2017",
            }

            existing = best_by_train.get(train_number)
            if existing is None or _duration_key(candidate) < _duration_key(existing):
                best_by_train[train_number] = candidate

        return list(best_by_train.values())


MIN_NAME_QUERY_LENGTH = 3


def _normalize_codes(codes: str | list[str]) -> list[str]:
    if isinstance(codes, str):
        codes = [codes]
    return [c.strip().upper() for c in codes if c and c.strip()]


def _safe_distance(start, end) -> float | None:
    if pd.isna(start) or pd.isna(end):
        return None
    return round(float(end) - float(start), 2)


def _duration_key(train: dict[str, Any]) -> float:
    duration = train.get("duration_minutes")
    return float("inf") if duration is None else duration
