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

        return route.copy()

    def find_stations(
        self,
        query: str,
    ) -> pd.DataFrame:

        query = query.strip().lower()

        code_match = (
            self.data["Station Code"]
            .str.lower()
            .eq(query)
        )

        name_match = (
            self.data["Station Name"]
            .str.lower()
            .str.contains(
                query,
                na=False,
            )
        )

        results = self.data[
            code_match | name_match
        ][
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
        origin: str,
        destination: str,
    ) -> list[dict[str, Any]]:

        origin = origin.strip().upper()
        destination = destination.strip().upper()

        results = []

        origin_mask = self.data["Station Code"].eq(origin)
        dest_mask = self.data["Station Code"].eq(destination)

        origin_df = self.data[origin_mask]
        dest_df = self.data[dest_mask]

        if origin_df.empty or dest_df.empty:
            return []

        common_trains = pd.merge(
            origin_df,
            dest_df,
            on="Train No",
            suffixes=("_orig", "_dest")
        )

        valid_trains = common_trains[
            common_trains["SEQ_orig"] < common_trains["SEQ_dest"]
        ]

        if valid_trains.empty:
            return []

        for _, row in valid_trains.iterrows():
            train_number = str(row["Train No"]).strip()

            group = self.train_routes.get(train_number)
            if group is None:
                continue

            origin_idx_series = group.index[group["Station Code"].eq(origin)]
            dest_idx_series = group.index[group["Station Code"].eq(destination)]
            
            if origin_idx_series.empty or dest_idx_series.empty:
                continue

            origin_index = origin_idx_series[0]
            destination_index = dest_idx_series[0]

            departure_time = str(row["Departure Time_orig"]).strip()
            arrival_time = str(row["Arrival time_dest"]).strip()

            duration_minutes = calculate_route_duration(
                route=group,
                origin_index=origin_index,
                destination_index=destination_index,
            )

            distance_km = float(row["Distance_dest"]) - float(row["Distance_orig"])

            results.append(
                {
                    "train_number": train_number,
                    "train_name": str(row["Train Name_orig"]).strip(),
                    "origin_code": origin,
                    "origin_name": str(row["Station Name_orig"]).strip(),
                    "destination_code": destination,
                    "destination_name": str(row["Station Name_dest"]).strip(),
                    "departure_time": departure_time,
                    "arrival_time": arrival_time,
                    "duration_minutes": duration_minutes,
                    "distance_km": round(distance_km, 2),
                    "source": "static_schedule_2017",
                }
            )

        return results