from datetime import datetime


TIME_FORMAT = "%H:%M:%S"


def time_to_minutes(time_value: str) -> int | None:
    """Convert HH:MM:SS into minutes after midnight."""

    if not time_value:
        return None

    try:
        parsed = datetime.strptime(
            time_value.strip(),
            TIME_FORMAT,
        )

        return (
            parsed.hour * 60
            + parsed.minute
            + parsed.second // 60
        )

    except ValueError:
        return None


def calculate_route_duration(
    route,
    origin_index: int,
    destination_index: int,
) -> int | None:
    """
    Calculate journey duration between two stops.

    Handles journeys that cross midnight by tracking
    day changes between consecutive stations.
    """

    if origin_index >= destination_index:
        return None

    current_day = 0
    previous_time = None

    absolute_times = []

    for index in range(
        origin_index,
        destination_index + 1,
    ):

        row = route.iloc[index]

        # Use departure time for intermediate stations.
        # At the final destination, arrival time is used.
        if index == destination_index:
            time_value = str(
                row["Arrival time"]
            ).strip()
        else:
            time_value = str(
                row["Departure Time"]
            ).strip()

        current_time = time_to_minutes(
            time_value
        )

        if current_time is None:
            return None

        if (
            previous_time is not None
            and current_time < previous_time
        ):
            current_day += 1

        absolute_time = (
            current_day * 24 * 60
            + current_time
        )

        absolute_times.append(
            absolute_time
        )

        previous_time = current_time

    return (
        absolute_times[-1]
        - absolute_times[0]
    )