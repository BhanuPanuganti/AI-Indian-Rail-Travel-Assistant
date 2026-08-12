def format_duration(
    minutes: int | None,
) -> str | None:

    if minutes is None:
        return None

    days = minutes // (24 * 60)

    remaining = minutes % (24 * 60)

    hours = remaining // 60

    mins = remaining % 60

    parts = []

    if days:
        parts.append(
            f"{days}d"
        )

    if hours:
        parts.append(
            f"{hours}h"
        )

    if mins:
        parts.append(
            f"{mins}m"
        )

    return " ".join(parts) or "0m"