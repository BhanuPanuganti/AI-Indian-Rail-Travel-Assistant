"""
Optional live train running status from the RailRadar API.

RailRadar's free plan allows only 100 requests a month, and every call counts
(errors included), so this client is deliberately defensive:

- results are cached per (train, date) for a few minutes, including
  "not found" / "not running" answers;
- train numbers are validated locally before spending a request;
- a monthly request budget is enforced in-process, and RailRadar's own
  X-RateLimit-Remaining-* headers stop calls before the quota is gone;
- an invalid or revoked key, or an exhausted quota, disables further calls
  instead of failing on every request.

The in-process counter resets when the backend restarts; the rate-limit
headers from the first call after a restart restore the real picture.
"""

import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable

import httpx

_TRAIN_NUMBER = re.compile(r"^[0-9A-Za-z]{1,20}$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

SOURCE_LABEL = "RailRadar live status (crowdsourced; may be delayed or inaccurate)"


class LiveStatusError(Exception):
    """A live-status lookup that produced no data, with a user-facing reason."""

    def __init__(self, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        payload = {"error": self.code, "message": self.message}
        if self.details:
            payload["details"] = self.details
        return payload


def _month_key(now: datetime) -> str:
    return now.strftime("%Y-%m")


class RailRadarClient:

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.railradar.in",
        monthly_budget: int = 90,
        min_remaining: int = 5,
        cache_ttl_seconds: int = 300,
        timeout_seconds: float = 10.0,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_seconds,
            transport=transport,
        )
        self._monthly_budget = monthly_budget
        self._min_remaining = min_remaining
        self._cache_ttl = cache_ttl_seconds
        self._clock = clock
        self._now = now

        self._lock = threading.Lock()
        self._cache: dict[tuple[str, str], tuple[float, dict | LiveStatusError]] = {}
        self._month = _month_key(now())
        self._requests_this_month = 0
        self._blocked_until_month: str | None = None   # quota exhausted
        self._paused_until: float = 0.0                 # per-minute limit hit
        self._disabled_reason: str | None = None        # bad or revoked key

    # -- public -------------------------------------------------------------

    @property
    def requests_this_month(self) -> int:
        return self._requests_this_month

    def get_live_status(self, train_number: str, date: str | None = None) -> dict[str, Any]:
        """Return a compact live-status summary, or raise LiveStatusError."""

        train_number = str(train_number).strip().upper()
        date = (date or "").strip() or None

        if not _TRAIN_NUMBER.match(train_number):
            raise LiveStatusError(
                "invalid_train_number",
                f"'{train_number}' is not a valid train number.",
            )
        if date and not _DATE.match(date):
            raise LiveStatusError("invalid_date", "Dates must be in YYYY-MM-DD format.")

        key = (train_number, date or "")

        with self._lock:
            cached = self._cache.get(key)
            if cached and cached[0] > self._clock():
                result = cached[1]
                if isinstance(result, LiveStatusError):
                    raise result
                return {**result, "cached": True}

            self._check_allowed()
            self._requests_this_month += 1

        params = {"date": date} if date else None
        try:
            response = self._http.get(f"/v1/trains/{train_number}/live", params=params)
        except httpx.HTTPError as exc:
            # Network failures are not cached: they say nothing about the train.
            raise LiveStatusError(
                "live_status_unreachable",
                "The live status service could not be reached. Please try again shortly.",
            ) from exc

        with self._lock:
            self._apply_rate_limit_headers(response)
            try:
                result: dict | LiveStatusError = self._parse(response, train_number)
            except LiveStatusError as error:
                result = error

            if not isinstance(result, LiveStatusError) or result.code in ("train_not_found", "train_not_running"):
                self._cache[key] = (self._clock() + self._cache_ttl, result)

        if isinstance(result, LiveStatusError):
            raise result
        return result

    # -- guards ---------------------------------------------------------------

    def _check_allowed(self) -> None:
        month = _month_key(self._now())
        if month != self._month:
            self._month = month
            self._requests_this_month = 0
            self._blocked_until_month = None

        if self._disabled_reason:
            raise LiveStatusError("live_status_unavailable", self._disabled_reason)

        if self._blocked_until_month == month or self._requests_this_month >= self._monthly_budget:
            raise LiveStatusError(
                "live_status_quota_reached",
                "The monthly limit for live train status has been reached. "
                "Live status will be available again next month; timetable data still works.",
            )

        if self._clock() < self._paused_until:
            raise LiveStatusError(
                "live_status_busy",
                "Live status is receiving too many requests right now. Please try again in a minute.",
            )

    def _apply_rate_limit_headers(self, response: httpx.Response) -> None:
        remaining_month = _int_header(response, "X-RateLimit-Remaining-Month")
        remaining_minute = _int_header(response, "X-RateLimit-Remaining-Min")

        if remaining_month is not None and remaining_month <= self._min_remaining:
            self._blocked_until_month = self._month
        if remaining_minute == 0:
            self._paused_until = self._clock() + 60

    # -- parsing --------------------------------------------------------------

    def _parse(self, response: httpx.Response, train_number: str) -> dict[str, Any]:
        try:
            body = response.json()
        except ValueError:
            body = {}

        if response.is_success and body.get("success", True) and isinstance(body.get("data"), dict):
            return _summarise(body["data"], train_number)

        error = body.get("error") or {}
        code = error.get("code", "")
        details = error.get("details") or {}
        status = response.status_code

        if status == 404:
            raise LiveStatusError(
                "train_not_found",
                f"RailRadar has no live data for train {train_number}. Check the train number.",
            )
        if status == 422 or code == "TRAIN_NOT_RUNNING":
            raise LiveStatusError(
                "train_not_running",
                f"Train {train_number} does not run on the requested date.",
                {
                    "run_days": details.get("runDays"),
                    "candidate_runs": [
                        c.get("startDate") or c.get("date") or c
                        for c in (details.get("candidates") or [])
                    ][:4],
                },
            )
        if status == 403 and code == "HISTORICAL_NOT_INCLUDED":
            raise LiveStatusError(
                "date_too_old",
                "Live status is only available for runs in the last 30 days.",
            )
        if status in (401, 403):
            self._disabled_reason = (
                "Live train status is unavailable because the RailRadar API key "
                "was rejected. Timetable data still works."
            )
            raise LiveStatusError("live_status_unavailable", self._disabled_reason)
        if status == 429:
            if _int_header(response, "X-RateLimit-Remaining-Min") == 0 and _int_header(
                response, "X-RateLimit-Remaining-Month"
            ) not in (0, None):
                self._paused_until = self._clock() + 60
                raise LiveStatusError(
                    "live_status_busy",
                    "Live status is receiving too many requests right now. Please try again in a minute.",
                )
            self._blocked_until_month = self._month
            raise LiveStatusError(
                "live_status_quota_reached",
                "The monthly limit for live train status has been reached. "
                "Live status will be available again next month; timetable data still works.",
            )
        if status == 400:
            raise LiveStatusError("invalid_request", error.get("message") or "Invalid live status request.")

        raise LiveStatusError(
            "live_status_unreachable",
            "The live status service had a problem. Please try again shortly.",
        )


def _int_header(response: httpx.Response, name: str) -> int | None:
    value = response.headers.get(name)
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _halt(halt: Any) -> dict | None:
    if not isinstance(halt, dict):
        return None
    return {
        "station_code": halt.get("stationCode"),
        "station_name": halt.get("stationName"),
    }


def _summarise(data: dict, train_number: str) -> dict[str, Any]:
    """Keep only what the assistant needs, to save LLM tokens."""

    location = data.get("currentLocation") or {}
    train = data.get("train") or {}

    return {
        "train_number": data.get("trainNumber") or train_number,
        "train_name": data.get("trainName") or train.get("name"),
        "journey_start_date": data.get("startDate"),
        "status": data.get("status"),
        "is_live": data.get("isLive"),
        "delay_minutes": data.get("delayMinutes", location.get("delayMinutes")),
        "last_updated_at": data.get("lastUpdatedAt"),
        "current_location": {
            "station_code": location.get("stationCode"),
            "station_name": location.get("stationName"),
            "status": location.get("status"),
            "distance_from_origin_km": location.get("distanceFromOriginKm"),
            "speed_kmh": location.get("speedKmh"),
        } if location else None,
        "previous_halt": _halt(data.get("previousHalt")),
        "next_halt": _halt(data.get("nextHalt")),
        "source_station": _halt({"stationCode": (train.get("source") or {}).get("code"),
                                 "stationName": (train.get("source") or {}).get("name")}),
        "destination_station": _halt({"stationCode": (train.get("destination") or {}).get("code"),
                                      "stationName": (train.get("destination") or {}).get("name")}),
        "exceptions": data.get("exceptions") or [],
        "source": SOURCE_LABEL,
        "cached": False,
    }


_client: RailRadarClient | None = None
_client_lock = threading.Lock()


def get_railradar_client() -> RailRadarClient | None:
    """Shared client, or None when RAILRADAR_API_KEY is not configured."""

    global _client

    try:
        from backend.config import settings
    except ModuleNotFoundError:
        from config import settings

    if not settings.railradar_api_key:
        return None

    with _client_lock:
        if _client is None:
            _client = RailRadarClient(
                api_key=settings.railradar_api_key,
                base_url=settings.railradar_base_url,
                monthly_budget=settings.railradar_monthly_budget,
                min_remaining=settings.railradar_min_remaining,
                cache_ttl_seconds=settings.railradar_cache_ttl_seconds,
                timeout_seconds=settings.railradar_timeout_seconds,
            )
    return _client


def get_live_train_status(train_number: str, date: str | None = None) -> dict[str, Any]:
    """Live status summary, or an {"error", "message"} dict. Never raises."""

    client = get_railradar_client()
    if client is None:
        return {
            "error": "live_status_not_configured",
            "message": (
                "Live running status is not enabled on this deployment. "
                "I can still show the scheduled timetable for this train."
            ),
        }
    try:
        return client.get_live_status(train_number, date)
    except LiveStatusError as error:
        return error.to_dict()
