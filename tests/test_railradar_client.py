"""RailRadar client tests against a mocked HTTP transport (no network, no quota)."""

import json
from datetime import datetime, timezone

import httpx
import pytest

from backend import railradar_client
from backend.railradar_client import LiveStatusError, RailRadarClient

# Shape taken from the example response in RailRadar's live-status docs.
LIVE_OK = {
    "success": True,
    "data": {
        "trainNumber": "12919",
        "trainName": "Malwa SF Express",
        "startDate": "2026-11-02",
        "lastUpdatedAt": "2026-11-02T14:01:51+05:30",
        "status": "running",
        "isLive": True,
        "trackingMode": "real-time",
        "delayMinutes": 4,
        "train": {
            "number": "12919",
            "name": "Malwa SF Express",
            "source": {"code": "INDB", "name": "Indore Jn"},
            "destination": {"code": "SVDK", "name": "Shri Mata Vaishno Devi Katra"},
        },
        "previousHalt": {"stationCode": "DWX", "stationName": "Dewas", "sequence": 12},
        "nextHalt": {"stationCode": "NAD", "stationName": "Nagda Jn", "sequence": 24},
        "currentLocation": {
            "stationCode": "UJN",
            "stationName": "Ujjain Jn",
            "sequence": 19,
            "status": "at-station",
            "distanceFromOriginKm": 100.6,
            "speedKmh": 0,
            "delayMinutes": 4,
        },
        "route": [{"sequence": 1, "stationCode": "INDB"}],
        "exceptions": [],
    },
    "meta": {"source": "hybrid"},
}


def error_body(code, message="error", details=None):
    error = {"code": code, "message": message}
    if details:
        error["details"] = details
    return {"success": False, "error": error, "meta": {}}


class FakeApi:
    """Records requests and replays queued (status, body, headers) responses."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        status, body, headers = self.responses.pop(0)
        return httpx.Response(status, json=body, headers=headers or {})


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make_client(api: FakeApi, clock=None, now=None, **kwargs) -> RailRadarClient:
    return RailRadarClient(
        api_key="rr_test_key",
        transport=httpx.MockTransport(api),
        clock=clock or Clock(),
        now=now or (lambda: datetime(2026, 11, 2, tzinfo=timezone.utc)),
        **kwargs,
    )


def test_summarises_live_status_and_sends_auth():
    api = FakeApi((200, LIVE_OK, {}))
    result = make_client(api).get_live_status("12919")

    assert result["train_name"] == "Malwa SF Express"
    assert result["delay_minutes"] == 4
    assert result["current_location"]["station_code"] == "UJN"
    assert result["next_halt"] == {"station_code": "NAD", "station_name": "Nagda Jn"}
    assert result["destination_station"]["station_code"] == "SVDK"
    assert "RailRadar" in result["source"]
    assert "route" not in result  # full route is dropped to save tokens
    json.dumps(result)

    request = api.requests[0]
    assert request.url.path == "/v1/trains/12919/live"
    assert request.headers["Authorization"] == "Bearer rr_test_key"


def test_date_is_passed_as_query_param():
    api = FakeApi((200, LIVE_OK, {}))
    make_client(api).get_live_status("12919", "2026-11-02")
    assert api.requests[0].url.params["date"] == "2026-11-02"


def test_cache_avoids_second_request_until_ttl_expires():
    clock = Clock()
    api = FakeApi((200, LIVE_OK, {}), (200, LIVE_OK, {}))
    client = make_client(api, clock=clock, cache_ttl_seconds=300)

    client.get_live_status("12919")
    assert client.get_live_status("12919")["cached"] is True
    assert len(api.requests) == 1

    clock.t += 301
    client.get_live_status("12919")
    assert len(api.requests) == 2


@pytest.mark.parametrize("bad", ["", "12 919", "12919;DROP", "x" * 21])
def test_invalid_train_number_spends_no_request(bad):
    api = FakeApi()
    with pytest.raises(LiveStatusError) as exc:
        make_client(api).get_live_status(bad)
    assert exc.value.code == "invalid_train_number"
    assert api.requests == []


def test_invalid_date_spends_no_request():
    api = FakeApi()
    with pytest.raises(LiveStatusError) as exc:
        make_client(api).get_live_status("12919", "02-11-2026")
    assert exc.value.code == "invalid_date"
    assert api.requests == []


def test_monthly_budget_stops_calls():
    api = FakeApi((200, LIVE_OK, {}), (200, LIVE_OK, {}))
    client = make_client(api, monthly_budget=2, cache_ttl_seconds=0)

    client.get_live_status("12919")
    client.get_live_status("12920")
    with pytest.raises(LiveStatusError) as exc:
        client.get_live_status("12921")
    assert exc.value.code == "live_status_quota_reached"
    assert len(api.requests) == 2


def test_budget_resets_in_a_new_month():
    current = {"now": datetime(2026, 11, 30, tzinfo=timezone.utc)}
    api = FakeApi((200, LIVE_OK, {}), (200, LIVE_OK, {}))
    client = make_client(api, now=lambda: current["now"], monthly_budget=1, cache_ttl_seconds=0)

    client.get_live_status("12919")
    with pytest.raises(LiveStatusError):
        client.get_live_status("12919")

    current["now"] = datetime(2026, 12, 1, tzinfo=timezone.utc)
    client.get_live_status("12919")
    assert len(api.requests) == 2


def test_low_remaining_month_header_stops_further_calls():
    api = FakeApi((200, LIVE_OK, {"X-RateLimit-Remaining-Month": "5"}))
    client = make_client(api, min_remaining=5, cache_ttl_seconds=0)

    client.get_live_status("12919")  # this answer is still returned
    with pytest.raises(LiveStatusError) as exc:
        client.get_live_status("12920")
    assert exc.value.code == "live_status_quota_reached"
    assert len(api.requests) == 1


def test_per_minute_limit_pauses_for_a_minute():
    clock = Clock()
    api = FakeApi(
        (429, error_body("TOO_MANY_REQUESTS"), {"X-RateLimit-Remaining-Min": "0", "X-RateLimit-Remaining-Month": "50"}),
        (200, LIVE_OK, {}),
    )
    client = make_client(api, clock=clock)

    with pytest.raises(LiveStatusError) as exc:
        client.get_live_status("12919")
    assert exc.value.code == "live_status_busy"

    with pytest.raises(LiveStatusError):
        client.get_live_status("12919")
    assert len(api.requests) == 1

    clock.t += 61
    assert client.get_live_status("12919")["delay_minutes"] == 4


def test_monthly_429_blocks_rest_of_month():
    api = FakeApi((429, error_body("TOO_MANY_REQUESTS"), {"X-RateLimit-Remaining-Month": "0"}))
    client = make_client(api)

    for _ in range(2):
        with pytest.raises(LiveStatusError) as exc:
            client.get_live_status("12919")
        assert exc.value.code == "live_status_quota_reached"
    assert len(api.requests) == 1


def test_rejected_key_disables_client():
    api = FakeApi((401, error_body("UNAUTHORIZED"), {}))
    client = make_client(api)

    for _ in range(2):
        with pytest.raises(LiveStatusError) as exc:
            client.get_live_status("12919")
        assert exc.value.code == "live_status_unavailable"
    assert len(api.requests) == 1


def test_not_found_is_cached():
    api = FakeApi((404, error_body("TRAIN_NOT_FOUND"), {}))
    client = make_client(api)

    for _ in range(2):
        with pytest.raises(LiveStatusError) as exc:
            client.get_live_status("99999")
        assert exc.value.code == "train_not_found"
    assert len(api.requests) == 1


def test_not_running_reports_candidate_runs():
    details = {"runDays": ["mon", "thu"], "candidates": [{"startDate": "2026-11-02"}, {"startDate": "2026-11-05"}]}
    api = FakeApi((422, error_body("TRAIN_NOT_RUNNING", details=details), {}))

    with pytest.raises(LiveStatusError) as exc:
        make_client(api).get_live_status("12919", "2026-11-03")
    assert exc.value.code == "train_not_running"
    assert exc.value.details["candidate_runs"] == ["2026-11-02", "2026-11-05"]


def test_server_error_is_not_cached():
    api = FakeApi((503, error_body("SERVICE_UNAVAILABLE"), {}), (200, LIVE_OK, {}))
    client = make_client(api)

    with pytest.raises(LiveStatusError) as exc:
        client.get_live_status("12919")
    assert exc.value.code == "live_status_unreachable"
    assert client.get_live_status("12919")["delay_minutes"] == 4


def test_network_failure_returns_friendly_error():
    def broken(request):
        raise httpx.ConnectError("boom", request=request)

    client = RailRadarClient(api_key="k", transport=httpx.MockTransport(broken))
    with pytest.raises(LiveStatusError) as exc:
        client.get_live_status("12919")
    assert exc.value.code == "live_status_unreachable"


def test_tool_reports_not_configured_without_key(monkeypatch):
    monkeypatch.setattr(railradar_client, "get_railradar_client", lambda: None)
    result = railradar_client.get_live_train_status("12919")
    assert result["error"] == "live_status_not_configured"


def test_live_status_tool_returns_json(monkeypatch):
    from backend import tools

    monkeypatch.setattr(tools, "get_live_train_status", lambda number, date=None: {"train_number": number})
    output = tools.live_train_status.invoke({"train_number": "12302"})
    assert json.loads(output) == {"train_number": "12302"}
