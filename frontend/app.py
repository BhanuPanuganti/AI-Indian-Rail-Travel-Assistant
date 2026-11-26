"""
Streamlit frontend for the AI Indian Rail Travel Assistant.

Layout: a header, then two columns. The left column holds the train search
(2017 timetable) and live running status (RailRadar) panels; the right
column holds the chat with the AI assistant. Every chat answer is labelled
with where it came from (timetable, official documents, or live data).
"""

import html
import os
import re
import uuid
from datetime import date, datetime, timedelta

import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Indian Rail Travel Assistant",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
CHAT_TIMEOUT = 120
SEARCH_TIMEOUT = 30
WAKE_TIMEOUT = 90

PREFERENCES = {
    "balanced": "Best overall",
    "fastest": "Fastest",
    "earliest": "Earliest",
    "shortest_distance": "Shortest",
}

SUGGESTIONS = [
    ("Tatkal timings", "What time does Tatkal booking open for AC and non-AC classes?"),
    ("Cancellation charges", "How much is deducted if I cancel a confirmed ticket 10 hours before departure?"),
    ("Luggage allowance", "How much luggage can I carry free in sleeper class?"),
    ("Delhi → Mumbai trains", "Show me trains from Delhi to Mumbai"),
]

FOLLOW_UPS = [
    ("Fastest", "Which train is the fastest?"),
    ("Earliest", "Which train departs earliest?"),
    ("Route", "Show me the route of the recommended train"),
    ("Compare", "Compare the top 2-3 trains for me"),
    ("Live status", "What is the live running status of the recommended train?"),
]

DOCUMENT_NAMES = {
    "irctc_eticket_faq.pdf": "IRCTC e-ticket FAQ",
    "irctc_terms_conditions.pdf": "IRCTC terms & conditions",
    "irctc_cancellation_refund_rules.pdf": "IRCTC cancellation & refund rules",
    "indian_railways_luggage_rules.pdf": "Indian Railways luggage rules",
    "indian_railways_concession_rules.pdf": "Indian Railways concession & refund rules",
    "irctc_tatkal_faq.pdf": "IRCTC Tatkal FAQ",
}

TIMETABLE_TOOLS = {"train_search", "train_details", "station_search"}

# ─────────────────────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --navy: #0E2A47; --navy-2: #163A61; --ink: #16202E; --muted: #5B6779;
  --line: #E1E6EE; --surface: #FFFFFF; --accent: #D9480F; --accent-soft: #FFF1EA;
  --ok: #1F7A45; --ok-soft: #E7F5EC; --warn: #9A6700; --warn-soft: #FFF6DD;
  --bad: #B42318; --bad-soft: #FDECEA; --info-soft: #EAF2FB;
}
html, body, .stApp, button, input, textarea { font-family: 'Inter', sans-serif; }
[data-testid="stIconMaterial"], .material-symbols-outlined, .material-icons, [class*="material-symbols"] {
  font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1440px; }

/* Header */
.hero { background: linear-gradient(120deg, var(--navy) 0%, var(--navy-2) 100%); border-radius: 16px;
  padding: 22px 28px; color: #fff; display: flex; justify-content: space-between; align-items: center;
  gap: 18px; flex-wrap: wrap; margin-bottom: 18px; }
.hero-title { font-size: 26px; font-weight: 700; letter-spacing: -0.3px; margin: 0; }
.hero-sub { color: #C9D6E6; font-size: 14.5px; margin-top: 4px; }
.chips { display: flex; gap: 8px; flex-wrap: wrap; }
.chip { background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); color: #EAF1F8;
  padding: 6px 12px; border-radius: 999px; font-size: 12.5px; font-weight: 500; white-space: nowrap; }
.chip b { color: #FFB38A; font-weight: 600; }

/* Section headings */
.section-title { font-size: 17px; font-weight: 700; color: var(--ink); margin: 2px 0 2px 0; }
.section-sub { font-size: 13px; color: var(--muted); margin-bottom: 10px; }

/* Result summary */
.summary { display: flex; justify-content: space-between; align-items: baseline; margin: 14px 0 8px 0; }
.summary-main { font-size: 15px; font-weight: 600; color: var(--ink); }
.summary-note { font-size: 12px; color: var(--muted); }

/* Ticket-style train card */
.tcard { background: var(--surface); border: 1px solid var(--line); border-radius: 14px; padding: 16px 18px; margin-bottom: 10px; }
.tcard.rec { border: 1.5px solid var(--accent); box-shadow: 0 4px 18px rgba(217,72,15,0.08); }
.tcard-top { display: flex; justify-content: space-between; align-items: center; gap: 10px; margin-bottom: 12px; flex-wrap: wrap; }
.tno { font-weight: 700; color: var(--navy); background: #EEF3F9; border-radius: 6px; padding: 2px 8px; font-size: 13px; margin-right: 8px; }
.tname { font-weight: 600; color: var(--ink); font-size: 15px; }
.badge { background: var(--accent-soft); color: var(--accent); font-size: 12px; font-weight: 600; padding: 4px 10px; border-radius: 999px; }
.legs { display: grid; grid-template-columns: 1fr 1.3fr 1fr; align-items: center; gap: 10px; }
.stn .time { font-size: 24px; font-weight: 700; color: var(--ink); line-height: 1.1; }
.stn .time sup { font-size: 11px; color: var(--accent); font-weight: 700; margin-left: 2px; }
.stn .code { font-size: 12px; font-weight: 700; color: var(--navy); margin-top: 3px; }
.stn .sname { font-size: 12px; color: var(--muted); }
.stn.right { text-align: right; }
.track { text-align: center; }
.track .dur { font-size: 13px; font-weight: 600; color: var(--ink); }
.track .bar { height: 2px; background: repeating-linear-gradient(90deg, var(--line) 0 6px, transparent 6px 10px); margin: 6px 0; position: relative; }
.track .bar:before, .track .bar:after { content: ""; position: absolute; top: -3px; width: 8px; height: 8px; border-radius: 50%; background: var(--navy); }
.track .bar:before { left: 0; } .track .bar:after { right: 0; background: var(--accent); }
.track .dist { font-size: 12px; color: var(--muted); }

/* Compact rows for other trains */
.trow { display: grid; grid-template-columns: 2.2fr 1.3fr 0.8fr 0.7fr; gap: 8px; align-items: center;
  padding: 10px 12px; border-bottom: 1px solid var(--line); font-size: 13.5px; background: var(--surface); }
.trow:first-child { border-top-left-radius: 12px; border-top-right-radius: 12px; }
.trow:last-child { border-bottom: none; border-bottom-left-radius: 12px; border-bottom-right-radius: 12px; }
.trows { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
.trow .tname { font-size: 13.5px; }
.trow .t { font-weight: 600; color: var(--ink); }
.trow .t sup { color: var(--accent); font-size: 10px; }
.trow .m { color: var(--muted); text-align: right; }
.trow .codes { font-size: 11.5px; font-weight: 500; color: var(--muted); margin-top: 1px; }
.trow.head { background: #F0F3F7; font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.4px; color: var(--muted); font-weight: 600; }

/* Live status card */
.live { background: var(--surface); border: 1px solid var(--line); border-radius: 14px; padding: 16px 18px; margin-top: 12px; }
.pill { font-size: 12.5px; font-weight: 700; padding: 4px 10px; border-radius: 999px; }
.pill.ok { background: var(--ok-soft); color: var(--ok); }
.pill.warn { background: var(--warn-soft); color: var(--warn); }
.pill.bad { background: var(--bad-soft); color: var(--bad); }
.pill.na { background: #EEF1F5; color: var(--muted); }
.live-now .label { font-size: 11.5px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.4px; font-weight: 600; }
.live-now .big { font-size: 22px; font-weight: 700; color: var(--ink); margin: 2px 0 12px 0; }
.halts { display: grid; grid-template-columns: 1fr auto 1fr; gap: 10px; align-items: center; background: #F6F8FB; border-radius: 10px; padding: 10px 12px; }
.halts .h-label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.4px; }
.halts .h-name { font-size: 14px; font-weight: 600; color: var(--ink); }
.halts .arrow { color: var(--accent); font-weight: 700; }
.halts .right { text-align: right; }
.live-meta { font-size: 12px; color: var(--muted); margin-top: 10px; }

/* Chat */
.welcome { text-align: center; padding: 26px 10px 8px 10px; }
.welcome .w-title { font-size: 19px; font-weight: 700; color: var(--ink); }
.welcome .w-sub { font-size: 13.5px; color: var(--muted); margin-top: 4px; }
.src-label { display: inline-flex; align-items: center; gap: 6px; font-size: 11.5px; font-weight: 600; padding: 3px 9px;
  border-radius: 999px; margin-top: 4px; }
.src-label.timetable { background: #EEF3F9; color: var(--navy); }
.src-label.docs { background: var(--info-soft); color: #1F5C99; }
.src-label.live { background: var(--accent-soft); color: var(--accent); }
.sources { border-top: 1px solid var(--line); margin-top: 10px; padding-top: 8px; }
.sources .s-title { font-size: 11.5px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.4px; margin-bottom: 4px; }
.sources .s-item { font-size: 12.5px; color: #1F5C99; margin: 2px 0; }
.sources .s-item span { color: var(--muted); }
.notice { border-radius: 10px; padding: 10px 12px; font-size: 13.5px; }
.notice.error { background: var(--bad-soft); color: var(--bad); }

/* Pill-style buttons: suggestions and follow-ups */
[class*="st-key-chip_"] button, [class*="st-key-fu_"] button {
  border-radius: 999px !important; border: 1px solid var(--line) !important; background: var(--surface) !important;
  color: var(--ink) !important; font-size: 13px !important; font-weight: 500 !important; padding: 4px 12px !important;
  min-height: 0 !important;
}
[class*="st-key-chip_"] button:hover, [class*="st-key-fu_"] button:hover {
  border-color: var(--accent) !important; color: var(--accent) !important;
}
.st-key-swap button { margin-top: 28px; border-radius: 999px !important; }
.fu-label { font-size: 11.5px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.4px; margin: 10px 0 4px 0; }
/* Phones: columns stack, so the swap button would become a full-width bar */
@media (max-width: 640px) {
  .st-key-swap { display: none; }
  .hero { padding: 18px 18px; }
  .hero-title { font-size: 21px; }
  .stn .time { font-size: 20px; }
  .trow { grid-template-columns: 1.8fr 1.4fr 0.9fr; }
  .trow > div:last-child { display: none; }
}
.footer { text-align: center; color: var(--muted); font-size: 12px; margin-top: 22px; }
.footer a { color: var(--muted); }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────
defaults = {
    "thread_id": str(uuid.uuid4()),
    "messages": [],
    "pending_prompt": None,
    "search": None,
    "live": None,
    "backend_ready": False,
    "from_input": "",
    "to_input": "",
    "pref": "balanced",
    "left_view": "search",
    "last_view": "search",
    "live_number": "",
    "live_date": None,
    "run_count": 0,
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)

# Streamlit drops the state of widgets that are not drawn in a run. Only one
# of the two left-hand views is drawn at a time, so re-assign their inputs to
# keep what the user typed when they switch views.
for key in ("from_input", "to_input", "pref", "live_number", "live_date"):
    st.session_state[key] = st.session_state[key]

st.session_state.run_count += 1


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def esc(value) -> str:
    return html.escape("" if value is None else str(value))


def hhmm(value: str | None) -> str:
    """'16:50:00' -> '16:50'."""
    if not value:
        return "—"
    match = re.match(r"^(\d{1,2}):(\d{2})", value.strip())
    return f"{int(match.group(1)):02d}:{match.group(2)}" if match else value


def day_offset(departure: str | None, minutes: int | None) -> int:
    """How many days after departure the train arrives (0 = same day)."""
    match = re.match(r"^(\d{1,2}):(\d{2})", departure or "")
    if not match or minutes is None:
        return 0
    start = int(match.group(1)) * 60 + int(match.group(2))
    return (start + minutes) // (24 * 60)


def api_get(path: str, timeout: int = SEARCH_TIMEOUT, **params):
    response = requests.get(f"{API_URL}{path}", params=params or None, timeout=timeout)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict, timeout: int = SEARCH_TIMEOUT):
    response = requests.post(f"{API_URL}{path}", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def friendly_error(err: Exception) -> str:
    if isinstance(err, requests.Timeout):
        return "The server took too long to respond. Please try again."
    if isinstance(err, requests.ConnectionError):
        return "Cannot reach the backend server. It may be starting up — please try again in a minute."
    if isinstance(err, requests.HTTPError):
        return f"The server returned an error (HTTP {err.response.status_code}). Please try again."
    return f"Something went wrong: {err}"


def wake_backend() -> None:
    """The free backend sleeps when idle; wake it once per session with a visible message."""
    if st.session_state.backend_ready:
        return
    with st.spinner("Waking up the server — the free hosting plan sleeps when idle, this can take up to a minute…"):
        try:
            api_get("/health", timeout=WAKE_TIMEOUT)
            st.session_state.backend_ready = True
        except Exception:
            st.warning("The backend is not responding yet. Searches and chat will retry when you use them.")


def search_context(data: dict) -> str:
    """Plain-text summary of search results, sent with follow-up questions."""
    lines = [f"Train search results: {data['origin']} to {data['destination']}"]
    rec = data.get("recommended")
    if rec:
        lines.append(
            f"Recommended: #{rec.get('train_number')} {rec.get('train_name')} | Departs {rec.get('departure_time')} "
            f"| Arrives {rec.get('arrival_time')} | Duration {rec.get('duration')} | {rec.get('distance_km')} km"
        )
    for t in data.get("trains", [])[:6]:
        lines.append(
            f"- #{t.get('train_number')} {t.get('train_name')} | Dep {t.get('departure_time')} "
            f"| Arr {t.get('arrival_time')} | {t.get('duration')} | {t.get('distance_km')} km"
        )
    return "\n".join(lines)


def track_live(train_number: str) -> None:
    """Open the live status view with this train filled in (no lookup until the user asks)."""
    st.session_state.left_view = "live"
    st.session_state.last_view = "live"
    st.session_state.live_number = train_number
    st.session_state.live = None


def keep_view() -> None:
    """Clicking the selected view again would clear the switcher; keep the last view instead."""
    if st.session_state.left_view is None:
        st.session_state.left_view = st.session_state.last_view
    st.session_state.last_view = st.session_state.left_view


def ask(prompt: str, context: str | None = None) -> None:
    st.session_state.pending_prompt = f"{prompt}\n\n[Context from my search]\n{context}" if context else prompt


# ─────────────────────────────────────────────────────────────────────────────
# Rendering: trains
# ─────────────────────────────────────────────────────────────────────────────
def train_card_html(train: dict, badge: str | None = None) -> str:
    offset = day_offset(train.get("departure_time"), train.get("duration_minutes"))
    plus = f"<sup>+{offset}</sup>" if offset else ""
    distance = f"{train['distance_km']:g} km" if train.get("distance_km") is not None else "distance n/a"
    return (
        f'<div class="tcard{" rec" if badge else ""}">'
        f'<div class="tcard-top"><div><span class="tno">{esc(train.get("train_number"))}</span>'
        f'<span class="tname">{esc(train.get("train_name"))}</span></div>'
        f'{f"<span class=badge>{esc(badge)}</span>" if badge else ""}</div>'
        f'<div class="legs">'
        f'<div class="stn"><div class="time">{esc(hhmm(train.get("departure_time")))}</div>'
        f'<div class="code">{esc(train.get("origin_code"))}</div><div class="sname">{esc(train.get("origin_name"))}</div></div>'
        f'<div class="track"><div class="dur">{esc(train.get("duration") or "—")}</div><div class="bar"></div>'
        f'<div class="dist">{esc(distance)}</div></div>'
        f'<div class="stn right"><div class="time">{esc(hhmm(train.get("arrival_time")))}{plus}</div>'
        f'<div class="code">{esc(train.get("destination_code"))}</div><div class="sname">{esc(train.get("destination_name"))}</div></div>'
        f"</div></div>"
    )


def train_rows_html(trains: list[dict]) -> str:
    rows = ['<div class="trow head"><div>Train</div><div>Departs → Arrives</div><div class="m">Duration</div><div class="m">Distance</div></div>']
    for t in trains:
        offset = day_offset(t.get("departure_time"), t.get("duration_minutes"))
        plus = f"<sup>+{offset}</sup>" if offset else ""
        distance = f"{t['distance_km']:g} km" if t.get("distance_km") is not None else "—"
        rows.append(
            f'<div class="trow"><div><span class="tno">{esc(t.get("train_number"))}</span>'
            f'<span class="tname">{esc(t.get("train_name"))}</span></div>'
            f'<div class="t">{esc(hhmm(t.get("departure_time")))} → {esc(hhmm(t.get("arrival_time")))}{plus}'
            f'<div class="codes">{esc(t.get("origin_code"))} → {esc(t.get("destination_code"))}</div></div>'
            f'<div class="m">{esc(t.get("duration") or "—")}</div><div class="m">{esc(distance)}</div></div>'
        )
    return '<div class="trows">' + "".join(rows) + "</div>"


def render_search_results(data: dict) -> None:
    trains = data.get("trains", [])
    if not trains:
        st.info(
            f"No direct trains found between **{data['origin']}** and **{data['destination']}** in the timetable. "
            "Try a station code (for example HYB, MAS, NDLS) or a nearby major station."
        )
        return

    rec = data.get("recommended") or trains[0]
    label = PREFERENCES.get(data.get("preference"), "Best overall")
    st.markdown(
        f'<div class="summary"><div class="summary-main">{len(trains)} direct train{"s" if len(trains) != 1 else ""} · '
        f'{esc(data["origin"])} → {esc(data["destination"])}</div>'
        f'<div class="summary-note">Sorted by {esc(label.lower())} · 2017 timetable</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(train_card_html(rec, badge=f"Recommended · {label.lower()}"), unsafe_allow_html=True)
    st.button(
        f"Track {rec.get('train_number')} live",
        key="track_live",
        icon=":material/my_location:",
        on_click=track_live,
        args=(str(rec.get("train_number")),),
    )

    others = [t for t in trains if t.get("train_number") != rec.get("train_number")]
    if others:
        st.markdown(train_rows_html(others[:8]), unsafe_allow_html=True)
        if len(others) > 8:
            with st.expander(f"{len(others) - 8} more trains"):
                st.markdown(train_rows_html(others[8:]), unsafe_allow_html=True)

    st.markdown('<div class="fu-label">Ask the assistant about these trains</div>', unsafe_allow_html=True)
    cols = st.columns(len(FOLLOW_UPS))
    for i, (label, prompt) in enumerate(FOLLOW_UPS):
        with cols[i]:
            if st.button(label, key=f"fu_search_{i}", use_container_width=True):
                ask(prompt, search_context(data))
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Rendering: live status
# ─────────────────────────────────────────────────────────────────────────────
def delay_pill(minutes) -> str:
    if minutes is None:
        return '<span class="pill na">Delay unknown</span>'
    if minutes <= 5:
        return f'<span class="pill ok">On time{f" · {minutes} min" if minutes > 0 else ""}</span>'
    css = "warn" if minutes <= 30 else "bad"
    return f'<span class="pill {css}">{minutes} min late</span>'


def format_ist(value: str | None) -> str:
    if not value:
        return "unknown"
    try:
        return datetime.fromisoformat(value).strftime("%d %b, %H:%M IST")
    except ValueError:
        return value


def render_live(result: dict) -> None:
    if "error" in result:
        code = result.get("error")
        message = result.get("message", "Live status is unavailable right now.")
        if code == "train_not_running":
            runs = [r for r in (result.get("details") or {}).get("candidate_runs") or [] if isinstance(r, str)]
            hint = f" Runs you can check: {', '.join(runs)}." if runs else ""
            st.warning(message + hint)
        elif code in ("live_status_not_configured", "live_status_quota_reached", "live_status_busy"):
            st.info(message)
        else:
            st.warning(message)
        return

    live = result["live"]
    location = live.get("current_location") or {}
    where = location.get("station_name")
    status = (live.get("status") or "").replace("-", " ").strip()
    finished = status.lower() in ("completed", "arrived", "terminated")
    if where and finished:
        now_text = f"Arrived at {where}"
    elif where:
        prefix = "At" if location.get("status") == "at-station" else "Near"
        now_text = f"{prefix} {where}"
    else:
        now_text = status.capitalize() or "Position not available"

    prev_halt = (live.get("previous_halt") or {}).get("station_name") or "—"
    next_halt = (live.get("next_halt") or {}).get("station_name") or "—"
    meta = [f"Updated {format_ist(live.get('last_updated_at'))}"]
    if live.get("journey_start_date"):
        meta.append(f"journey started {esc(live['journey_start_date'])}")
    if location.get("distance_from_origin_km") is not None:
        meta.append(f"{location['distance_from_origin_km']:g} km from origin")
    if status:
        meta.append(f"status: {esc(status)}")
    if live.get("cached"):
        meta.append("cached result")

    st.markdown(
        f'<div class="live"><div class="tcard-top"><div><span class="tno">{esc(live.get("train_number"))}</span>'
        f'<span class="tname">{esc(live.get("train_name"))}</span></div>{delay_pill(live.get("delay_minutes"))}</div>'
        f'<div class="live-now"><div class="label">Right now</div><div class="big">{esc(now_text)}</div></div>'
        f'<div class="halts"><div><div class="h-label">Previous halt</div><div class="h-name">{esc(prev_halt)}</div></div>'
        f'<div class="arrow">→</div>'
        f'<div class="right"><div class="h-label">Next halt</div><div class="h-name">{esc(next_halt)}</div></div></div>'
        f'<div class="live-meta">{" · ".join(meta)}</div></div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Rendering: chat
# ─────────────────────────────────────────────────────────────────────────────
def split_sources(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Separate the '📚 Sources' block the RAG tool appends to answers."""
    if "📚 Sources" not in text:
        return text.strip(), []
    answer, block = text.split("📚 Sources", 1)
    sources = []
    for line in block.splitlines():
        match = re.match(r"^\s*[•\-*]?\s*(.+?)\s+—\s+Page\s+(\d+)", line)
        if match:
            sources.append((match.group(1).strip(), match.group(2)))
    return answer.strip(), sources


def sources_html(sources: list[tuple[str, str]]) -> str:
    pages: dict[str, list[str]] = {}
    for doc, page in sources:
        pages.setdefault(doc, [])
        if page not in pages[doc]:
            pages[doc].append(page)
    items = "".join(
        f'<div class="s-item">{esc(DOCUMENT_NAMES.get(doc, doc))} <span>· p. {esc(", ".join(sorted(p, key=int)))}</span></div>'
        for doc, p in pages.items()
    )
    return f'<div class="sources"><div class="s-title">Sources</div>{items}</div>'


def source_label(meta: dict) -> str:
    tools = set(meta.get("tools_used") or [])
    if "live_train_status" in tools:
        return '<span class="src-label live">📍 Live status · RailRadar</span>'
    if tools & TIMETABLE_TOOLS or meta.get("intent") in ("TRAIN_SEARCH", "TRAIN_DETAILS", "STATION_SEARCH"):
        return '<span class="src-label timetable">🗂 Timetable · 2017 schedule</span>'
    if meta.get("intent") == "RAILWAY_RULES":
        return '<span class="src-label docs">📚 Official IRCTC &amp; Indian Railways documents</span>'
    return ""


def tidy_markdown(text: str) -> str:
    """Turn inline '•' bullets (occasionally produced by the LLM) into markdown lists."""
    text = re.sub(r"\s*•\s*", "\n- ", text)
    return text.strip()


def render_assistant(message: dict, show_follow_ups: bool) -> None:
    meta = message.get("meta") or {}
    if meta.get("error"):
        st.markdown(f'<div class="notice error">{esc(message["content"])}</div>', unsafe_allow_html=True)
        return

    answer, sources = split_sources(message["content"])
    st.markdown(tidy_markdown(answer))
    if sources:
        st.markdown(sources_html(sources), unsafe_allow_html=True)
    label = source_label(meta)
    if label:
        st.markdown(label, unsafe_allow_html=True)

    if show_follow_ups and "train_search" in (meta.get("tools_used") or []):
        st.markdown('<div class="fu-label">Follow up</div>', unsafe_allow_html=True)
        cols = st.columns(len(FOLLOW_UPS))
        for i, (label_text, prompt) in enumerate(FOLLOW_UPS):
            with cols[i]:
                if st.button(label_text, key=f"fu_chat_{i}", use_container_width=True):
                    ask(prompt)
                    st.rerun()


def call_chat(prompt: str) -> dict:
    try:
        data = api_post(
            "/chat",
            {"message": prompt, "thread_id": st.session_state.thread_id},
            timeout=CHAT_TIMEOUT,
        )
        return {
            "content": data.get("response") or "No answer was returned.",
            "meta": {
                "intent": data.get("intent"),
                "tools_used": data.get("tools_used") or [],
                "error": bool(data.get("error")),
            },
        }
    except Exception as err:
        return {"content": friendly_error(err), "meta": {"error": True}}


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🚆 Rail Assistant")
    if st.button("Start a new chat", icon=":material/add_comment:", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.pending_prompt = None
        st.rerun()

    st.markdown("#### Where answers come from")
    st.markdown(
        """
| You ask about | Answered from |
|---|---|
| Trains, routes, stations | 2017 timetable |
| Rules, refunds, Tatkal, luggage | 6 official PDFs |
| Where a train is now | RailRadar (live) |
"""
    )
    st.caption(
        "Train timings come from a static 2017 schedule and may have changed. "
        "Always confirm on [IRCTC](https://www.irctc.co.in/) before you travel."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="hero"><div><div class="hero-title">🚆 Indian Rail Travel Assistant</div>'
    '<div class="hero-sub">Find trains, check railway rules and track where your train is right now.</div></div>'
    '<div class="chips"><span class="chip"><b>Timetable</b> · 2017 schedule</span>'
    '<span class="chip"><b>Rules</b> · 6 official documents</span>'
    '<span class="chip"><b>Live</b> · RailRadar</span></div></div>',
    unsafe_allow_html=True,
)

wake_backend()

left, right = st.columns([5, 6], gap="large")

# ─────────────────────────────────────────────────────────────────────────────
# Left column: search and live status
# ─────────────────────────────────────────────────────────────────────────────
with left:
    st.segmented_control(
        "View",
        options=["search", "live"],
        format_func={"search": "🔍  Find trains", "live": "📍  Live status"}.get,
        key="left_view",
        on_change=keep_view,
        label_visibility="collapsed",
    )
    view = st.session_state.left_view or "search"

    if view == "search":
        st.markdown(
            '<div class="section-sub">Direct trains from the Indian Railways timetable. '
            "Use city names or station codes.</div>",
            unsafe_allow_html=True,
        )

        def swap_stations():
            st.session_state.from_input, st.session_state.to_input = (
                st.session_state.to_input,
                st.session_state.from_input,
            )

        c1, c2, c3 = st.columns([10, 2, 10])
        with c1:
            st.text_input("From", key="from_input", placeholder="e.g. Hyderabad or HYB")
        with c2:
            st.button("⇄", key="swap", on_click=swap_stations, help="Swap stations", use_container_width=True)
        with c3:
            st.text_input("To", key="to_input", placeholder="e.g. Chennai or MAS")

        preference = st.segmented_control(
            "Sort by",
            options=list(PREFERENCES),
            format_func=PREFERENCES.get,
            key="pref",
        ) or "balanced"

        if st.button("Search trains", type="primary", use_container_width=True, icon=":material/search:"):
            origin = st.session_state.from_input.strip()
            destination = st.session_state.to_input.strip()
            if not origin or not destination:
                st.warning("Enter both a starting point and a destination.")
            else:
                with st.spinner(f"Searching trains from {origin} to {destination}…"):
                    try:
                        data = api_post(
                            "/trains/search",
                            {"origin": origin, "destination": destination, "preference": preference},
                        )
                        if data.get("error"):
                            st.error(data["error"])
                            st.session_state.search = None
                        else:
                            st.session_state.search = {
                                "origin": origin,
                                "destination": destination,
                                "preference": data.get("ranking_preference") or preference,
                                "trains": data.get("trains", []),
                                "recommended": data.get("recommended_train"),
                                "reason": data.get("reason", ""),
                            }
                    except Exception as err:
                        st.error(friendly_error(err))

        if st.session_state.search:
            render_search_results(st.session_state.search)
        else:
            st.caption(
                "The timetable is a fixed 2017 schedule with no travel dates, so results list every "
                "direct train on the route. For today's running status, use Live status."
            )

    else:
        st.markdown(
            '<div class="section-sub">Where a train is right now and how late it is, from RailRadar.</div>',
            unsafe_allow_html=True,
        )
        today = date.today()
        l1, l2 = st.columns([1, 1])
        with l1:
            train_number = st.text_input("Train number", key="live_number", placeholder="e.g. 12302", max_chars=20)
        with l2:
            journey_date = st.date_input(
                "Journey start date",
                key="live_date",
                min_value=today - timedelta(days=29),
                max_value=today + timedelta(days=1),
                format="DD/MM/YYYY",
                help="The day the train left its first station. Leave empty for the current run.",
            )

        if st.button("Check live status", type="primary", use_container_width=True, icon=":material/my_location:"):
            number = train_number.strip()
            if not number:
                st.warning("Enter a train number.")
            else:
                with st.spinner(f"Locating train {number}…"):
                    try:
                        params = {"date": journey_date.isoformat()} if journey_date else {}
                        st.session_state.live = api_get(f"/trains/{number}/live", **params)
                    except Exception as err:
                        st.session_state.live = {"error": "request_failed", "message": friendly_error(err)}

        if st.session_state.live:
            render_live(st.session_state.live)

        st.caption(
            "Live positions are crowdsourced by RailRadar and may be delayed. The free plan allows about "
            "100 lookups a month, so each result is reused for 5 minutes."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Right column: chat
# ─────────────────────────────────────────────────────────────────────────────
with right:
    st.markdown('<div class="section-title">Ask the assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Ask about trains, routes, live status, or railway rules such as Tatkal, '
        "refunds, luggage and concessions.</div>",
        unsafe_allow_html=True,
    )

    chat_box = st.container(height=590, border=True)
    with chat_box:
        if not st.session_state.messages:
            st.markdown(
                '<div class="welcome"><div class="w-title">How can I help with your journey?</div>'
                '<div class="w-sub">Try one of these, or type your own question below.</div></div>',
                unsafe_allow_html=True,
            )
            s1, s2 = st.columns(2)
            for i, (label, prompt) in enumerate(SUGGESTIONS):
                with (s1 if i % 2 == 0 else s2):
                    if st.button(label, key=f"chip_{i}", use_container_width=True):
                        ask(prompt)
                        st.rerun()

        last_assistant = max(
            (i for i, m in enumerate(st.session_state.messages) if m["role"] == "assistant"),
            default=-1,
        )
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                with st.chat_message("user", avatar=":material/person:"):
                    st.markdown(message["content"])
            else:
                with st.chat_message("assistant", avatar=":material/train:"):
                    render_assistant(message, show_follow_ups=(i == last_assistant))

    typed = st.chat_input("Ask about trains, live status or railway rules…")

    prompt = st.session_state.pending_prompt or typed
    if prompt:
        st.session_state.pending_prompt = None
        shown = prompt.split("\n\n[Context from")[0].strip()
        st.session_state.messages.append({"role": "user", "content": shown})
        with chat_box:
            with st.chat_message("user", avatar=":material/person:"):
                st.markdown(shown)
            with st.chat_message("assistant", avatar=":material/train:"):
                with st.spinner("Thinking…"):
                    reply = call_chat(prompt)
        st.session_state.messages.append({"role": "assistant", **reply})
        st.rerun()

st.markdown(
    '<div class="footer">Timetable data: Indian Railways, 2017 · Rules: official IRCTC and Indian Railways documents · '
    'Live status: <a href="https://railradar.in" target="_blank">RailRadar</a> · '
    "Always confirm on IRCTC before you travel.</div>",
    unsafe_allow_html=True,
)

# Keep the newest chat message in view inside the scrollable chat panel. The
# run counter makes the snippet differ on every rerun so Streamlit re-runs it.
if st.session_state.messages:
    components.html(
        f"<!-- run {st.session_state.run_count} -->"
        """
<script>
const doc = window.parent.document;
function scrollChat() {
  doc.querySelectorAll('[data-testid="stVerticalBlockBorderWrapper"]').forEach((el) => {
    if (el.querySelector('[data-testid="stChatMessage"]')) {
      const scroller = el.querySelector('div[style*="overflow"]') || el;
      scroller.scrollTop = scroller.scrollHeight;
      el.scrollTop = el.scrollHeight;
    }
  });
}
[50, 250, 600].forEach((t) => setTimeout(scrollChat, t));
</script>
""",
        height=0,
    )
