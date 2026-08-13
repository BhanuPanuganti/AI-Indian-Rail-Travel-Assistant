import json
import re
import time
import uuid
import requests
import streamlit as st
import streamlit.components.v1 as components
import os
# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Indian Rail Assistant · AI",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
)
_CHAT_TIMEOUT = 120
_SEARCH_TIMEOUT = 30

QUICK_PROMPTS = [
    ("📋", "Tatkal rules", "What are the Tatkal booking rules?"),
    ("❌", "Cancellation & refund", "What is the cancellation and refund policy?"),
    ("🪪", "Travel documents", "What documents do I need while travelling by train?"),
]

FOLLOWUP_BUTTONS = [
    ("⚡ Fastest", "Which train is the fastest?"),
    ("🌅 Earliest", "Which train departs earliest?"),
    ("🗺️ Route", "Show me the route of the recommended train"),
    ("🔄 Compare", "Compare the top 2-3 trains for me"),
]

TRAIN_KEYWORDS = [
    "train", "trains", "fastest", "earliest", "route", "depart", "arrive",
    "duration", "schedule", "recommended", "found", "km", "express",
]

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, .stApp { font-family: 'Inter', sans-serif; }

/* Preserve Streamlit Material Icons font so icons don't turn into literal text like 'face', 'smart_toy', '_arriVaTightt' */
[data-testid="stIconMaterial"], .material-symbols-outlined, .material-icons, [class*="material-symbols"] {
    font-family: 'Material Symbols Outlined', 'Material Symbols Rounded', 'Material Icons' !important;
}

.train-card {
    border: 1px solid #2a2a3e;
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 12px;
    background: #16162a;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.train-card:hover { transform: translateY(-2px); box-shadow: 0 6px 24px rgba(0,0,0,0.3); }
.train-card-recommended {
    border: 2px solid #22c55e;
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 12px;
    background: linear-gradient(135deg, #0f2a1a 0%, #0a1f14 100%);
    box-shadow: 0 0 20px rgba(34,197,94,0.15);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.train-card-recommended:hover { transform: translateY(-2px); box-shadow: 0 6px 30px rgba(34,197,94,0.25); }
.train-badge {
    display: inline-block;
    background: linear-gradient(90deg, #22c55e, #16a34a);
    color: white; font-size: 11px; font-weight: 700;
    padding: 3px 12px; border-radius: 20px;
    margin-bottom: 10px; letter-spacing: 1px; text-transform: uppercase;
}
.train-name { font-size: 19px; font-weight: 700; color: #f1f5f9; margin: 0 0 2px 0; }
.train-number { font-size: 13px; color: #6b7280; margin-bottom: 12px; font-weight: 500; }
.train-time-row { display: flex; align-items: center; gap: 12px; margin: 10px 0; }
.train-dep, .train-arr { flex: 0 0 auto; }
.train-time { font-size: 26px; font-weight: 700; color: #f8fafc; line-height: 1; }
.train-time-label { font-size: 11px; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px; }
.train-station { font-size: 12px; color: #6b7280; margin-top: 2px; }
.train-line { flex: 1; display: flex; align-items: center; gap: 6px; }
.train-line-bar { flex: 1; height: 2px; background: linear-gradient(90deg, #22c55e, #3b82f6); border-radius: 2px; }
.train-arrow-icon { color: #3b82f6; font-size: 18px; }
.train-meta { display: flex; gap: 16px; margin-top: 12px; padding-top: 12px; border-top: 1px solid #1e293b; flex-wrap: wrap; }
.train-meta-item { display: flex; align-items: center; gap: 5px; font-size: 13px; color: #94a3b8; background: #1e293b; padding: 4px 10px; border-radius: 6px; font-weight: 500; }
.reason-pill { display: inline-block; background: #1c1a07; color: #fbbf24; border: 1px solid #d97706; font-size: 12px; padding: 4px 12px; border-radius: 20px; margin-top: 10px; font-weight: 600; }

.source-box { border: 1px solid #1e3a5f; border-left: 4px solid #3b82f6; border-radius: 8px; padding: 12px 16px; background: #0f1e35; margin-top: 14px; font-size: 13px; color: #94a3b8; }
.source-title { font-weight: 700; color: #60a5fa; margin-bottom: 8px; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px; }
.source-item { padding: 3px 0; color: #7dd3fc; }

.followup-label { font-size: 12px; color: #6b7280; margin-bottom: 6px; margin-top: 12px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
div[data-testid="column"] .stButton button {
    border: 1px solid #2a2a3e !important;
    border-radius: 20px !important;
    background: #1a1a2e !important;
    color: #a5b4fc !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    padding: 6px 12px !important;
    text-align: center !important;
}
div[data-testid="column"] .stButton button:hover {
    background: #2a2a4e !important;
    border-color: #6366f1 !important;
    color: #c7d2fe !important;
}

.welcome-title { font-size: 26px; font-weight: 700; color: #f1f5f9; margin-bottom: 6px; }
.welcome-subtitle { font-size: 15px; color: #6b7280; margin-bottom: 24px; }
.result-banner { background: linear-gradient(90deg, #0f2a1a, #091a2e); border: 1px solid #22c55e; border-radius: 10px; padding: 12px 18px; margin-bottom: 14px; color: #86efac; font-weight: 600; font-size: 15px; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
if "last_manual_search_context" not in st.session_state:
    st.session_state.last_manual_search_context = None
if "manual_search_data" not in st.session_state:
    st.session_state.manual_search_data = None


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def get_spinner_text(message: str) -> str:
    """Pick a contextual loading message based on the user's query."""
    msg = message.lower()
    if any(kw in msg for kw in [
        "trains from", "trains between", "find train", "train from", "train to",
        "trains to", "search train",
    ]):
        return "🔍 Searching trains..."
    if any(kw in msg for kw in [
        "route", "stops", "fastest", "earliest", "compare", "which train",
        "which one", "slower", "faster", "earliest",
    ]):
        return "🗺️ Looking up train details..."
    if any(kw in msg for kw in [
        "rule", "policy", "tatkal", "cancel", "refund", "luggage",
        "document", "id proof", "ticket", "pnr", "booking", "quota",
        "concession", "reservation", "waitlist",
    ]):
        return "📚 Checking railway rules..."
    return "💬 Thinking..."


RAG_POLICY_KEYWORDS = [
    "tatkal", "cancellation", "cancel", "refund", "luggage", "concession",
    "identity", "document", "aadhaar", "passport", "policy", "rule",
    "booking rule", "allowance", "penalty", "tdr", "senior citizen",
    "quota", "agent", "waitlist", "rac", "ticket rules"
]

TRAIN_SEARCH_INDICATORS = [
    "recommended", "found", "departure:", "arrival:", "duration:",
    "shortest duration", "earliest departure", "fastest option",
    "other available train", "other options include"
]


def is_train_search_response(text: str) -> bool:
    """Return True ONLY for train search / schedule responses, excluding RAG, rules, or chat."""
    lower = text.lower()
    if any(rag_kw in lower for rag_kw in RAG_POLICY_KEYWORDS):
        return False
    return any(ind in lower for ind in TRAIN_SEARCH_INDICATORS)


def split_answer_sources(text: str):
    if "📚 Sources" in text:
        parts = text.split("📚 Sources", 1)
        return parts[0].strip(), parts[1].strip()
    return text.strip(), None


def render_source_block(sources_text: str):
    lines = [ln.strip() for ln in sources_text.strip().splitlines() if ln.strip()]
    items_html = "".join(
        f'<div class="source-item">▸ {ln.lstrip("•").strip()}</div>'
        for ln in lines
    )
    st.markdown(
        f'<div class="source-box"><div class="source-title">📚 Sources</div>{items_html}</div>',
        unsafe_allow_html=True,
    )


def render_followup_buttons(key_prefix: str, search_context: str = None):
    """Render follow-up suggestion buttons after a train-related response."""
    st.markdown('<div class="followup-label">💡 Ask a follow-up</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for i, (label, prompt) in enumerate(FOLLOWUP_BUTTONS):
        with cols[i]:
            if st.button(label, key=f"{key_prefix}_fu_{i}", use_container_width=True):
                if search_context:
                    st.session_state.pending_prompt = (
                        f"{prompt}\n\n"
                        f"[Context from search results]\n{search_context}"
                    )
                else:
                    st.session_state.pending_prompt = prompt
                st.rerun()


def format_train_response(text: str) -> str:
    """Ensure train search responses are formatted with clean markdown line breaks and bullet points."""
    if not text:
        return text

    # Convert inline bullet character '•' into newlines with markdown dashes '- '
    text = re.sub(r'\s*•\s*Reason:\s*', '\n- **Reason:** ', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*•\s*Departure:\s*', '\n- **Departure:** ', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*•\s*Arrival:\s*', '\n- **Arrival:** ', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*•\s*Duration:\s*', '\n- **Duration:** ', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*•\s*Distance:\s*', '\n- **Distance:** ', text, flags=re.IGNORECASE)

    # Convert remaining inline bullets in 'Other available trains' into list items
    text = re.sub(r'(Other available train\(s\):|Other available trains:)\s*•\s*', r'\1\n- ', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*•\s*', '\n- ', text)

    # Ensure headers have proper surrounding newlines
    text = re.sub(r'(\n|^)(Recommended:)\s*', r'\1⭐ **Recommended:** ', text, flags=re.IGNORECASE)
    text = re.sub(r'(\n|^)(Other available train\(s\):|Other available trains:)\s*', r'\1\n\n**Other available trains:**\n', text, flags=re.IGNORECASE)

    return text.strip()


def render_assistant_message(content: str, msg_index: int = None, show_followup: bool = False, stream: bool = False):
    answer, _ = split_answer_sources(content)
    answer = format_train_response(answer)

    if stream and answer:
        def stream_generator():
            words = answer.split(" ")
            for i, word in enumerate(words):
                yield word + (" " if i < len(words) - 1 else "")
                time.sleep(0.012)
        st.write_stream(stream_generator)
    else:
        st.markdown(answer)

    if show_followup and is_train_search_response(content):
        render_followup_buttons(key_prefix=f"chat_{msg_index}")


def build_search_context(trains: list, recommended: dict, reason: str, origin: str, destination: str) -> str:
    lines = [f"Train search results: {origin} to {destination}"]
    if recommended:
        lines.append(
            f"Recommended: #{recommended.get('train_number')} {recommended.get('train_name')} "
            f"| Departs {recommended.get('departure_time')} | Arrives {recommended.get('arrival_time')} "
            f"| Duration {recommended.get('duration', 'N/A')} | {recommended.get('distance_km', 'N/A')} km"
            f"\nReason: {reason}"
        )
    for t in trains[:5]:
        lines.append(
            f"- #{t.get('train_number')} {t.get('train_name')} "
            f"| Dep {t.get('departure_time')} | Arr {t.get('arrival_time')} "
            f"| {t.get('duration', 'N/A')} | {t.get('distance_km', 'N/A')} km"
        )
    return "\n".join(lines)


def render_train_card(train: dict, reason: str = "", is_recommended: bool = False):
    card_class = "train-card-recommended" if is_recommended else "train-card"
    badge_html = '<span class="train-badge">⭐ Recommended</span><br>' if is_recommended else ""
    reason_html = f'<div class="reason-pill">💡 {reason}</div>' if reason and is_recommended else ""
    duration = train.get("duration") or (
        f"{train['duration_minutes'] // 60}h {train['duration_minutes'] % 60}m"
        if train.get("duration_minutes") else "N/A"
    )
    html = f"""
<div class="{card_class}">
  {badge_html}
  <div class="train-name">{train.get('train_name', 'N/A')}</div>
  <div class="train-number">Train #{train.get('train_number', '')}</div>
  <div class="train-time-row">
    <div class="train-dep">
      <div class="train-time">{train.get('departure_time', '—')}</div>
      <div class="train-time-label">Departs</div>
      <div class="train-station">{train.get('origin_name', '')} ({train.get('origin_code', '')})</div>
    </div>
    <div class="train-line">
      <div class="train-line-bar"></div>
      <span class="train-arrow-icon">✈</span>
      <div class="train-line-bar"></div>
    </div>
    <div class="train-arr">
      <div class="train-time">{train.get('arrival_time', '—')}</div>
      <div class="train-time-label">Arrives</div>
      <div class="train-station">{train.get('destination_name', '')} ({train.get('destination_code', '')})</div>
    </div>
  </div>
  <div class="train-meta">
    <div class="train-meta-item">⏱️ {duration}</div>
    <div class="train-meta-item">📏 {train.get('distance_km', 'N/A')} km</div>
    <div class="train-meta-item">🚉 {train.get('source', 'Timetable')}</div>
  </div>
  {reason_html}
</div>"""
    st.markdown(html, unsafe_allow_html=True)


def call_chat_api(user_message: str) -> str:
    """POST to /chat and return the assistant response string."""
    response = requests.post(
        f"{API_URL}/chat",
        json={
            "message": user_message,
            "thread_id": st.session_state.thread_id,
        },
        timeout=_CHAT_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    reply = data["response"]
    # Guard against raw tracebacks leaking from backend
    if isinstance(reply, str) and reply.strip().startswith("Traceback"):
        return "❌ An internal error occurred. Please try again."
    return reply


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚆 Rail Assistant")
    st.divider()

    if st.button("🗑️ New Chat", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.pending_prompt = None
        st.session_state.last_manual_search_context = None
        st.session_state.manual_search_data = None
        st.rerun()

    st.divider()

    with st.expander("ℹ️ About", expanded=False):
        st.markdown(
            """
**AI Indian Rail Travel Assistant** combines:

- 🤖 Gemini AI + Groq fallback
- 📚 RAG over official IRCTC/IR documents
- 🗄️ Indian Railways schedule database

Ask me about:
- Train schedules & routes
- Tatkal & booking rules
- Cancellation & refunds
- Travel documents & IDs
"""
        )

    st.divider()
    st.caption("⚠️ **Data Disclaimer**")
    st.caption(
        "Train schedule data is from **2017**. Times and availability may "
        "differ. Always verify on "
        "[IRCTC](https://www.irctc.co.in/) before booking."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# 🚆 Indian Rail Travel Assistant")
st.caption("Search schedules · Check rules · Plan your journey")

# ─────────────────────────────────────────────────────────────────────────────
# Train search form
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("🔍 Search Trains", expanded=True):

    col1, col2 = st.columns(2)
    with col1:
        origin = st.text_input("From", placeholder="e.g. Hyderabad")
    with col2:
        destination = st.text_input("To", placeholder="e.g. Chennai")

    col3, col4 = st.columns(2)
    with col3:
        st.date_input("Travel date")
    with col4:
        preference = st.selectbox(
            "Preference",
            ["balanced", "fastest", "shortest_distance"],
            format_func=lambda v: {
                "balanced": "🏅 Best overall",
                "fastest": "⚡ Fastest",
                "shortest_distance": "📏 Shortest distance",
            }[v],
        )

    search_btn = st.button("Search Trains", type="primary", use_container_width=True)

    if search_btn:
        if not origin.strip():
            st.error("Please enter the origin.")
        elif not destination.strip():
            st.error("Please enter the destination.")
        else:
            with st.spinner(f"🔍 Searching trains from {origin} to {destination}..."):
                try:
                    resp = requests.post(
                        f"{API_URL}/trains/search",
                        json={
                            "origin": origin,
                            "destination": destination,
                            "preference": preference,
                        },
                        timeout=_SEARCH_TIMEOUT,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    if data.get("error"):
                        st.error(f"⚠️ {data['error']}")
                        st.session_state.manual_search_data = None
                    else:
                        trains = data.get("trains", [])
                        recommended = data.get("recommended_train")
                        reason = data.get("reason", "")

                        st.session_state.manual_search_data = {
                            "origin": origin,
                            "destination": destination,
                            "trains": trains,
                            "recommended": recommended,
                            "reason": reason,
                        }

                except requests.Timeout:
                    st.error("⏱️ Search timed out. Please try again.")
                except requests.ConnectionError:
                    st.error("🔌 Cannot connect to the backend server. Is it running?")
                except requests.HTTPError as err:
                    st.error(
                        f"❌ Server error (HTTP {err.response.status_code}). Please try again."
                    )
                except Exception as err:
                    st.error(f"⚠️ Something went wrong: {err}")

    # Render manual search results outside `if search_btn:` so buttons stay mounted across reruns!
    if st.session_state.manual_search_data:
        sdata = st.session_state.manual_search_data
        trains = sdata["trains"]
        recommended = sdata["recommended"]
        reason = sdata["reason"]
        s_orig = sdata["origin"]
        s_dest = sdata["destination"]

        if not trains:
            st.warning(
                "No trains found for this route. "
                "Try station codes (e.g. HYB, MAS) or a nearby major station."
            )
        else:
            ctx = build_search_context(trains, recommended, reason, s_orig, s_dest)

            st.markdown(
                f'<div class="result-banner">🚆 Found <strong>{len(trains)} train(s)</strong>'
                f' &nbsp;·&nbsp; {s_orig} → {s_dest}</div>',
                unsafe_allow_html=True,
            )

            if recommended:
                render_train_card(recommended, reason, is_recommended=True)

            others = [
                t for t in trains
                if t.get("train_number")
                != (recommended or {}).get("train_number")
            ]
            if others:
                with st.expander(
                    f"Show all {len(others)} other train(s)", expanded=False
                ):
                    for t in others:
                        render_train_card(t, is_recommended=False)

            # ── Follow-up buttons (pass full search context to AI) ──
            st.markdown("---")
            st.markdown(
                '<div class="followup-label">💡 Ask the AI about these results</div>',
                unsafe_allow_html=True,
            )
            fc1, fc2, fc3, fc4 = st.columns(4)
            for col, (label, prompt) in zip(
                [fc1, fc2, fc3, fc4], FOLLOWUP_BUTTONS
            ):
                with col:
                    if st.button(label, key=f"mfu_{label}", use_container_width=True):
                        st.session_state.pending_prompt = (
                            f"{prompt}\n\n"
                            f"[Context from my search]\n{ctx}"
                        )
                        st.rerun()

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Chat section
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 💬 AI Travel Assistant")

# ── Welcome screen (only when no messages) ──
if not st.session_state.messages:
    st.markdown(
        '<div class="welcome-title">How can I help you today?</div>'
        '<div class="welcome-subtitle">Choose a topic below or type your question.</div>',
        unsafe_allow_html=True,
    )
    qc1, qc2 = st.columns(2)
    for i, (icon, title, prompt) in enumerate(QUICK_PROMPTS):
        with (qc1 if i % 2 == 0 else qc2):
            if st.button(
                f"{icon} **{title}**",
                key=f"qp_{i}",
                use_container_width=True,
            ):
                st.session_state.pending_prompt = prompt
                st.rerun()
    st.markdown("")  # spacing

# ── Display chat history ──
messages = st.session_state.messages
for i, msg in enumerate(messages):
    avatar_icon = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar_icon):
        is_last = i == len(messages) - 1
        if msg["role"] == "assistant":
            # Show follow-up buttons only on the last assistant message
            render_assistant_message(msg["content"], msg_index=i, show_followup=is_last)
        else:
            st.write(msg["content"])

# ── Chat input ──
user_input = st.chat_input("Ask about trains, routes, railway rules...")

# Handle pending prompt injected from buttons
if st.session_state.pending_prompt:
    user_input = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

# ── Process message ──
if user_input:
    # Strip hidden context for display only
    display_input = user_input.split("\n\n[Context from")[0].strip()

    st.session_state.messages.append({"role": "user", "content": display_input})

    with st.chat_message("user", avatar="👤"):
        st.write(display_input)

    with st.chat_message("assistant", avatar="🤖"):
        spinner_text = get_spinner_text(display_input)
        with st.spinner(spinner_text):
            try:
                reply = call_chat_api(user_input)  # send with full context
            except requests.Timeout:
                reply = (
                    "⏱️ The request timed out. The AI took too long to respond — "
                    "please try again or rephrase your question."
                )
            except requests.ConnectionError:
                reply = (
                    "🔌 Cannot connect to the backend server. "
                    "Please make sure it is running and try again."
                )
            except requests.HTTPError as err:
                reply = (
                    f"❌ Server returned an error (HTTP {err.response.status_code}). "
                    "Please try again."
                )
            except Exception as err:
                reply = f"⚠️ Something went wrong: {err}"

        render_assistant_message(reply, msg_index=len(messages), show_followup=True, stream=True)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()

# ── Auto-scroll to latest prompt ──
if st.session_state.messages:
    components.html(
        """
        <script>
            function scrollToBottom() {
                const mainSec = window.parent.document.querySelector('section.main') || 
                                window.parent.document.querySelector('[data-testid="stMain"]') ||
                                window.parent.document.querySelector('.main');
                if (mainSec) {
                    mainSec.scrollTo({ top: mainSec.scrollHeight, behavior: 'smooth' });
                }
            }
            setTimeout(scrollToBottom, 100);
            setTimeout(scrollToBottom, 300);
            setTimeout(scrollToBottom, 600);
        </script>
        """,
        height=0,
        width=0,
    )