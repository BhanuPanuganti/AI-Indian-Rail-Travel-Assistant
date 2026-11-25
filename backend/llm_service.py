import sys
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI

for candidate in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
    backend_dir = candidate / "backend"
    if backend_dir.exists():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        if str(backend_dir) not in sys.path:
            sys.path.insert(0, str(backend_dir))
        break

try:
    from backend.config import settings
    from backend.models import UserIntent
except ModuleNotFoundError:
    from config import settings
    from models import UserIntent


# Per-request timeout (seconds) for every LLM HTTP call. This is what actually
# bounds how long a worker thread can be stuck waiting on the provider.
LLM_REQUEST_TIMEOUT_SECONDS = 30


# ---------------------------------------------------------------------------
# Primary LLM — Gemini
# ---------------------------------------------------------------------------

_gemini_llm = ChatGoogleGenerativeAI(
    model=settings.gemini_model,
    google_api_key=settings.gemini_api_key,
    max_output_tokens=8192,
    max_retries=1,          # Low retries — we want to fail fast and hit fallback
    timeout=LLM_REQUEST_TIMEOUT_SECONDS,
)


# ---------------------------------------------------------------------------
# Fallback LLM — Groq (LLaMA)
# Used automatically when Gemini hits quota (429 / RESOURCE_EXHAUSTED)
# ---------------------------------------------------------------------------

_groq_llm = None

if settings.groq_api_key:
    try:
        from langchain_groq import ChatGroq
        _groq_llm = ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            max_tokens=8192,
            max_retries=1,
            timeout=LLM_REQUEST_TIMEOUT_SECONDS,
        )
        print(f"✅ Groq fallback LLM ready ({settings.groq_model})")
    except ImportError:
        print("⚠️  langchain-groq not installed. Run: pip install langchain-groq")
        _groq_llm = None


# ---------------------------------------------------------------------------
# Quota-aware wrapper
# Tries Gemini first; on 429/RESOURCE_EXHAUSTED falls back to Groq.
# ---------------------------------------------------------------------------

_QUOTA_SIGNALS = ("429", "RESOURCE_EXHAUSTED", "quota", "rate limit")


def is_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(signal.lower() in msg for signal in _QUOTA_SIGNALS)


# Process-wide counters, read by the offline evaluation scripts.
LLM_STATS = {"calls": 0, "fallbacks": 0, "errors": 0, "input_tokens": 0, "output_tokens": 0}


def reset_llm_stats() -> None:
    for key in LLM_STATS:
        LLM_STATS[key] = 0


def _record_usage(result) -> None:
    # Plain chat responses carry usage_metadata; structured-output results
    # (pydantic objects) do not, so their tokens are not counted.
    usage = getattr(result, "usage_metadata", None) or {}
    LLM_STATS["input_tokens"] += usage.get("input_tokens", 0) or 0
    LLM_STATS["output_tokens"] += usage.get("output_tokens", 0) or 0


class FallbackLLM:
    """
    Wraps primary (Gemini) + optional fallback (Groq).
    Automatically falls back on quota errors.
    Supports .invoke(), .bind_tools(), and .with_structured_output()
    so it can be used as a drop-in replacement for a LangChain ChatModel.
    """

    def __init__(self, primary, fallback=None):
        self._primary = primary
        self._fallback = fallback

    def invoke(self, messages, **kwargs):
        LLM_STATS["calls"] += 1
        try:
            result = self._primary.invoke(messages, **kwargs)
        except Exception as exc:
            if is_quota_error(exc) and self._fallback:
                print(f"⚡ Gemini quota hit — switching to Groq fallback. ({exc})")
                LLM_STATS["fallbacks"] += 1
                result = self._fallback.invoke(messages, **kwargs)
            else:
                LLM_STATS["errors"] += 1
                raise
        _record_usage(result)
        return result

    def bind_tools(self, tools, **kwargs):
        primary_bound = self._primary.bind_tools(tools, **kwargs)
        fallback_bound = self._fallback.bind_tools(tools, **kwargs) if self._fallback else None
        return FallbackLLM(primary_bound, fallback_bound)

    def with_structured_output(self, schema, **kwargs):
        primary_struct = self._primary.with_structured_output(schema, **kwargs)
        fallback_struct = self._fallback.with_structured_output(schema, **kwargs) if self._fallback else None
        return FallbackLLM(primary_struct, fallback_struct)


llm = FallbackLLM(_gemini_llm, _groq_llm)

intent_llm = llm.with_structured_output(UserIntent)