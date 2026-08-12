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
    from backend.models import TravelIntent, UserIntent
    from backend.tools import railway_tools
except ModuleNotFoundError:
    from config import settings
    from models import TravelIntent, UserIntent
    from tools import railway_tools


# ---------------------------------------------------------------------------
# Primary LLM — Gemini
# ---------------------------------------------------------------------------

_gemini_llm = ChatGoogleGenerativeAI(
    model=settings.gemini_model,
    google_api_key=settings.gemini_api_key,
    max_output_tokens=8192,
    max_retries=2,          # Low retries — we want to fail fast and hit fallback
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
            max_retries=2,
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


def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(signal.lower() in msg for signal in _QUOTA_SIGNALS)


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
        try:
            return self._primary.invoke(messages, **kwargs)
        except Exception as exc:
            if _is_quota_error(exc) and self._fallback:
                print(f"⚡ Gemini quota hit — switching to Groq fallback. ({exc})")
                return self._fallback.invoke(messages, **kwargs)
            raise

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

tool_llm = llm.bind_tools(railway_tools)