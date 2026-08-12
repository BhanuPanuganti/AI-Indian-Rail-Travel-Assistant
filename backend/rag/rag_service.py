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
except ModuleNotFoundError:
    from config import settings

try:
    from backend.rag.reranker import rerank
except ModuleNotFoundError:
    from rag.reranker import rerank


_QUOTA_SIGNALS = ("429", "RESOURCE_EXHAUSTED", "quota", "rate limit")

def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(s.lower() in msg for s in _QUOTA_SIGNALS)


# Primary: Gemini
_gemini_llm = ChatGoogleGenerativeAI(
    model=settings.gemini_model,
    google_api_key=settings.gemini_api_key,
    max_output_tokens=8192,
    max_retries=2,
)

# Fallback: Groq
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
    except ImportError:
        _groq_llm = None


def _invoke_llm(messages):
    """Invoke Gemini; fall back to Groq on quota errors."""
    try:
        return _gemini_llm.invoke(messages)
    except Exception as exc:
        if _is_quota_error(exc) and _groq_llm:
            print(f"⚡ RAG: Gemini quota hit — switching to Groq. ({exc})")
            return _groq_llm.invoke(messages)
        raise


def _extract_text(content) -> str:
    """
    Normalize LLM response content to a plain string.
    Some models return a list of content blocks instead of a string;
    this handles both cases robustly.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


SYSTEM_PROMPT = """
You are an Indian Railways travel information assistant.

Answer the user's question using ONLY the provided
retrieved documents.

Rules:
1. Do not invent facts.
2. If the documents do not contain enough information,
   clearly say that the available documents do not provide
   the answer.
3. Preserve important conditions, limits, deadlines,
   exceptions, and amounts from the documents.
4. Do not treat old/historical information as current unless
   the document itself establishes that it is current.
5. Keep the answer clear and concise.
6. Do NOT use HTML tags or angle brackets (< or >) in your response. Use standard quotes or parentheses instead.
"""


def build_context(results):

    context_parts = []

    for result in results:

        metadata = result["metadata"]

        context_parts.append(
            f"""
SOURCE:
{metadata["source"]}

PAGE:
{metadata["page"]}

TOPIC:
{metadata["topic"]}

CONTENT:
{result["text"]}
"""
        )

    return "\n".join(context_parts)


def answer_question(
    question: str,
):

    results = rerank(
        query=question,
        top_k=5,
        candidate_k=15,
    )

    if not results:

        return {
            "answer": (
                "I could not find relevant information "
                "in the railway knowledge base."
            ),
            "sources": [],
        }

    context = build_context(
        results
    )

    prompt = f"""
USER QUESTION:
{question}

RETRIEVED DOCUMENTS:
{context}

Answer the question using only these documents.
"""

    response = _invoke_llm(
        [
            (
                "system",
                SYSTEM_PROMPT,
            ),
            (
                "human",
                prompt,
            ),
        ]
    )

    seen = set()
    sources = []

    for result in results:

        source = result["metadata"]["source"]
        page = result["metadata"]["page"]

        key = (
            source,
            page,
        )

        if key in seen:
            continue

        seen.add(key)

        sources.append(
            {
                "source": source,
                "page": page,
                "score": result["rerank_score"],
            }
        )

    return {
        "answer": _extract_text(response.content),
        "sources": sources,
    }