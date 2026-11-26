import concurrent.futures

from langchain_core.messages import HumanMessage

try:
    from backend.tool_graph import tool_graph
    from backend.llm_service import is_quota_error
    from backend.conversation_store import conversation_store
except ModuleNotFoundError:
    from tool_graph import tool_graph
    from llm_service import is_quota_error
    from conversation_store import conversation_store


# Upper bound on how long the HTTP request waits for an answer. Python threads
# cannot be cancelled, so a graph run that exceeds this keeps running in the
# background; the per-call LLM timeouts in llm_service are what bound that.
_GRAPH_TIMEOUT_SECONDS = 60

_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)


def _invoke_graph(message: str, thread_id: str) -> dict:
    history = conversation_store.get(thread_id)

    result = tool_graph.invoke(
        {
            "messages": history + [
                HumanMessage(content=message)
            ]
        },
    )

    conversation_store.save(thread_id, result["messages"])

    # Tools called while answering this message (not earlier turns).
    new_messages = result["messages"][len(history) + 1:]
    tools_used = []
    for msg in new_messages:
        name = getattr(msg, "name", None) if msg.type == "tool" else None
        if name and name not in tools_used:
            tools_used.append(name)

    return {
        "response": _message_text(result["messages"][-1].content),
        "intent": result.get("intent"),
        "tools_used": tools_used,
        "error": False,
    }


def _message_text(content) -> str:

    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )

    if isinstance(content, str) and content.strip().startswith("[") and "type" in content:
        import json
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return "".join(
                    block.get("text", "")
                    for block in parsed
                    if isinstance(block, dict) and block.get("type") == "text"
                )
        except Exception:
            pass

    return str(content)


def chat(
    message: str,
    thread_id: str,
) -> str:
    """Answer a chat message and return only the reply text."""

    return chat_with_details(message, thread_id)["response"]


def chat_with_details(
    message: str,
    thread_id: str,
) -> dict:
    """
    Answer a chat message. Returns the reply plus how it was produced:
    the classified intent and the tools called, so the UI can show where
    an answer came from. On failure, `error` is True and `response` holds
    a user-facing message.
    """

    try:
        return _run_graph(message, thread_id)
    except _ChatFailure as failure:
        return {"response": str(failure), "intent": None, "tools_used": [], "error": True}


class _ChatFailure(Exception):
    pass


def _run_graph(message: str, thread_id: str) -> dict:

    print("GRAPH START")
    print("CALLING LANGGRAPH")

    future = _executor.submit(_invoke_graph, message, thread_id)

    try:
        response = future.result(timeout=_GRAPH_TIMEOUT_SECONDS)
        print("LANGGRAPH RETURNED")
        return response

    except concurrent.futures.TimeoutError:
        print(f"LANGGRAPH TIMEOUT after {_GRAPH_TIMEOUT_SECONDS}s")
        raise _ChatFailure(
            "⏱️ The request timed out — the AI took too long to respond. "
            "Please try again, or rephrase your question."
        )

    except Exception as e:
        error_msg = str(e)
        print(f"LANGGRAPH ERROR: {error_msg}")

        if is_quota_error(e):
            raise _ChatFailure(
                "🚦 The AI service's rate limit has been reached. "
                "Please wait about a minute and try again."
            )

        if "DEADLINE_EXCEEDED" in error_msg or "timeout" in error_msg.lower():
            raise _ChatFailure(
                "⏱️ The request timed out while contacting the AI service. "
                "Please try again."
            )

        if "PERMISSION_DENIED" in error_msg or "API_KEY" in error_msg.upper():
            raise _ChatFailure(
                "🔑 There is an issue with the API key configuration. "
                "Please contact the administrator."
            )

        if "UNAVAILABLE" in error_msg or "503" in error_msg or "high demand" in error_msg.lower():
            raise _ChatFailure(
                "🚧 The AI service is busy right now. Please try again in a moment."
            )

        raise _ChatFailure(
            "❌ An internal error occurred while processing your request. "
            "Please try again, or rephrase your question."
        )