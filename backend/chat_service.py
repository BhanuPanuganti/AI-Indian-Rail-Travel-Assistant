import concurrent.futures

from langchain_core.messages import HumanMessage

try:
    from backend.tool_graph import tool_graph
except ModuleNotFoundError:
    from tool_graph import tool_graph


_GRAPH_TIMEOUT_SECONDS = 60

_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)


def _invoke_graph(message: str, thread_id: str) -> str:
    result = tool_graph.invoke(
        {
            "messages": [
                HumanMessage(content=message)
            ]
        },
        config={
            "configurable": {
                "thread_id": thread_id,
            }
        }
    )

    content = result["messages"][-1].content

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

    print("GRAPH START")
    print("CALLING LANGGRAPH")

    future = _executor.submit(_invoke_graph, message, thread_id)

    try:
        response = future.result(timeout=_GRAPH_TIMEOUT_SECONDS)
        print("LANGGRAPH RETURNED")
        return response

    except concurrent.futures.TimeoutError:
        future.cancel()
        print(f"LANGGRAPH TIMEOUT after {_GRAPH_TIMEOUT_SECONDS}s")
        return (
            "⏱️ The request timed out — the AI took too long to respond. "
            "Please try again, or rephrase your question."
        )

    except Exception as e:
        error_msg = str(e)
        print(f"LANGGRAPH ERROR: {error_msg}")

        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            return (
                "🚦 Google Gemini API rate limit reached (15 requests/minute). "
                "Please wait about 60 seconds and try again."
            )

        if "DEADLINE_EXCEEDED" in error_msg or "timeout" in error_msg.lower():
            return (
                "⏱️ The request timed out while contacting the AI service. "
                "Please try again."
            )

        if "PERMISSION_DENIED" in error_msg or "API_KEY" in error_msg.upper():
            return (
                "🔑 There is an issue with the API key configuration. "
                "Please contact the administrator."
            )

        return (
            "❌ An internal error occurred while processing your request. "
            "Please try again, or rephrase your question."
        )