from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend.conversation_store import ConversationStore, trim_history


def turn(i):
    return [
        HumanMessage(content=f"q{i}"),
        AIMessage(content="", tool_calls=[{"name": "train_search", "args": {}, "id": f"c{i}"}]),
        ToolMessage(content="{}", tool_call_id=f"c{i}"),
        AIMessage(content=f"a{i}"),
    ]


def test_trim_keeps_short_history():
    messages = turn(1)
    assert trim_history(messages, 10) == messages


def test_trim_starts_at_a_human_message():
    messages = turn(1) + turn(2) + turn(3)          # 12 messages
    trimmed = trim_history(messages, 6)            # cut would land mid-turn
    assert trimmed[0].content == "q3"
    assert len(trimmed) == 4
    # no tool result is left without the AI message that called it
    assert trimmed[1].tool_calls[0]["id"] == trimmed[2].tool_call_id


def test_store_caps_threads_least_recently_used_first():
    store = ConversationStore(max_threads=2, max_messages=10)
    store.save("a", turn(1))
    store.save("b", turn(2))
    store.get("a")                     # touch "a" so "b" is the oldest
    store.save("c", turn(3))

    assert len(store) == 2
    assert store.get("b") == []
    assert store.get("a")[0].content == "q1"


def test_get_returns_a_copy():
    store = ConversationStore()
    store.save("t", turn(1))
    store.get("t").append(HumanMessage(content="mutated"))
    assert len(store.get("t")) == 4


def test_chat_passes_history_between_turns(monkeypatch):
    from backend import chat_service

    seen = []

    class FakeGraph:
        def invoke(self, state):
            seen.append([m.content for m in state["messages"]])
            reply = AIMessage(content=f"reply to {state['messages'][-1].content}")
            return {"messages": state["messages"] + [reply]}

    store = ConversationStore()
    monkeypatch.setattr(chat_service, "tool_graph", FakeGraph())
    monkeypatch.setattr(chat_service, "conversation_store", store)

    assert chat_service.chat("hello", "t1") == "reply to hello"
    assert chat_service.chat("again", "t1") == "reply to again"
    chat_service.chat("other", "t2")

    assert seen[1] == ["hello", "reply to hello", "again"]
    assert seen[2] == ["other"]


# --- LLM fallback ----------------------------------------------------------

class _Raises:
    def __init__(self, exc):
        self.exc = exc

    def invoke(self, messages, **kwargs):
        raise self.exc


class _Returns:
    def invoke(self, messages, **kwargs):
        return AIMessage(content="from fallback")


import pytest


@pytest.mark.parametrize(
    "error",
    [
        "429 RESOURCE_EXHAUSTED quota exceeded",
        "503 UNAVAILABLE. This model is currently experiencing high demand.",
        "504 DEADLINE_EXCEEDED. Deadline expired before operation could complete.",
        "Request timed out.",
    ],
)
def test_fallback_llm_switches_on_quota_overload_and_timeouts(error):
    from backend.llm_service import FallbackLLM

    llm = FallbackLLM(_Raises(RuntimeError(error)), _Returns())
    assert llm.invoke("hi").content == "from fallback"


def test_fallback_llm_does_not_hide_other_errors():
    from backend.llm_service import FallbackLLM

    llm = FallbackLLM(_Raises(ValueError("400 INVALID_ARGUMENT: bad request")), _Returns())
    with pytest.raises(ValueError):
        llm.invoke("hi")


def test_chat_with_details_reports_intent_and_tools(monkeypatch):
    from backend import chat_service

    class FakeGraph:
        def invoke(self, state):
            call = AIMessage(content="", tool_calls=[{"name": "train_search", "args": {}, "id": "c1"}])
            result = ToolMessage(content="{}", tool_call_id="c1", name="train_search")
            answer = AIMessage(content="2 trains found")
            return {"messages": state["messages"] + [call, result, answer], "intent": "TRAIN_SEARCH"}

    monkeypatch.setattr(chat_service, "tool_graph", FakeGraph())
    monkeypatch.setattr(chat_service, "conversation_store", ConversationStore())

    details = chat_service.chat_with_details("trains from A to B", "t")
    assert details == {
        "response": "2 trains found",
        "intent": "TRAIN_SEARCH",
        "tools_used": ["train_search"],
        "error": False,
    }


def test_chat_with_details_marks_errors(monkeypatch):
    from backend import chat_service

    class Broken:
        def invoke(self, state):
            raise RuntimeError("503 UNAVAILABLE high demand")

    monkeypatch.setattr(chat_service, "tool_graph", Broken())
    monkeypatch.setattr(chat_service, "conversation_store", ConversationStore())

    details = chat_service.chat_with_details("hi", "t")
    assert details["error"] is True
    assert "busy" in details["response"]
