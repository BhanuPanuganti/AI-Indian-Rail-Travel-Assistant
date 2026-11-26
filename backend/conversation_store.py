"""
Bounded in-memory chat history.

LangGraph's MemorySaver keeps every checkpoint of every conversation forever,
so memory grows with each message until the process restarts. This store keeps
only the last `max_messages` messages of the `max_threads` most recently used
conversations, which bounds memory on small instances.

History is trimmed so it always starts at a user message: an AI tool call is
never kept without the tool result that answers it, or vice versa.
"""

import os
import threading
from collections import OrderedDict

from langchain_core.messages import BaseMessage


def trim_history(messages: list[BaseMessage], max_messages: int) -> list[BaseMessage]:
    """Keep at most `max_messages`, starting at a human message."""

    if len(messages) <= max_messages:
        return list(messages)

    start = len(messages) - max_messages
    while start < len(messages) and messages[start].type != "human":
        start += 1
    return list(messages[start:])


class ConversationStore:

    def __init__(self, max_threads: int = 200, max_messages: int = 20):
        self._max_threads = max_threads
        self._max_messages = max_messages
        self._threads: OrderedDict[str, list[BaseMessage]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, thread_id: str) -> list[BaseMessage]:
        with self._lock:
            history = self._threads.get(thread_id)
            if history is None:
                return []
            self._threads.move_to_end(thread_id)
            return list(history)

    def save(self, thread_id: str, messages: list[BaseMessage]) -> None:
        with self._lock:
            self._threads[thread_id] = trim_history(messages, self._max_messages)
            self._threads.move_to_end(thread_id)
            while len(self._threads) > self._max_threads:
                self._threads.popitem(last=False)

    def __len__(self) -> int:
        return len(self._threads)


conversation_store = ConversationStore(
    max_threads=int(os.getenv("CHAT_MAX_THREADS", "200")),
    max_messages=int(os.getenv("CHAT_MAX_MESSAGES", "20")),
)
