from langchain_core.messages import SystemMessage

try:
    from backend.llm_service import intent_llm
    from backend.models import UserIntent
except ModuleNotFoundError:
    from llm_service import intent_llm
    from models import UserIntent


INTENT_SYSTEM_PROMPT = """
You are an intent classifier for an Indian Railways travel assistant.
Based on the user's latest query and the conversation history, classify the user's intent into one of the following categories:

TRAIN_SEARCH: User wants to find trains between stations or cities, or asks to compare trains (e.g. "Which one is faster?").
TRAIN_DETAILS: User is asking for the route, schedule, or stops of a specific train (e.g., "Telangana Express" or "Train 12723" or "its route").
STATION_SEARCH: User is looking for station codes or city stations.
RAILWAY_RULES: User is asking about policies, Tatkal, cancellations, luggage, IDs, etc.
GENERAL_CHAT: User is saying hello, thanks, or asking general questions unrelated to the railway system.

If the user uses pronouns like "Which one is faster?" or "Show me its route", use the conversation history to resolve the context. For example, "Which one is faster?" after a list of trains means TRAIN_SEARCH. "Show me its route" means TRAIN_DETAILS.
"""


def extract_user_intent(
    messages: list,
) -> UserIntent:

    prompt = [
        SystemMessage(
            content=INTENT_SYSTEM_PROMPT
        )
    ] + messages

    return intent_llm.invoke(prompt)