import sys
from pathlib import Path
from typing import Annotated, TypedDict

for candidate in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
    backend_dir = candidate / "backend"
    if backend_dir.exists():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        if str(backend_dir) not in sys.path:
            sys.path.insert(0, str(backend_dir))
        break

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

try:
    from backend.llm_service import llm
    from backend.tools import station_search, train_search, train_details, railway_knowledge_search
    from backend.intent_service import extract_user_intent
except ModuleNotFoundError:
    from llm_service import llm
    from tools import station_search, train_search, train_details, railway_knowledge_search
    from intent_service import extract_user_intent

class ToolState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    intent: str


train_tools = [station_search, train_search, train_details]
rag_tools = [railway_knowledge_search]

train_llm = llm.bind_tools(train_tools)
rag_llm = llm.bind_tools(rag_tools)
chat_llm = llm

TRAIN_SYSTEM_PROMPT = """You are a Train Expert Agent for Indian Railways.
You MUST use railway tools for railway information.

CRITICAL RULES:
1. Never invent train numbers, names, or timings.
2. Only report trains that appear in the tool output.
3. If train_search returns an empty list or an "error" key, relay the error message
   from the tool clearly and helpfully. Do NOT call train_search again — just explain
   the situation to the user (e.g. station not found, no direct trains, etc.).
4. Use train_search to find trains. Use station_search if needed. Use train_details
   for specific route/stops.
5. NEVER call train_details on multiple trains at once. Only call it for the single
   recommended train, if the user specifically asks for route details.
6. If station_search returns an "ambiguous" result, list the top station options
   and ask the user to specify which one they mean.
7. If station_search returns a "no_stations_found" error, tell the user the station
   was not found and ask them to check the spelling or provide a station code.
8. If the user asks a follow-up like "which one is faster?" or "compare them" but
   there are NO previous train search results visible in the conversation history,
   politely tell them you don't have a previous search result to compare —
   ask them to first search for trains between their desired stations.

RESPONSE FORMAT RULES (MANDATORY):
When presenting train search results, ALWAYS format your output using clean Markdown list syntax (`- `) with explicit line breaks. EVERY SINGLE field MUST be on its own line starting with a dash `-`. NEVER combine multiple fields onto the same line or use bullet symbols `•` on a single line.

Use this exact structure when trains are found:

I found **X** train(s) between **[Origin]** and **[Destination]**.

⭐ **Recommended:** [TRAIN NUMBER] — [TRAIN NAME]
- **Reason:** [Why this train is recommended]
- **Departure:** [Time] from [Origin Station]
- **Arrival:** [Time] at [Destination Station]
- **Duration:** [Duration]
- **Distance:** [Distance] km

**Other available trains:**
- **[TRAIN NUMBER] — [TRAIN NAME]** (Departs: [Time], Duration: [Duration])
- **[TRAIN NUMBER] — [TRAIN NAME]** (Departs: [Time], Duration: [Duration])
"""

RAG_SYSTEM_PROMPT = """You are a Railway Policy Expert Agent.
You MUST use the railway_knowledge_search tool to answer policy questions.

CRITICAL RULES:
1. Do not invent rules.
2. Use the railway_knowledge_search tool to retrieve knowledge. The tool will return the answer and the sources.
3. Simply present the answer and sources exactly as the tool returns them.
4. If the tool returns that it could not find relevant information, say:
   "I couldn't find this information in the official railway documents I have."
   Do NOT make up an answer.
"""

CHAT_SYSTEM_PROMPT = """You are a friendly Indian Railway Travel Assistant.
Answer general greetings and casual conversation warmly.
If the user asks about trains, routes, or railway rules, gently inform them you
can help if they ask a specific question.
"""

def intent_node(state: ToolState):
    user_intent = extract_user_intent(state["messages"])
    intent_val = user_intent.intent.value

    print(f"ROUTER → {intent_val}")

    return {"intent": intent_val}

def route_intent(state: ToolState):
    intent = state.get("intent")
    if intent in ("TRAIN_SEARCH", "TRAIN_DETAILS", "STATION_SEARCH"):
        return "train_agent"
    elif intent == "RAILWAY_RULES":
        return "rag_agent"
    else:
        return "chat_agent"

def train_agent(state: ToolState):
    print("TRAIN AGENT invoked")
    messages = [SystemMessage(content=TRAIN_SYSTEM_PROMPT)] + state["messages"]
    response = train_llm.invoke(messages)
    return {"messages": [response]}

def rag_agent(state: ToolState):
    print("RAG AGENT invoked")

    # Extract the last human message
    human_messages = [m.content for m in state["messages"] if m.type == "human"]
    question = human_messages[-1] if human_messages else ""

    # Directly call the knowledge tool, skipping the LLM decision loop!
    # This prevents the infinite loop and saves exactly 1 API call per RAG query.
    answer = railway_knowledge_search.invoke({"question": question})

    if not answer or not answer.strip():
        answer = "I couldn't find this information in the official railway documents I have."

    return {"messages": [AIMessage(content=answer)]}

def chat_agent(state: ToolState):
    print("CHAT AGENT invoked")
    messages = [SystemMessage(content=CHAT_SYSTEM_PROMPT)] + state["messages"]
    response = chat_llm.invoke(messages)
    return {"messages": [response]}


builder = StateGraph(ToolState)

builder.add_node("intent_node", intent_node)
builder.add_node("train_agent", train_agent)
builder.add_node("rag_agent", rag_agent)
builder.add_node("chat_agent", chat_agent)

builder.add_node("train_tools_node", ToolNode(train_tools))

builder.add_edge(START, "intent_node")

builder.add_conditional_edges(
    "intent_node",
    route_intent,
    {
        "train_agent": "train_agent",
        "rag_agent": "rag_agent",
        "chat_agent": "chat_agent",
    }
)

builder.add_conditional_edges(
    "train_agent",
    tools_condition,
    {
        "tools": "train_tools_node",
        END: END,
    }
)
builder.add_edge("train_tools_node", "train_agent")

builder.add_edge("rag_agent", END)

builder.add_edge("chat_agent", END)

memory = MemorySaver()
tool_graph = builder.compile(checkpointer=memory)