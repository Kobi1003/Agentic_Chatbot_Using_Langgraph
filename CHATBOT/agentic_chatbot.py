"""
Agentic Chatbot — Smart Routing + Human-in-the-Loop (HITL) Pipeline
===================================================================
Graph flow:
  START → classifier_node → chat_node ──(no tool)────────▶ END
                                      ├─(safe tool)──────▶ tools → chat_node → END
                                      └─(HITL tool)──[⏸]─▶ human_approval → tools → chat_node → END

Human-in-the-Loop (HITL):
  - Graph is compiled with `interrupt_before=["human_approval"]`.
  - When the LLM requests any tool in `state["hitl_tools"]`, execution pauses
    BEFORE `human_approval` and waits for human assurance in the Streamlit UI.
  - The human can:
      1. Approve & Execute (as-is)
      2. Edit arguments & Approve (via update_state as_node="human_approval")
      3. Reject with reason (via update_state as_node="tools")
"""

import uuid
from datetime import datetime
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# ── LLMs ──────────────────────────────────────────────────────────────────────
# Main model: tool calling + answer synthesis
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)

# Classifier model: use the verified available Groq model with temperature=0.0 for deterministic routing
llm_classifier = ChatGroq(model="openai/gpt-oss-120b", temperature=0.0)

# Keep only the last N messages in context to avoid 413 token errors
MAX_HISTORY = 8

# Default tools that require Human-in-the-Loop assurance before execution
DEFAULT_HITL_TOOLS = ["book_stock", "web_search", "get_ai_news", "get_stock_info"]

# Metadata about each tool for the HITL UI (risk level, label, description)
TOOL_METADATA = {
    "book_stock": {
        "label": "📈🛒 Book Stock Order",
        "risk": "HIGH",
        "description": "Places a simulated BUY/SELL stock order with financial impact.",
    },
    "web_search": {
        "label": "🌐 Web Search",
        "risk": "MEDIUM",
        "description": "Queries external web platforms via Tavily for live information.",
    },
    "get_ai_news": {
        "label": "🤖 Get AI News",
        "risk": "MEDIUM",
        "description": "Fetches live AI industry news from external web sources.",
    },
    "get_stock_info": {
        "label": "📈 Stock Market Lookup",
        "risk": "MEDIUM",
        "description": "Searches live stock market prices and financial data.",
    },
    "get_weather": {
        "label": "🌤️ Weather Lookup",
        "risk": "LOW",
        "description": "Fetches current weather conditions for a city.",
    },
    "search_document": {
        "label": "📄 Document Search (RAG)",
        "risk": "LOW",
        "description": "Searches your locally uploaded document (read-only).",
    },
}


# ── State ─────────────────────────────────────────────────────────────────────
class ChatState(TypedDict):
    messages:   Annotated[list[BaseMessage], add_messages]
    intent:     str        # set by classifier_node, consumed by chat_node
    hitl_tools: list[str]  # tools requiring human approval (configurable from UI)


# ═══════════════════════════════════════════════════════════════════════════════
# TOOLS — results capped to avoid bloating context
# ═══════════════════════════════════════════════════════════════════════════════

def _trim(results, max_chars: int = 600) -> str:
    """Truncate Tavily results — supports both langchain-tavily dict and legacy list."""
    items = results.get("results", []) if isinstance(results, dict) else (results or [])
    return "\n\n".join(r.get("content", "")[:max_chars] for r in items if isinstance(r, dict))


def _tavily_search(query: str, max_results: int = 3) -> str:
    """Run Tavily search using langchain_tavily (with fallback to legacy tool)."""
    try:
        from langchain_tavily import TavilySearch
        raw = TavilySearch(max_results=max_results).invoke(query)
    except ImportError:
        from langchain_community.tools.tavily_search import TavilySearchResults
        raw = TavilySearchResults(max_results=max_results).invoke(query)
    return _trim(raw, 600)


@tool
def search_document(query: str) -> str:
    """
    Search the uploaded document using hybrid semantic + keyword search (RRF).
    Use ONLY for questions about the content inside the uploaded file.
    """
    from rag_engine import hybrid_search, is_document_loaded, get_doc_name
    if not is_document_loaded():
        return "No document uploaded. Ask the user to upload a file first."
    result = hybrid_search(query)
    return f"[Source: {get_doc_name()}]\n\n{result[:1500]}"


@tool
def get_weather(location: str) -> str:
    """Get current weather conditions for a city or location."""
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        return _tavily_search(f"current weather {location} today {today} temperature", max_results=3)
    except Exception as e:
        return f"Weather error: {e}"


@tool
def get_stock_info(ticker_or_company: str) -> str:
    """Get today's stock price and market data for a company or ticker symbol."""
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        return _tavily_search(f"{ticker_or_company} stock price today {today} market", max_results=3)
    except Exception as e:
        return f"Stock error: {e}"


@tool
def book_stock(
    ticker: str,
    action: str,
    quantity: int,
    order_type: str = "MARKET",
) -> str:
    """
    Place a stock BUY or SELL order.
    Args:
        ticker: Stock symbol or company name (e.g. 'AAPL', 'TSLA', 'RELIANCE').
        action: 'BUY' or 'SELL'.
        quantity: Number of shares to trade (positive integer).
        order_type: 'MARKET' or 'LIMIT'.
    """
    action_clean = action.strip().upper()
    if action_clean not in ("BUY", "SELL"):
        return f"Invalid order action '{action}'. Must be BUY or SELL."
    if quantity <= 0:
        return f"Invalid quantity '{quantity}'. Must be a positive integer."

    order_id = f"ORD-{uuid.uuid4().hex[:8].upper()}"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"ORDER EXECUTED (Simulated)\n"
        f"- Order ID: {order_id}\n"
        f"- Action: {action_clean} {quantity} shares of {ticker.upper()}\n"
        f"- Order Type: {order_type.upper()}\n"
        f"- Timestamp: {ts}\n"
        f"- Status: CONFIRMED"
    )


@tool
def get_ai_news() -> str:
    """Fetch today's latest AI news, model releases, and research announcements."""
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        return _tavily_search(f"latest AI news today {today} new models breakthroughs", max_results=4)
    except Exception as e:
        return f"AI news error: {e}"


@tool
def web_search(query: str) -> str:
    """Real-time web search for any current information."""
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        return _tavily_search(f"{query} {today}", max_results=3)
    except Exception as e:
        return f"Web search error: {e}"


tools = [
    search_document,
    get_weather,
    get_stock_info,
    book_stock,
    get_ai_news,
    web_search,
]
llm_with_tools = llm.bind_tools(tools)


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTING INSTRUCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
INTENT_INSTRUCTIONS = {
    "document":   "MUST call `search_document` now. Never answer from memory about the file.",
    "weather":    "MUST call `get_weather` now. Extract the location from the user message.",
    "stocks":     "MUST call `get_stock_info` now. Extract the company/ticker from the message.",
    "book_stock": "MUST call `book_stock` now. Extract ticker, action (BUY/SELL), quantity, and order_type.",
    "ai_news":    "MUST call `get_ai_news` now. Do not answer AI news from memory.",
    "web":        "MUST call `web_search` now. Form a concise query from the user question.",
    "direct":     "Answer directly from your knowledge. No tools needed.",
}

SYNTHESIS_INSTRUCTION = (
    "Tool results (or human rejection notice) are shown above. "
    "Write a clear, helpful answer citing the result. Do NOT call any tool again."
)


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 1 — Classifier
# ═══════════════════════════════════════════════════════════════════════════════
def classifier_node(state: ChatState) -> dict:
    """Classifies the latest user message into one of 7 routing categories."""
    last_msg = ""
    for msg in reversed(state["messages"]):
        if msg.__class__.__name__ == "HumanMessage":
            last_msg = msg.content
            break

    from rag_engine import is_document_loaded, get_doc_name
    doc_loaded = is_document_loaded()
    doc_info = (
        f"Uploaded doc available: '{get_doc_name()}'"
        if doc_loaded else "No document uploaded."
    )

    prompt = f"""Classify the query into one word: document, weather, stocks, book_stock, ai_news, web, or direct.

{doc_info}

- document: asks about uploaded file ("document","file","pdf","report","uploaded","according to")
- weather: weather/temperature for a location
- book_stock: user wants to buy, sell, book, order, or trade shares/stocks (e.g. "buy 10 AAPL", "book 5 TSLA")
- stocks: checking stock prices, market indices, share performance (read-only)
- ai_news: latest AI news, new AI models, AI research announcements
- web: other real-time/current info (news, events, prices)
- direct: general knowledge, math, coding, definitions (no live data needed)
- If doc-related but no doc loaded → web

Query: "{last_msg}"
One word:"""

    response = llm_classifier.invoke([HumanMessage(content=prompt)])

    raw    = response.content.strip().lower().split()[0] if response.content.strip() else "direct"
    valid  = {"document", "weather", "stocks", "book_stock", "ai_news", "web", "direct"}
    intent = raw if raw in valid else "web"

    if intent == "document" and not doc_loaded:
        intent = "web"

    return {"intent": intent}


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 2 — Chat node
# ═══════════════════════════════════════════════════════════════════════════════
def chat_node(state: ChatState) -> dict:
    """Tool-calling LLM node with windowed history."""
    intent = state.get("intent", "direct")
    today  = datetime.now().strftime("%A, %d %B %Y")

    last_human_idx = 0
    for i, msg in enumerate(state["messages"]):
        if msg.__class__.__name__ == "HumanMessage":
            last_human_idx = i

    tool_already_ran = any(
        msg.__class__.__name__ == "ToolMessage"
        for msg in state["messages"][last_human_idx:]
    )

    instruction = (
        SYNTHESIS_INSTRUCTION
        if tool_already_ran
        else INTENT_INSTRUCTIONS.get(intent, "Answer directly.")
    )

    system = SystemMessage(content=(
        f"You are a helpful AI assistant. Today: {today}\n"
        f"Tools: search_document, get_weather, get_stock_info, book_stock, get_ai_news, web_search\n"
        f"INSTRUCTION: {instruction}"
    ))

    history = list(state["messages"])
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]

    response = llm_with_tools.invoke([system] + history)
    return {"messages": [response]}


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 3 — Human Approval gate (graph pauses BEFORE this node when HITL needed)
# ═══════════════════════════════════════════════════════════════════════════════
def human_approval_node(state: ChatState) -> dict:
    """
    Pass-through node used as the HITL checkpoint gate.
    Because the graph is compiled with `interrupt_before=["human_approval"]`,
    execution pauses right before entering this node so the user can
    Approve, Edit, or Reject the pending tool calls in the UI.
    """
    return {}


def route_after_chat(state: ChatState) -> str:
    """
    Decide where to go after chat_node:
      - No tool calls       → END
      - Tool in hitl_tools  → human_approval (triggers HITL interrupt)
      - Safe tool           → tools (executes immediately)
    """
    last_msg = state["messages"][-1]
    tool_calls = getattr(last_msg, "tool_calls", None)
    if not tool_calls:
        return END

    hitl_set = set(state.get("hitl_tools", DEFAULT_HITL_TOOLS))
    if any(tc.get("name") in hitl_set for tc in tool_calls):
        return "human_approval"
    return "tools"


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD GRAPH
# ═══════════════════════════════════════════════════════════════════════════════
checkpoint = MemorySaver()
graph      = StateGraph(ChatState)

graph.add_node("classifier_node", classifier_node)
graph.add_node("chat_node",       chat_node)
graph.add_node("human_approval",  human_approval_node)
graph.add_node("tools",           ToolNode(tools))

graph.add_edge(START, "classifier_node")
graph.add_edge("classifier_node", "chat_node")
graph.add_conditional_edges(
    "chat_node",
    route_after_chat,
    {"human_approval": "human_approval", "tools": "tools", END: END},
)
graph.add_edge("human_approval", "tools")
graph.add_edge("tools", "chat_node")

# Pause BEFORE human_approval whenever a gated tool is called
chatbot = graph.compile(
    checkpointer=checkpoint,
    interrupt_before=["human_approval"],
)
