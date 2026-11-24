"""
LangGraph Chatbot — ChatGPT-style UI with RAG, Web Tools & Full HITL Pipeline
==============================================================================
Features:
  ✅ Multi-thread chat history sidebar (newest at top, ⏸ badge on paused threads)
  ✅ File upload → RAG (PDF, TXT, DOCX, MD) with FAISS + BM25 RRF hybrid search
  ✅ Real-time tools: Stock Booking, Weather, Stock Market, AI News, Web Search
  ✅ Full Human-in-the-Loop (HITL) Pipeline:
       - Configurable per-tool HITL safety gates in the sidebar
       - Graph pauses BEFORE executing gated tools (`interrupt_before=["human_approval"]`)
       - Interactive Approval Card: Inspect, Edit arguments, Approve, or Reject with reason
       - Session HITL Audit Log tracking every decision
  ✅ Streaming responses with live tool-call status display
  ✅ Persistent chat titles (chat_meta.json)

Run from CHATBOT/ directory:
    streamlit run app.py
"""

import os
import uuid
import json
import tempfile
from pathlib import Path
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    AIMessageChunk,
    ToolMessage,
)

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
METADATA_FILE = BASE_DIR / "chat_meta.json"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LangGraph Chatbot · HITL",
    page_icon="🛡️",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
section[data-testid="stSidebar"] .stButton > button {
    text-align: left !important;
    justify-content: flex-start !important;
    font-size: 0.84rem;
    padding: 5px 10px;
    border-radius: 6px;
    margin-bottom: 2px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.hitl-card {
    border: 2px solid #f59e0b;
    border-radius: 10px;
    padding: 16px 20px;
    background-color: rgba(245, 158, 11, 0.08);
    margin: 12px 0;
}
.risk-high {
    background-color: #dc2626;
    color: white;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 700;
}
.risk-medium {
    background-color: #d97706;
    color: white;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 700;
}
.risk-low {
    background-color: #059669;
    color: white;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ── Tool display labels ───────────────────────────────────────────────────────
TOOL_LABELS = {
    "search_document": "📄 Document Search",
    "get_weather":     "🌤️ Weather",
    "get_stock_info":  "📈 Stock Market",
    "book_stock":      "📈🛒 Book Stock",
    "get_ai_news":     "🤖 AI News",
    "web_search":      "🌐 Web Search",
}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Load chatbot & metadata (cached once per process)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_chatbot_bundle():
    """
    Import the compiled LangGraph agent and HITL metadata from agentic_chatbot.py.
    """
    from agentic_chatbot import chatbot, DEFAULT_HITL_TOOLS, TOOL_METADATA
    return chatbot, DEFAULT_HITL_TOOLS, TOOL_METADATA


chatbot, DEFAULT_HITL_TOOLS, TOOL_METADATA = load_chatbot_bundle()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Chat metadata helpers
# ═══════════════════════════════════════════════════════════════════════════════
def load_metadata() -> list:
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_metadata(chats: list):
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(chats, f, indent=2, ensure_ascii=False)


def make_title(msg: str) -> str:
    text = msg.strip().replace("\n", " ")
    return text[:38] + "…" if len(text) > 38 else text


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Session state initialisation
# ═══════════════════════════════════════════════════════════════════════════════
if "chats"           not in st.session_state: st.session_state.chats           = load_metadata()
if "active_tid"      not in st.session_state: st.session_state.active_tid      = None
if "thread_messages" not in st.session_state: st.session_state.thread_messages = {}
if "delete_confirm"  not in st.session_state: st.session_state.delete_confirm  = None
if "rag_status"      not in st.session_state: st.session_state.rag_status      = None
if "hitl_tools"      not in st.session_state: st.session_state.hitl_tools      = list(DEFAULT_HITL_TOOLS)
if "hitl_audit_log"  not in st.session_state: st.session_state.hitl_audit_log  = []


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Checkpoint & message helpers
# ═══════════════════════════════════════════════════════════════════════════════
def is_thread_paused_for_hitl(thread_id: str) -> bool:
    """Return True if the LangGraph thread is currently paused at human_approval."""
    try:
        snapshot = chatbot.get_state({"configurable": {"thread_id": thread_id}})
        return bool(snapshot and snapshot.next and "human_approval" in snapshot.next)
    except Exception:
        return False


def get_pending_tool_calls(thread_id: str) -> tuple:
    """Return (snapshot, last_ai_message, tool_calls_list) for a paused thread."""
    try:
        snapshot = chatbot.get_state({"configurable": {"thread_id": thread_id}})
        if not snapshot or not snapshot.next or "human_approval" not in snapshot.next:
            return None, None, []
        last_msg = snapshot.values["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", []) or []
        return snapshot, last_msg, tool_calls
    except Exception:
        return None, None, []


def restore_messages(thread_id: str) -> list:
    """Re-build display messages from the LangGraph MemorySaver checkpoint."""
    try:
        config   = {"configurable": {"thread_id": thread_id}}
        snapshot = chatbot.get_state(config)
        if not snapshot or not snapshot.values:
            return []
        result = []
        for msg in snapshot.values.get("messages", []):
            cls = msg.__class__.__name__
            if cls == "ToolMessage":
                continue
            if cls in ("AIMessage", "AIMessageChunk"):
                content = msg.content
                if isinstance(content, list):
                    text = "".join(b.get("text", "") for b in content if isinstance(b, dict))
                else:
                    text = str(content) if content else ""
                if text.strip():
                    result.append({"role": "assistant", "content": text})
            elif cls == "HumanMessage":
                result.append({"role": "user", "content": msg.content})
        return result
    except Exception:
        return []


def get_active_messages() -> list:
    tid = st.session_state.active_tid
    if tid is None:
        return []
    if tid not in st.session_state.thread_messages:
        st.session_state.thread_messages[tid] = restore_messages(tid)
    return st.session_state.thread_messages.get(tid, [])


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Agent streaming (supports initial turn AND resuming after HITL)
# ═══════════════════════════════════════════════════════════════════════════════
def run_agent(user_message: str | None, thread_id: str, container) -> str:
    """
    Stream the LangGraph agent response with live tool-call status.

    - Pass `user_message = str` for a new user turn.
    - Pass `user_message = None` to RESUME a paused HITL thread from checkpoint.
    """
    config = {"configurable": {"thread_id": thread_id}}

    graph_input = (
        {
            "messages": [HumanMessage(content=user_message)],
            "hitl_tools": st.session_state.hitl_tools,
        }
        if user_message is not None
        else None
    )

    tool_calls_seen  = []
    full_response    = ""
    response_started = False

    status_slot   = container.empty()
    response_slot = container.empty()

    for chunk, metadata in chatbot.stream(
        graph_input,
        config=config,
        stream_mode="messages",
    ):
        node = metadata.get("langgraph_node", "")

        # ── AI initiating a tool call ─────────────────────────────────────────
        if isinstance(chunk, AIMessageChunk) and chunk.tool_call_chunks:
            for tc in chunk.tool_call_chunks:
                name = tc.get("name", "")
                if name and name not in tool_calls_seen:
                    tool_calls_seen.append(name)
            if tool_calls_seen:
                labels = " → ".join(TOOL_LABELS.get(t, t) for t in tool_calls_seen)
                status_slot.info(f"🔧 **Preparing tool call:** {labels}…")

        # ── Tool result received ──────────────────────────────────────────────
        elif isinstance(chunk, ToolMessage):
            label = TOOL_LABELS.get(chunk.name, chunk.name)
            status_slot.info(f"⏳ **Processing result from:** {label}…")

        # ── Final AI text streaming ───────────────────────────────────────────
        elif node == "chat_node" and isinstance(chunk, AIMessageChunk):
            if isinstance(chunk.content, str):
                text = chunk.content
            elif isinstance(chunk.content, list):
                text = "".join(b.get("text", "") for b in chunk.content if isinstance(b, dict))
            else:
                text = ""

            if text:
                if not response_started:
                    if tool_calls_seen:
                        labels = " + ".join(TOOL_LABELS.get(t, t) for t in tool_calls_seen)
                        status_slot.caption(f"🔧 Used: {labels}")
                    else:
                        status_slot.empty()
                    response_started = True

                full_response += text
                response_slot.markdown(full_response + "▌")

    # If paused at HITL, show a clear paused indicator instead of blank
    if is_thread_paused_for_hitl(thread_id):
        status_slot.warning("⏸ **Execution paused** — waiting for your HITL assurance below.")
        response_slot.empty()
    else:
        response_slot.markdown(full_response)

    return full_response


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Interactive HITL Approval Panel
# ═══════════════════════════════════════════════════════════════════════════════
def render_hitl_panel(thread_id: str):
    """
    Renders the Human-in-the-Loop (HITL) Assurance UI when a thread is paused
    before `human_approval`. Allows Inspecting, Editing args, Approving, or Rejecting.
    """
    snapshot, orig_msg, tool_calls = get_pending_tool_calls(thread_id)
    if not tool_calls:
        return

    st.markdown(
        """
        <div class="hitl-card">
            <h4 style="margin:0 0 4px 0;">🛡️ Human-in-the-Loop (HITL) Assurance Required</h4>
            <p style="margin:0; font-size:0.9rem; opacity:0.85;">
                The agent wants to execute the gated action(s) below.
                Review or edit the parameters, then <b>Approve</b> or <b>Reject</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    edited_calls = []

    for idx, tc in enumerate(tool_calls):
        tool_name = tc.get("name", "unknown")
        tool_args = dict(tc.get("args", {}) or {})
        meta      = TOOL_METADATA.get(tool_name, {"label": tool_name, "risk": "MEDIUM", "description": ""})
        risk      = meta["risk"]
        risk_cls  = f"risk-{risk.lower()}"

        with st.container(border=True):
            col_h1, col_h2 = st.columns([4, 1])
            with col_h1:
                st.markdown(f"**{meta['label']}** (`{tool_name}`)")
                if meta["description"]:
                    st.caption(meta["description"])
            with col_h2:
                st.markdown(
                    f'<div style="text-align:right;"><span class="{risk_cls}">{risk} RISK</span></div>',
                    unsafe_allow_html=True,
                )

            # ── Editable arguments UI ─────────────────────────────────────────
            new_args = {}
            if not tool_args:
                st.caption("No parameters required for this tool.")
            elif tool_name == "book_stock":
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    new_args["ticker"] = st.text_input(
                        "Ticker / Symbol",
                        value=str(tool_args.get("ticker", "")),
                        key=f"hitl_{thread_id}_{idx}_ticker",
                    )
                with c2:
                    cur_action = str(tool_args.get("action", "BUY")).upper()
                    actions = ["BUY", "SELL"]
                    new_args["action"] = st.selectbox(
                        "Action",
                        options=actions,
                        index=actions.index(cur_action) if cur_action in actions else 0,
                        key=f"hitl_{thread_id}_{idx}_action",
                    )
                with c3:
                    new_args["quantity"] = st.number_input(
                        "Quantity",
                        min_value=1,
                        step=1,
                        value=max(1, int(tool_args.get("quantity", 1))),
                        key=f"hitl_{thread_id}_{idx}_qty",
                    )
                with c4:
                    cur_type = str(tool_args.get("order_type", "MARKET")).upper()
                    types = ["MARKET", "LIMIT"]
                    new_args["order_type"] = st.selectbox(
                        "Order Type",
                        options=types,
                        index=types.index(cur_type) if cur_type in types else 0,
                        key=f"hitl_{thread_id}_{idx}_otype",
                    )
            else:
                for arg_k, arg_v in tool_args.items():
                    if isinstance(arg_v, int):
                        new_args[arg_k] = st.number_input(
                            arg_k,
                            value=arg_v,
                            key=f"hitl_{thread_id}_{idx}_{arg_k}",
                        )
                    else:
                        new_args[arg_k] = st.text_input(
                            arg_k,
                            value=str(arg_v),
                            key=f"hitl_{thread_id}_{idx}_{arg_k}",
                        )

            edited_calls.append({
                "id": tc["id"],
                "name": tool_name,
                "args": new_args,
                "type": tc.get("type", "tool_call"),
            })

    # Optional reviewer note / rejection reason
    reviewer_note = st.text_input(
        "Reviewer note / rejection reason (optional)",
        placeholder="e.g., Approved for 5 shares only / Do not search external web",
        key=f"hitl_note_{thread_id}",
    )

    col_approve, col_reject, _ = st.columns([2, 2, 4])

    # ── APPROVE & EXECUTE ─────────────────────────────────────────────────────
    with col_approve:
        if st.button("✅ Approve & Execute", type="primary", use_container_width=True, key=f"approve_{thread_id}"):
            config = {"configurable": {"thread_id": thread_id}}
            args_changed = any(
                edited_calls[i]["args"] != (tool_calls[i].get("args") or {})
                for i in range(len(tool_calls))
            )

            # If human edited arguments, patch the AIMessage in-place as_node="human_approval"
            # so graph resumes straight into `tools` without pausing a second time.
            if args_changed:
                updated_ai_msg = AIMessage(
                    content=orig_msg.content,
                    tool_calls=edited_calls,
                    id=orig_msg.id,
                )
                chatbot.update_state(
                    config,
                    {"messages": [updated_ai_msg]},
                    as_node="human_approval",
                )

            decision_label = "✏️ Edited & Approved" if args_changed else "✅ Approved"
            for ec in edited_calls:
                st.session_state.hitl_audit_log.insert(0, {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "decision": decision_label,
                    "tool": ec["name"],
                    "args": ec["args"],
                    "note": reviewer_note,
                })

            with st.chat_message("assistant"):
                reply = run_agent(None, thread_id, st)

            if reply.strip():
                st.session_state.thread_messages.setdefault(thread_id, []).append(
                    {"role": "assistant", "content": reply}
                )
            st.rerun()

    # ── REJECT ACTION ─────────────────────────────────────────────────────────
    with col_reject:
        if st.button("❌ Reject Action", use_container_width=True, key=f"reject_{thread_id}"):
            config = {"configurable": {"thread_id": thread_id}}
            reason_text = reviewer_note.strip() or "User declined to authorize this action."

            rejection_messages = [
                ToolMessage(
                    tool_call_id=tc["id"],
                    name=tc["name"],
                    content=(
                        f"ACTION REJECTED BY HUMAN REVIEWER. "
                        f"Tool '{tc['name']}' was NOT executed. Reason: {reason_text}"
                    ),
                )
                for tc in tool_calls
            ]

            # Update state as_node="tools" so the graph skips tool execution
            # and routes straight to chat_node for a polite acknowledgement.
            chatbot.update_state(
                config,
                {"messages": rejection_messages},
                as_node="tools",
            )

            for tc in tool_calls:
                st.session_state.hitl_audit_log.insert(0, {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "decision": "❌ Rejected",
                    "tool": tc["name"],
                    "args": tc.get("args", {}),
                    "note": reason_text,
                })

            with st.chat_message("assistant"):
                reply = run_agent(None, thread_id, st)

            if reply.strip():
                st.session_state.thread_messages.setdefault(thread_id, []).append(
                    {"role": "assistant", "content": reply}
                )
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🤖 LangGraph Chat")

    # ── New Chat ──────────────────────────────────────────────────────────────
    if st.button("➕  New Chat", use_container_width=True, type="primary"):
        st.session_state.active_tid     = str(uuid.uuid4())
        st.session_state.delete_confirm = None
        st.rerun()

    # ── HITL Safety Controls ──────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("🛡️ HITL Safety Gates", expanded=True):
        st.caption("Select tools requiring human assurance before execution:")

        p1, p2 = st.columns(2)
        with p1:
            if st.button("Default Gates", use_container_width=True, key="preset_default"):
                st.session_state.hitl_tools = list(DEFAULT_HITL_TOOLS)
                st.rerun()
        with p2:
            if st.button("Gate All", use_container_width=True, key="preset_all"):
                st.session_state.hitl_tools = list(TOOL_METADATA.keys())
                st.rerun()

        updated_hitl = []
        for t_name, meta in TOOL_METADATA.items():
            risk_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(meta["risk"], "⚪")
            checked = st.checkbox(
                f"{risk_icon} {meta['label']}",
                value=(t_name in st.session_state.hitl_tools),
                key=f"gate_{t_name}",
                help=meta["description"],
            )
            if checked:
                updated_hitl.append(t_name)
        st.session_state.hitl_tools = updated_hitl

    # ── HITL Audit Log ────────────────────────────────────────────────────────
    with st.expander(f"📜 HITL Audit Log ({len(st.session_state.hitl_audit_log)})", expanded=False):
        if not st.session_state.hitl_audit_log:
            st.caption("No HITL decisions recorded yet.")
        else:
            for entry in st.session_state.hitl_audit_log[:10]:
                st.markdown(
                    f"**{entry['decision']}** · `{entry['tool']}`  \n"
                    f"<span style='font-size:0.78rem; opacity:0.8;'>"
                    f"{entry['time']} · Args: `{entry['args']}`"
                    f"{(' · Note: ' + entry['note']) if entry['note'] else ''}"
                    f"</span>",
                    unsafe_allow_html=True,
                )
                st.markdown("---")

    # ── File Upload (RAG) ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**📎 Document (RAG)**")

    uploaded_file = st.file_uploader(
        "Upload PDF, TXT, DOCX, or MD",
        type=["pdf", "txt", "md", "docx"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        import rag_engine
        if rag_engine.get_doc_name() != uploaded_file.name:
            with st.spinner(f"🔄 Indexing **{uploaded_file.name}**…"):
                suffix = Path(uploaded_file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                try:
                    num_chunks = rag_engine.index_document(tmp_path, uploaded_file.name)
                    st.session_state.rag_status = (
                        f"✅ **{uploaded_file.name}**\n{num_chunks} chunks indexed"
                    )
                except Exception as e:
                    st.session_state.rag_status = f"❌ Error: {str(e)}"
                finally:
                    os.unlink(tmp_path)

    if st.session_state.rag_status:
        st.success(st.session_state.rag_status)
    else:
        import rag_engine as _re
        if _re.is_document_loaded():
            st.success(f"✅ **{_re.get_doc_name()}**\n{_re.get_chunk_count()} chunks")
        else:
            st.caption("No document loaded yet.")

    # ── Chat History ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**💬 Chat History**")

    if not st.session_state.chats:
        st.caption("No chats yet — click ➕ New Chat!")
    else:
        for chat in st.session_state.chats:
            tid       = chat["thread_id"]
            title     = chat["title"]
            ts        = chat.get("created_at", "")
            is_active = tid == st.session_state.active_tid
            is_paused = is_thread_paused_for_hitl(tid)

            col_chat, col_del = st.columns([5, 1])
            with col_chat:
                prefix = "⏸ " if is_paused else ("▶ " if is_active else "   ")
                label  = prefix + title
                if st.button(
                    label,
                    key=f"open_{tid}",
                    use_container_width=True,
                    help=f"Started: {ts}" + (" · Waiting for HITL approval" if is_paused else ""),
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state.active_tid     = tid
                    st.session_state.delete_confirm = None
                    st.rerun()
            with col_del:
                if st.button("🗑", key=f"del_{tid}", help="Delete"):
                    st.session_state.delete_confirm = tid
                    st.rerun()

            if st.session_state.delete_confirm == tid:
                st.warning(f"Delete **{title}**?")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Yes", key=f"confirm_{tid}", type="primary"):
                        st.session_state.chats = [
                            c for c in st.session_state.chats if c["thread_id"] != tid
                        ]
                        save_metadata(st.session_state.chats)
                        st.session_state.thread_messages.pop(tid, None)
                        if st.session_state.active_tid == tid:
                            st.session_state.active_tid = None
                        st.session_state.delete_confirm = None
                        st.rerun()
                with c2:
                    if st.button("No", key=f"cancel_{tid}"):
                        st.session_state.delete_confirm = None
                        st.rerun()

    st.markdown("---")
    st.caption("**Model:** Groq · openai/gpt-oss-120b")
    st.caption("**RAG:** FAISS + BM25 (RRF hybrid search)")
    st.caption("**HITL:** LangGraph `interrupt_before`")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — MAIN CHAT AREA
# ═══════════════════════════════════════════════════════════════════════════════
active_tid = st.session_state.active_tid

# ── Welcome screen ────────────────────────────────────────────────────────────
if active_tid is None:
    st.markdown("# 🛡️ LangGraph Agentic Chatbot with HITL")
    st.markdown(
        "Your AI assistant with **Human-in-the-Loop (HITL) Assurance**, "
        "**Document RAG**, and **Real-time Web Tools**"
    )
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.info("🛡️ **HITL Assurance**\n\nPause before high-stakes actions — edit args, approve, or reject")
    with c2: st.info("📈🛒 **Stock Booking**\n\nPlace simulated BUY/SELL orders with mandatory human sign-off")
    with c3: st.info("🌐 **Gated Web & AI News**\n\nReview external web & AI news queries before execution")
    with c4: st.info("📄 **Document RAG**\n\nUpload PDF/TXT/DOCX — hybrid FAISS + BM25 search")

    st.markdown("---")
    st.markdown("### 💡 Example prompts to try")
    ex1, ex2, ex3 = st.columns(3)
    with ex1:
        st.code("Buy 10 shares of AAPL at market price")
        st.code("Sell 5 shares of TSLA")
    with ex2:
        st.code("What are the latest AI news today?")
        st.code("Search the web for Nvidia earnings")
    with ex3:
        st.code("What is the weather in Mumbai today?")
        st.code("Summarise the uploaded document")

    st.markdown("\n\n👈 **Click ➕ New Chat to begin!**")

# ── Active chat ───────────────────────────────────────────────────────────────
else:
    active_meta = next(
        (c for c in st.session_state.chats if c["thread_id"] == active_tid), None
    )

    if active_meta:
        st.markdown(f"### 💬 {active_meta['title']}")
        st.caption(
            f"Thread: `{active_tid[:8]}…`  ·  Started: {active_meta.get('created_at', '')}"
        )
    else:
        st.markdown("### 💬 New Chat")
        st.caption("Your first message becomes the chat title automatically")
    st.markdown("---")

    # Render past messages
    for msg in get_active_messages():
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Check if this thread is currently paused awaiting HITL approval
    waiting_hitl = is_thread_paused_for_hitl(active_tid)

    if waiting_hitl:
        render_hitl_panel(active_tid)

    # Chat input (disabled while HITL approval is pending)
    user_input = st.chat_input(
        "⏸ Resolve the pending HITL approval above first…"
        if waiting_hitl else
        "Ask anything — book stocks, search web, AI news, weather, or your document…",
        disabled=waiting_hitl,
    )

    if user_input:
        if active_tid not in st.session_state.thread_messages:
            st.session_state.thread_messages[active_tid] = []

        is_new_chat = active_meta is None

        # 1. User bubble
        st.session_state.thread_messages[active_tid].append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        # 2. Register chat in sidebar on first message
        if is_new_chat:
            new_entry = {
                "thread_id" : active_tid,
                "title"     : make_title(user_input),
                "created_at": datetime.now().strftime("%d %b %Y, %H:%M"),
            }
            st.session_state.chats.insert(0, new_entry)
            save_metadata(st.session_state.chats)

        # 3. Stream agent response (may pause at HITL gate)
        with st.chat_message("assistant"):
            assistant_reply = run_agent(user_input, active_tid, st)

        # 4. Store non-empty reply
        if assistant_reply.strip():
            st.session_state.thread_messages[active_tid].append(
                {"role": "assistant", "content": assistant_reply}
            )

        # Rerun if new chat (sidebar update) OR if paused for HITL (show Approval Card)
        if is_new_chat or is_thread_paused_for_hitl(active_tid):
            st.rerun()
