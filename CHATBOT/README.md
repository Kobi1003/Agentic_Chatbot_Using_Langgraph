<div align="center">

# 🤖 Agentic AI Chatbot with LangGraph, Hybrid RAG & HITL

**A ChatGPT-style multi-thread AI assistant powered by LangGraph, Groq, Hybrid RAG (FAISS + BM25), Tavily real-time web search, and a full Human-in-the-Loop (HITL) safety pipeline.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2.0-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Groq](https://img.shields.io/badge/Groq-openai%2Fgpt--oss--120b-F55036?style=for-the-badge)](https://groq.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

</div>

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| 🛡️ **Human-in-the-Loop (HITL)** | Uses LangGraph `interrupt_before=["human_approval"]` to pause before gated actions (`book_stock`, `web_search`, `get_ai_news`, `get_stock_info`). Inspect & **edit tool arguments**, **Approve & Execute**, or **Reject with a reason** directly in the UI, with a live **HITL Audit Log**. |
| 📄 **Hybrid RAG Pipeline** | Upload **PDF, TXT, DOCX, or MD** files. Combines **FAISS** semantic search (`all-MiniLM-L6-v2` embeddings) and **BM25** keyword search via **Reciprocal Rank Fusion (RRF)**. |
| 🧠 **Smart Intent Classifier** | Dedicated `classifier_node` routes queries across 7 intents (`document`, `book_stock`, `stocks`, `weather`, `ai_news`, `web`, `direct`) before tool calling. |
| 🌐 **Real-Time Web & Market Tools** | Live **Weather**, **Stock Prices**, **Today's AI News**, and **Web Search** powered by **Tavily**, plus simulated **Stock Order Booking (`book_stock`)**. |
| 💬 **ChatGPT-Style Multi-Thread UI** | Sidebar chat history (newest at top), isolated per-thread `MemorySaver` state, `⏸` indicator on threads awaiting HITL approval, and token-by-token streaming with live tool badges. |
| 📊 **LangSmith Tracing** | Built-in observability for every classifier call, tool execution, and synthesis step. |

---

## 🏗️ Agent Architecture

```mermaid
flowchart TD
    A([START]) --> B[classifier_node]
    B --> C[chat_node]
    C -->|No tool needed| Z([END])
    C -->|Safe tool| D[tools: ToolNode]
    C -->|Gated HITL tool| E["⏸ human_approval (Interrupt)"]
    E -->|Approve / Edit args| D
    E -->|Reject with reason| C
    D --> C
```

---

## 📁 Project Structure

```text
Agentic_Chatbot_Using_Langgraph/
├── CHATBOT/
│   ├── agentic_chatbot.py   # LangGraph agent, classifier, 6 tools, & HITL gate
│   ├── app.py               # Streamlit multi-thread UI, streaming & HITL approval panel
│   ├── rag_engine.py        # Document loader, chunking, FAISS + BM25 RRF hybrid search
│   ├── .env.example         # Template for required API keys
│   └── .gitignore           # Ignores local .env, DB, and cache files
├── requirements.txt         # Project dependencies
├── LICENSE                  # MIT License
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Kobi1003/Agentic_Chatbot_Using_Langgraph.git
cd Agentic_Chatbot_Using_Langgraph
```

### 2. Create and activate the Conda environment
```bash
conda create -n langgraph python=3.11 -y
conda activate langgraph
pip install -r requirements.txt
```

### 3. Configure environment variables
Copy the example template into `CHATBOT/.env` and add your API keys:
```bash
cp CHATBOT/.env.example CHATBOT/.env
```

```env
GROQ_API_KEY="your_groq_api_key"
TAVILY_API_KEY="your_tavily_api_key"

# Optional: LangSmith Tracing
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="your_langsmith_api_key"
LANGCHAIN_PROJECT="agentic_chatbot_project"
```

### 4. Run the Streamlit app
```bash
cd CHATBOT
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`.

---

## 💡 Example Prompts to Try

| Category | Example Prompt | Default HITL Gate |
| :--- | :--- | :--- |
| 📈🛒 **Stock Booking** | `"Buy 10 shares of AAPL at market price"` | 🔴 **Gated (High Risk)** — Pause, edit qty/ticker, Approve or Reject |
| 🤖 **AI News** | `"What are the latest AI news today?"` | 🟡 **Gated (Medium Risk)** — Confirm before live web query |
| 🌐 **Web Search** | `"Search the web for Nvidia earnings"` | 🟡 **Gated (Medium Risk)** — Confirm or edit search query |
| 🌤️ **Weather** | `"What is the weather in Mumbai today?"` | 🟢 **Auto-runs** *(toggleable in sidebar)* |
| 📄 **Document RAG** | `"Summarise the key points in the uploaded document"` | 🟢 **Auto-runs** *(after uploading a PDF/TXT/DOCX)* |

---

## 🧰 Tech Stack

- **Orchestration:** LangGraph (`StateGraph`, `MemorySaver`, `ToolNode`, `interrupt_before`)
- **LLM:** Groq (`openai/gpt-oss-120b`)
- **Embeddings & Vector Store:** `sentence-transformers/all-MiniLM-L6-v2` + `FAISS`
- **Keyword Search:** `BM25Okapi` (`rank-bm25`) + Reciprocal Rank Fusion
- **Web Search:** `langchain-tavily` / `tavily-python`
- **Frontend:** Streamlit
- **Observability:** LangSmith

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
