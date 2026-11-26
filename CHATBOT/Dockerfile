# ==============================================================================
# LangGraph Agentic Chatbot (RAG + HITL) — Self-Contained CHATBOT/ Dockerfile
# Build & run from INSIDE the CHATBOT/ folder:
#   cd CHATBOT
#   docker build -t langgraph-chatbot .
# ==============================================================================

FROM python:3.11-slim

# Prevent .pyc files & buffer stdout/stderr for AWS CloudWatch logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true

WORKDIR /app

# Install minimal OS dependencies (curl for HEALTHCHECK, build-essential for native wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Rust-based package installer) + dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=30 --retries 5 uv && \
    UV_HTTP_TIMEOUT=120 uv pip install --system --no-cache -r requirements.txt

# Pre-cache the local HuggingFace embedding model (~90MB) for fast AWS cold starts
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy only the CHATBOT folder contents into /app
COPY . .

# Run as non-root user for AWS ECS / App Runner security best practices
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# Container healthcheck (used by Docker Compose & AWS ECS/ALB/App Runner)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Bind to 0.0.0.0 so Streamlit is reachable outside the container
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
