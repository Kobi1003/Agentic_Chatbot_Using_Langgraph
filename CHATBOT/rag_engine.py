"""
RAG Engine — Full Pipeline
===========================
Handles: Document loading → Chunking → Embedding → Indexing → Hybrid Search

Supported file types: PDF, TXT, DOCX, MD

Two search modes combined via Reciprocal Rank Fusion (RRF):
  1. Semantic Search  → FAISS vector store (cosine similarity)
  2. Keyword Search   → BM25Okapi (term frequency ranking)

This module uses global state so the index persists across
Streamlit reruns (Python module singleton pattern).
"""

import os
import tempfile
import numpy as np
from pathlib import Path
from typing import List, Optional

# ── LangChain imports ─────────────────────────────────────────────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ── Keyword search ────────────────────────────────────────────────────────────
from rank_bm25 import BM25Okapi


# ═══════════════════════════════════════════════════════════════════════════════
# Global state — shared across all Streamlit reruns (module singleton)
# ═══════════════════════════════════════════════════════════════════════════════
_vector_store: Optional[FAISS]    = None   # FAISS semantic index
_bm25_index:   Optional[BM25Okapi] = None  # BM25 keyword index
_chunks:       List               = []     # raw LangChain Document chunks
_doc_name:     Optional[str]      = None   # name of the loaded document
_embeddings:   Optional[HuggingFaceEmbeddings] = None  # embedding model (lazy)

# Small but effective embedding model (~90 MB, downloaded on first use)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Lazy embedding model loader
# ═══════════════════════════════════════════════════════════════════════════════
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Load the HuggingFace embedding model once and reuse.
    First call downloads the model; subsequent calls reuse the cached instance.
    """
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},      # CPU — no GPU required
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Document loading
# ═══════════════════════════════════════════════════════════════════════════════
def load_document(file_path: str, extension: str) -> List:
    """
    Load document using the appropriate LangChain loader.

    Returns a list of LangChain Document objects (one per page for PDFs,
    one per file for TXT/DOCX).
    """
    ext = extension.lower().lstrip(".")

    if ext == "pdf":
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path)

    elif ext in ("txt", "md"):
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file_path, encoding="utf-8")

    elif ext == "docx":
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file_path)

    else:
        raise ValueError(f"Unsupported file type: .{ext}  (supported: pdf, txt, md, docx)")

    return loader.load()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Chunking
# ═══════════════════════════════════════════════════════════════════════════════
def chunk_documents(docs: List, chunk_size: int = 500, chunk_overlap: int = 50) -> List:
    """
    Split documents into overlapping chunks using RecursiveCharacterTextSplitter.

    Why RecursiveCharacterTextSplitter?
    It tries to split on paragraph breaks first, then sentences, then words —
    preserving semantic context better than fixed-size splitting.

    chunk_size=500   → roughly 1-3 sentences per chunk
    chunk_overlap=50 → consecutive chunks share 50 chars to avoid context loss
                       at boundaries
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Separators tried in order — paragraph → sentence → word → char
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
        length_function=len,
    )
    return splitter.split_documents(docs)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Full indexing pipeline
# ═══════════════════════════════════════════════════════════════════════════════
def index_document(file_path: str, filename: str) -> int:
    """
    Full pipeline: Load → Chunk → Build FAISS index → Build BM25 index

    Args:
        file_path: Absolute path to the saved temp file
        filename:  Original filename (used for display and extension detection)

    Returns:
        Number of chunks created
    """
    global _vector_store, _bm25_index, _chunks, _doc_name

    ext = Path(filename).suffix
    # 1. Load
    docs = load_document(file_path, ext)

    # 2. Chunk
    chunks = chunk_documents(docs)
    if not chunks:
        raise ValueError("Document appears to be empty or could not be parsed.")

    _chunks = chunks
    _doc_name = filename

    # 3. Semantic index — embed each chunk and store in FAISS
    #    FAISS.from_documents() calls embeddings.embed_documents() internally
    embeddings = get_embeddings()
    _vector_store = FAISS.from_documents(chunks, embeddings)

    # 4. Keyword index — BM25 over tokenized chunk text
    #    Simple whitespace tokenization (sufficient for most English text)
    tokenized_corpus = [chunk.page_content.lower().split() for chunk in chunks]
    _bm25_index = BM25Okapi(tokenized_corpus)

    return len(chunks)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Search functions
# ═══════════════════════════════════════════════════════════════════════════════
def semantic_search(query: str, top_k: int = 8) -> List:
    """
    FAISS cosine similarity search.
    Returns top_k most semantically similar chunks.
    """
    if _vector_store is None:
        return []
    # similarity_search_with_score returns (Document, score) tuples
    return _vector_store.similarity_search_with_score(query, k=top_k)


def keyword_search(query: str, top_k: int = 8) -> List:
    """
    BM25 keyword search.
    Returns top_k chunks by term frequency relevance.
    """
    if _bm25_index is None:
        return []
    tokens = query.lower().split()
    scores = _bm25_index.get_scores(tokens)
    # Sort indices by BM25 score descending
    ranked_indices = np.argsort(scores)[::-1][:top_k]
    # Return (Document, score) tuples matching FAISS format
    return [(_chunks[i], float(scores[i])) for i in ranked_indices if scores[i] > 0]


def hybrid_search(query: str, top_k: int = 5) -> str:
    """
    Hybrid search using Reciprocal Rank Fusion (RRF).

    RRF combines semantic + keyword rankings without needing to normalize scores:
        RRF_score(doc) = Σ  1 / (k + rank_i)
    where k=60 (standard constant), rank_i is 1-based rank in each result list.

    Higher RRF score = document appears at top of multiple result lists.

    Returns:
        Formatted string with the most relevant chunks, ready to inject as context.
    """
    if _vector_store is None:
        return "No document is currently indexed. Please upload a document first."

    sem_results = semantic_search(query, top_k=10)
    kw_results  = keyword_search(query, top_k=10)

    K = 60  # RRF constant (standard value)
    rrf_scores = {}  # key → {"doc": Document, "score": float}

    # Score semantic results
    for rank, (doc, _score) in enumerate(sem_results, start=1):
        key = doc.page_content[:120]   # use start of content as dedup key
        if key not in rrf_scores:
            rrf_scores[key] = {"doc": doc, "score": 0.0}
        rrf_scores[key]["score"] += 1.0 / (K + rank)

    # Score keyword results
    for rank, (doc, _score) in enumerate(kw_results, start=1):
        key = doc.page_content[:120]
        if key not in rrf_scores:
            rrf_scores[key] = {"doc": doc, "score": 0.0}
        rrf_scores[key]["score"] += 1.0 / (K + rank)

    # Sort by combined RRF score
    sorted_results = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
    top_docs = [item["doc"] for item in sorted_results[:top_k]]

    if not top_docs:
        return "No relevant content found in the document for this query."

    # Format as context blocks
    context_blocks = []
    for i, doc in enumerate(top_docs, start=1):
        # Include page metadata if available (PDFs have page numbers)
        page_info = ""
        if doc.metadata.get("page") is not None:
            page_info = f" (Page {doc.metadata['page'] + 1})"
        context_blocks.append(f"[Excerpt {i}{page_info}]\n{doc.page_content.strip()}")

    return "\n\n---\n\n".join(context_blocks)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Status helpers
# ═══════════════════════════════════════════════════════════════════════════════
def is_document_loaded() -> bool:
    """Returns True if a document has been indexed."""
    return _vector_store is not None

def get_doc_name() -> str:
    """Returns the name of the currently indexed document."""
    return _doc_name or "No document loaded"

def get_chunk_count() -> int:
    """Returns the number of chunks in the current index."""
    return len(_chunks)

def clear_index():
    """Reset the RAG index (called when user uploads a new document)."""
    global _vector_store, _bm25_index, _chunks, _doc_name
    _vector_store = None
    _bm25_index   = None
    _chunks       = []
    _doc_name     = None
