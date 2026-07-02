"""Central configuration for the LocalRAG pipeline."""

import os

# Paths
DB_NAME = os.environ.get("RAG_DB", "local_rag.db")
DOCS_DIR = os.environ.get("RAG_DOCS_DIR", "docs")

# Chunking
CHUNK_SIZE = 50
CHUNK_OVERLAP = 10

# Retrieval
TOP_K = 3
MIN_SIMILARITY = 0.20

# Foundry Local LLM
FOUNDRY_CHAT_MODEL = os.environ.get("RAG_FOUNDRY_MODEL", "phi-4-mini")
FOUNDRY_TEMPERATURE = float(os.environ.get("RAG_FOUNDRY_TEMPERATURE", "0.1"))
FOUNDRY_MAX_TOKENS = int(os.environ.get("RAG_FOUNDRY_MAX_TOKENS", "512"))
VERBOSE_LLM = os.environ.get("RAG_VERBOSE", "").lower() in ("1", "true", "yes")
