"""Shared configuration for Lab 18."""

import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# --- API Keys ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Retain the lab's existing key name for compatibility with its modules.
OPENAI_API_KEY = GEMINI_API_KEY if LLM_PROVIDER == "gemini" else os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or (
    "https://generativelanguage.googleapis.com/v1beta/openai/"
    if LLM_PROVIDER == "gemini" else "https://api.openai.com/v1"
)
LLM_MODEL = os.getenv("LLM_MODEL") or (
    "gemini-3.8-flash" if LLM_PROVIDER == "gemini" else "gpt-4o-mini"
)

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
