"""
config.py — Centralized configuration for the Intelligent Research Assistant.

Loads environment variables from .env and provides all constants
for LLM, embeddings, vector DB, and chunking parameters.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(Path(__file__).parent / ".env")


# ─── NVIDIA NIM API ──────────────────────────────────────────────────────────
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
LLM_MODEL = "meta/llama-3.1-8b-instruct"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 1024

# ─── Embedding Model ─────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384

# ─── PostgreSQL / pgvector ────────────────────────────────────────────────────
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "research_assistant")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "research_assistant_pwd")

DB_CONNECTION_STRING = (
    f"host={POSTGRES_HOST} port={POSTGRES_PORT} "
    f"dbname={POSTGRES_DB} user={POSTGRES_USER} password={POSTGRES_PASSWORD}"
)

# ─── Document Processing ─────────────────────────────────────────────────────
DOCUMENTS_DIR = Path(__file__).parent / "documents"
CHUNK_SIZE = 500          # target characters per chunk
CHUNK_OVERLAP = 100       # overlap between consecutive chunks

# ─── RAG Pipeline ────────────────────────────────────────────────────────────
TOP_K_RESULTS = 5         # number of chunks to retrieve

# ─── Document Topics (for router context) ────────────────────────────────────
DOCUMENT_TOPICS = [
    "Transformer architecture and self-attention mechanisms",
    "BERT pre-training (Masked Language Modeling, Next Sentence Prediction)",
    "GPT-2 language model and text generation",
    "LoRA parameter-efficient fine-tuning",
    "Retrieval-Augmented Generation (RAG)",
    "LLaMA large language model family",
    "Chain-of-Thought (CoT) prompting and reasoning",
]
