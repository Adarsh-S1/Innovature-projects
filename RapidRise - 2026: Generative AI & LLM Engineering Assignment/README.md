# Intelligent Research Assistant

An AI-powered research assistant that answers complex queries using a **Retrieval-Augmented Generation (RAG)** pipeline backed by a PostgreSQL/pgvector vector database, with intelligent **agentic routing** to fall back on web search when queries are outside the document scope.

## Architecture

```
User Query → [Agentic Router (LLM)] → RAG Pipeline (document-related)
                                     → Web Search (general knowledge)
                                     → Structured JSON Response
```

### Components
1. **LLM Client** — NVIDIA NIM API (Llama 3.1 8B) with Few-Shot + Chain-of-Thought prompting
2. **Document Processor** — PDF text extraction with recursive character chunking
3. **Embeddings** — HuggingFace `all-MiniLM-L6-v2` (384 dimensions)
4. **Vector Database** — PostgreSQL with pgvector (Docker)
5. **RAG Pipeline** — Semantic search + LLM generation with source citations
6. **Web Search** — DuckDuckGo fallback for out-of-scope queries
7. **Evaluation** — LLM-as-a-Judge scoring on Faithfulness and Relevance

## Quick Start

### Prerequisites
- Python 3.12+
- Docker
- NVIDIA API Key (free from [build.nvidia.com](https://build.nvidia.com))

### Setup

```bash
# 1. Start the vector database
docker compose up -d

# 2. Activate the Python environment
source ~/.venvs/research_assistant/bin/activate

# 3. Set your NVIDIA API key in .env
# Edit .env and replace nvapi-YOUR_KEY_HERE with your actual key

# 4. Run setup (downloads PDFs, processes, ingests into DB)
python main.py --setup
```

### Usage

```bash
# Interactive mode
python main.py --interactive

# Single query
python main.py --query "What is the key innovation in the Transformer architecture?"

# Routing demonstration
python main.py --demo
```

### Evaluation

```bash
python evaluation/evaluate.py
```

## Documents
The system uses 7 landmark AI research papers:
- Attention Is All You Need (Transformer)
- BERT
- GPT-2
- LoRA
- RAG
- LLaMA
- Chain-of-Thought Prompting

## Output Format
All responses are structured JSON:
```json
{
  "summary": "The answer with [Source: document.pdf] citations",
  "key_entities": ["entity1", "entity2"],
  "confidence_score": 0.92,
  "sources": ["document.pdf"]
}
```

## Evaluation Metrics
The LLM-as-a-Judge scores answers on:
- **Faithfulness** (1-5): Is the answer grounded in retrieved context?
- **Answer Relevance** (1-5): Does it directly address the question?
