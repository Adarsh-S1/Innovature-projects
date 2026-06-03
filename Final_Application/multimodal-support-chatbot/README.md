# Multimodal Intelligent Customer Support Chatbot

> **Multi-Agent LangGraph + Multimodal RAG Pipeline**

An intelligent, multimodal RAG-powered support chatbot that ingests technical PDF manuals, extracts both text and images, and delivers structured answers paired with relevant diagrams through a 5-agent LangGraph pipeline.

## 🧠 Architecture Overview

The system is built on a highly modular, decoupled architecture separating the fast API/Chat layer from the heavy background document ingestion processes. 

### 1. The Retrieval-Augmented Generation (RAG) Pipeline
When a user asks a question, the request is passed through a 5-agent **LangGraph** orchestrator:

```text
User Query 
   ↓
1. Context Router (Analyzes intent, reformulates query, detects if visual context is needed)
   ↓
2. Hybrid Search (Combines BM25 keyword search + Dense Vector search using RRF fusion)
   ↓
3. Visual Specialist (Retrieves relevant images via SigLIP embeddings if required)
   ↓
4. Answer Synthesizer (Synthesizes final answer using LLM, citing sources)
   ↓
5. Quality Guard (Evaluates answer. If quality < 0.75, loops back for refinement)
   ↓
Final Multimodal Response (Text + Presigned Image URLs + Citations)
```

### 2. The Ingestion Pipeline (Data Preprocessing)
Document ingestion is handled asynchronously by **Celery** workers to ensure the main API remains highly responsive. 

1. **PDF Parsing:** Uses **Docling** for advanced table, layout, and text extraction, falling back to **RapidOCR (Torch)** for scanned elements.
2. **Semantic Chunking:** Text is split into logical, overlapping semantic chunks while preserving document hierarchy (sections, pages).
3. **Vision Processing:** Extracted images are passed to a local Vision-Language Model (VLM) via **Groq** to automatically generate highly descriptive captions and summaries.
4. **Embedding Generation:** 
   * **Text:** Embedded locally using `all-MiniLM-L6-v2` (384-dim).
   * **Images:** Embedded locally using `ViT-B-16-SigLIP-256` (768-dim) for high-accuracy cross-modal matching.
5. **Cross-Modal Linking:** Text chunks and images are linked together based on page proximity and semantic similarity.

### 3. Data Storage Strategy
Data is routed to three distinct specialized storage systems:

* **Milvus (Vector Database):** Stores all text and image embeddings in highly optimized `HNSW` indexed collections for ultra-fast semantic similarity searches.
* **RustFS / MinIO (Object Storage):** Stores the raw extracted image files and thumbnails. Generates temporary, secure presigned URLs to display images in the frontend without exposing backend storage.
* **Redis (In-Memory Datastore):** Handles session histories (with 1-hour TTLs), document metadata lookups, global caching, and sliding-window rate limiting.

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **API Server** | FastAPI, Python 3.11 |
| **Agent Orchestration** | LangGraph, LangChain |
| **LLM (Reasoning/Synthesis)** | Groq LLaMA-3.1-8b-instant |
| **Text Embeddings** | SentenceTransformers (`all-MiniLM-L6-v2`) |
| **Image Embeddings** | OpenCLIP (`ViT-B-16-SigLIP-256`) |
| **Vector Database** | Milvus 2.4+ |
| **Cache / Sessions / Queues**| Redis 7+ |
| **Object Storage** | RustFS / MinIO (S3-compatible) |
| **Task Queue / Background** | Celery |
| **Frontend** | React, Vite, TailwindCSS |

## 🚀 Quick Start

### 1. Clone and Setup Environment

```bash
conda create -n multimodal-chatbot python=3.11
conda activate multimodal-chatbot
pip install -e ".[dev]"
cp .env.example .env
# Edit .env with your API keys (Groq, etc.)
```

### 2. Start Infrastructure Services

Use Docker Compose to spin up the databases and the Celery background worker:

```bash
docker compose -f infra/docker-compose.yml up -d --build
```
*(This starts Milvus, Redis, RustFS, Etcd, and the Celery worker container).*

### 3. Run the Application

Start the FastAPI backend:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Start the React frontend (in a separate terminal):
```bash
cd frontend
npm install
npm run dev
```

### 4. Access the Interfaces

- **Frontend UI:** http://localhost:5173
- **Swagger API Docs:** http://localhost:8000/docs
- **Redis GUI (RedisInsight):** http://localhost:8002
- **Milvus GUI (Attu):** http://localhost:8001

## 📂 Project Structure

```text
multimodal-support-chatbot/
├── app/
│   ├── api/v1/          # FastAPI endpoints (chat, ingest, documents, health)
│   ├── agents/          # LangGraph agents (router, search, visual, synth, guard)
│   ├── ingestion/       # PDF parsing (Docling), chunking, embedding, Celery tasks
│   ├── retrieval/       # Hybrid search, image search, reranking
│   ├── db/              # Milvus, Redis, MinIO clients
│   ├── models/          # Pydantic schemas
│   ├── core/            # Config, logging, exceptions
│   └── main.py          # FastAPI entrypoint
├── frontend/            # React + Vite application
├── scripts/             # CLI utilities (bulk ingestion)
├── infra/               # docker-compose.yml
└── Dockerfile           # Unified image for FastAPI & Celery
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|---------|-------------|
| **POST** | `/api/v1/chat` | Main chat endpoint (invokes LangGraph) |
| **POST** | `/api/v1/ingest/pdf` | Upload a PDF (returns Job ID, async processing) |
| **GET**  | `/api/v1/ingest/status/{id}`| Poll for Celery task progress/completion |
| **GET**  | `/api/v1/sessions/{id}` | Get conversation history |
| **DELETE**| `/api/v1/sessions/{id}` | Clear conversation memory |
| **GET**  | `/api/v1/documents` | List indexed documents |

## 🧪 Development

```bash
# Run linter
ruff check app/ tests/

# Run formatter
black app/ tests/

# Run tests
pytest tests/ -v
```
