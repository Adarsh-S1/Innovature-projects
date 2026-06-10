# Multimodal Intelligent Customer Support Chatbot

> **Multi-Agent LangGraph + Multimodal RAG Pipeline**

An intelligent, multimodal RAG-powered support chatbot that ingests technical PDF manuals, extracts both text and images, and delivers structured answers paired with relevant diagrams through a 5-agent LangGraph pipeline.

---

## Table of Contents

- [Architecture Overview](#-architecture-overview)
- [Tech Stack](#-tech-stack)
- [Prerequisites](#-prerequisites)
- [Installation & Setup](#-installation--setup)
  - [Step 1: Clone the Repository](#step-1-clone-the-repository)
  - [Step 2: Create Python Environment](#step-2-create-python-environment)
  - [Step 3: Install Python Dependencies](#step-3-install-python-dependencies)
  - [Step 4: Configure Environment Variables](#step-4-configure-environment-variables)
  - [Step 5: Start Infrastructure Services](#step-5-start-infrastructure-services-docker)
  - [Step 6: Ingest PDF Documents](#step-6-ingest-pdf-documents)
  - [Step 7: Start the Backend Server](#step-7-start-the-backend-server)
  - [Step 8: Start the Frontend](#step-8-start-the-frontend)
- [Accessing the Application](#-accessing-the-application)
- [Project Structure](#-project-structure)
- [API Endpoints](#-api-endpoints)
- [Development](#-development)

---

## 🧠 Architecture Overview

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
3. **Vision Processing:** Extracted images are captioned locally using **BLIP** and then formatted into structured metadata via **Groq LLM**.
4. **Embedding Generation:** 
   * **Text:** Embedded locally using `all-MiniLM-L6-v2` (384-dim).
   * **Images:** Embedded locally using `ViT-B-16-SigLIP-256` (768-dim) for high-accuracy cross-modal matching.
5. **Cross-Modal Linking:** Text chunks and images are linked together based on page proximity, explicit figure references, and semantic similarity.

### 3. Data Storage Strategy
Data is routed to three distinct specialized storage systems:

* **Milvus (Vector Database):** Stores all text and image embeddings in highly optimized `HNSW` indexed collections for ultra-fast semantic similarity searches.
* **RustFS (S3-compatible Object Storage):** Stores the raw extracted image files and thumbnails. Generates temporary, secure presigned URLs to display images in the frontend without exposing backend storage.
* **Redis (In-Memory Datastore):** Handles session histories (with 1-hour TTLs), document metadata lookups, global caching, and sliding-window rate limiting.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **API Server** | FastAPI, Python 3.11+ |
| **Agent Orchestration** | LangGraph, LangChain |
| **LLM (Reasoning/Synthesis)** | Groq LLaMA-3.3-70b-versatile |
| **Text Embeddings** | SentenceTransformers (`all-MiniLM-L6-v2`, 384-dim) |
| **Image Embeddings** | OpenCLIP SigLIP (`ViT-B-16-SigLIP-256`, 768-dim) |
| **Image Captioning** | BLIP (local, `Salesforce/blip-image-captioning-base`) |
| **Vector Database** | Milvus 2.4+ |
| **Cache / Sessions / Queues** | Redis 7+ |
| **Object Storage** | RustFS (S3-compatible) |
| **Task Queue / Background** | Celery |
| **PDF Processing** | Docling 2.0+ |
| **Frontend** | React 19, Vite, TailwindCSS 4 |

---

## 📋 Prerequisites

Before you begin, make sure the following tools are installed on your system:

| Prerequisite | Minimum Version | How to Check | Installation Guide |
|--------------|----------------|--------------|-------------------|
| **Python** | 3.11+ | `python --version` | [python.org](https://www.python.org/downloads/) |
| **Conda** (recommended) | Any | `conda --version` | [Miniconda](https://docs.conda.io/en/latest/miniconda.html) |
| **Docker** | 20.10+ | `docker --version` | [docs.docker.com](https://docs.docker.com/get-docker/) |
| **Docker Compose** | 2.0+ | `docker compose version` | Included with Docker Desktop |
| **Node.js** | 18+ | `node --version` | [nodejs.org](https://nodejs.org/) |
| **npm** | 9+ | `npm --version` | Included with Node.js |
| **Git** | Any | `git --version` | [git-scm.com](https://git-scm.com/) |
| **NVIDIA GPU** (optional) | CUDA 11.8+ | `nvidia-smi` | Recommended for faster SigLIP/BLIP inference |

### External API Keys Required

| Service | Purpose | How to Get |
|---------|---------|-----------|
| **Groq** | Primary LLM for reasoning, synthesis, and caption formatting | [console.groq.com/keys](https://console.groq.com/keys) (Free tier available) |

> **Note:** All embedding models (SigLIP, BLIP, all-MiniLM-L6-v2) run **locally** — no OpenAI or other paid API keys are required for embeddings or vision processing.

---

## 🚀 Installation & Setup

Follow these steps **in order** to set up the complete application from scratch.

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd multimodal-support-chatbot
```

### Step 2: Create Python Environment

We recommend using **Conda** for environment management:

```bash
conda create -n multimodal-chatbot python=3.11 -y
conda activate multimodal-chatbot
```

> **Alternative (venv):** If you don't use Conda:
> ```bash
> python3.11 -m venv venv
> source venv/bin/activate  # Linux/macOS
> ```

### Step 3: Install Python Dependencies

You can install using **either** method:

**Option A: Using requirements.txt**
```bash
pip install -r requirements.txt
```

**Option B: Using pyproject.toml (editable mode — recommended for development)**
```bash
pip install -e ".[dev]"
```

> **Note:** The first run will automatically download the local AI models (~2-3 GB total):
> - `all-MiniLM-L6-v2` (~80 MB) — text embeddings
> - `ViT-B-16-SigLIP-256` (~350 MB) — image embeddings
> - `Salesforce/blip-image-captioning-base` (~950 MB) — image captioning
> - Docling OCR models (~40 MB) — PDF text extraction
>
> These are cached in `~/.cache/huggingface` and only downloaded once.

### Step 4: Configure Environment Variables

```bash
cp .env.example .env
```

Open `.env` in your editor and fill in the **required** values:

```dotenv
# REQUIRED — Get your free key at https://console.groq.com/keys
GROQ_API_KEY=gsk_your_actual_groq_api_key_here

# Set to "cpu" if you don't have an NVIDIA GPU
CLIP_DEVICE=cuda   # or "cpu"
```

All other values have sensible defaults and should work out of the box for local development.

### Step 5: Start Infrastructure Services (Docker)

The application depends on four infrastructure services that are managed by Docker Compose:

| Service | Port | Purpose |
|---------|------|---------|
| **Milvus** | `19530` | Vector database for text & image embeddings |
| **Redis** | `6379` | Session cache, metadata store, Celery broker |
| **RustFS** | `9000` | S3-compatible object storage for images |
| **Etcd** | `2379` | Required by Milvus (internal coordination) |
| **Attu** | `8001` | Milvus web GUI (optional, for debugging) |
| **RedisInsight** | `8002` | Redis web GUI (optional, for debugging) |
| **Celery Worker** | — | Background PDF processing worker |

Start all services:

```bash
sudo docker compose -f infra/docker-compose.yml up -d
```

Verify all containers are healthy:

```bash
docker compose -f infra/docker-compose.yml ps
```

Wait until all services show `healthy` status (may take 30-60 seconds for Milvus).

> **GPU Note:** The Docker Compose file includes an NVIDIA GPU reservation for the Celery worker container. If you don't have NVIDIA Docker configured, comment out the `deploy:` block in `infra/docker-compose.yml` (lines 156-163).

### Step 6: Ingest PDF Documents

Before the chatbot can answer questions, you need to ingest your PDF technical manuals into the knowledge base.

1. **Place your PDF files** in the `Raw Data/` directory (create it if it doesn't exist):

```bash
mkdir -p "Raw Data"
# Copy your PDF manuals into the "Raw Data" folder
cp /path/to/your/manuals/*.pdf "Raw Data/"
```

2. **Run the bulk ingestion script:**

```bash
conda activate multimodal-chatbot
python scripts/ingest_bulk.py --dir "Raw Data"
```

This script will process each PDF through the full pipeline:
- Parse text and extract images using Docling
- Generate image captions using local BLIP model
- Create text embeddings (`all-MiniLM-L6-v2`) and image embeddings (`SigLIP`)
- Establish cross-modal links between text chunks and images
- Store vectors in Milvus and images in RustFS

> **Expected output per file:**
> ```
> [1/3] Ingesting: manual_guide.pdf
> 🧹 Removing old data for 'manual_guide' (if any)...
> ✅ Success! Chunks: 156, Images: 23
> ```
>
> **Timing:** Ingestion speed depends on PDF size and GPU availability. A 50-page PDF typically takes 2-5 minutes on GPU, 10-15 minutes on CPU.

> **Re-ingestion:** Running the script again on the same PDFs is safe — it automatically removes old data before re-inserting, so there are no duplicates.

### Step 7: Start the Backend Server

In a new terminal:

```bash
conda activate multimodal-chatbot
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Verify the backend is running:
```bash
curl http://localhost:8000/api/v1/health
```

### Step 8: Start the Frontend

In another new terminal:

```bash
cd frontend
npm install
npm run dev
```

---

## 🌐 Accessing the Application

Once all services are running, access the application at:

| Interface | URL | Description |
|-----------|-----|-------------|
| **Frontend UI** | http://localhost:5173 | Main chatbot interface |
| **Swagger API Docs** | http://localhost:8000/docs | Interactive API documentation |
| **ReDoc API Docs** | http://localhost:8000/redoc | Alternative API docs |
| **Milvus GUI (Attu)** | http://localhost:8001 | Browse vector collections |
| **Redis GUI (RedisInsight)** | http://localhost:8002 | Inspect cached sessions/metadata |

---

## 📂 Project Structure

```text
multimodal-support-chatbot/
├── app/
│   ├── api/v1/          # FastAPI endpoints (chat, ingest, documents, health)
│   ├── agents/          # LangGraph agents (router, search, visual, synth, guard)
│   ├── ingestion/       # PDF parsing (Docling), chunking, embedding, Celery tasks
│   ├── retrieval/       # Hybrid search, image search, cross-modal linking, reranking
│   ├── db/              # Milvus, Redis, RustFS clients
│   ├── models/          # Pydantic schemas and domain models
│   ├── core/            # Config, logging, exceptions, shared utilities
│   └── main.py          # FastAPI entrypoint
├── frontend/            # React + Vite + TailwindCSS application
├── scripts/             # CLI utilities (bulk ingestion, testing, collection management)
├── infra/               # docker-compose.yml (Milvus, Redis, RustFS, Celery)
├── tests/               # Pytest test suite
├── Raw Data/            # Place PDF manuals here for ingestion
├── Dockerfile           # Multi-stage image for FastAPI & Celery worker
├── pyproject.toml       # Python project configuration & dependencies
├── requirements.txt     # Python dependencies (pip install -r)
├── .env.example         # Environment variable template
└── README.md            # This file
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|---------|-------------|
| **POST** | `/api/v1/chat` | Main chat endpoint (invokes the full LangGraph pipeline) |
| **POST** | `/api/v1/ingest/pdf` | Upload a PDF for async ingestion (returns Job ID) |
| **GET**  | `/api/v1/ingest/status/{id}` | Poll Celery task status for an ingestion job |
| **GET**  | `/api/v1/documents` | List all indexed documents with metadata |
| **GET**  | `/api/v1/sessions/{id}` | Retrieve conversation history for a session |
| **DELETE** | `/api/v1/sessions/{id}` | Clear conversation memory for a session |
| **GET**  | `/api/v1/health` | Health check (Milvus, Redis, RustFS connectivity) |

---

## 🧪 Development

```bash
# Activate the environment
conda activate multimodal-chatbot

# Run linter
ruff check app/ tests/

# Run formatter
black app/ tests/

# Run tests
pytest tests/ -v

# Drop and recreate Milvus collections (useful during development)
python scripts/drop_collections.py

# Test API key connectivity
python scripts/test_api_keys.py
```

### Useful Scripts

| Script | Purpose |
|--------|---------|
| `scripts/ingest_bulk.py --dir <path>` | Bulk ingest all PDFs in a directory |
| `scripts/drop_collections.py` | Drop all Milvus collections (reset vector DB) |
| `scripts/reset_collections.py` | Reset collections to empty state |
| `scripts/test_api_keys.py` | Verify Groq/OpenAI API key connectivity |
| `scripts/test_groq_local.py` | Test Groq LLM integration locally |

### Stopping the Application

```bash
# Stop Docker infrastructure
sudo docker compose -f infra/docker-compose.yml down

# To also remove stored data (full reset)
sudo docker compose -f infra/docker-compose.yml down -v
```
