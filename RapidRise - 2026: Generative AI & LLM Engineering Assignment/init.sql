-- Initialize pgvector extension and create the document_chunks table
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    source_document TEXT NOT NULL,
    page_number INTEGER,
    chunk_index INTEGER,
    embedding vector(384)
);

-- HNSW index for fast cosine similarity search
CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding
    ON document_chunks USING hnsw (embedding vector_cosine_ops);

-- Auto-generated document topics for query routing
CREATE TABLE IF NOT EXISTS document_topics (
    id SERIAL PRIMARY KEY,
    source_document TEXT NOT NULL,
    topic TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
