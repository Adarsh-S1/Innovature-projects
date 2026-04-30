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
