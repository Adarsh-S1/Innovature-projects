"""
vector_db.py — PostgreSQL + pgvector operations.
"""

import psycopg2
from pgvector.psycopg2 import register_vector
import config
import embeddings as emb


def get_connection(register_vec=True):
    conn = psycopg2.connect(config.DB_CONNECTION_STRING)
    if register_vec:
        register_vector(conn)
    return conn


def setup_database():
    # First connection without registering vector (extension may not exist yet)
    conn = get_connection(register_vec=False)
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    # Now register the vector type
    register_vector(conn)
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            source_document TEXT NOT NULL,
            page_number INTEGER,
            chunk_index INTEGER,
            embedding vector({config.EMBEDDING_DIMENSIONS})
        );
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding
        ON document_chunks USING hnsw (embedding vector_cosine_ops);
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("[VectorDB] Database setup complete.")


def clear_database():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM document_chunks;")
    conn.commit()
    cur.close()
    conn.close()
    print("[VectorDB] All data cleared.")


def insert_chunks(chunks: list[dict], batch_size: int = 100):
    conn = get_connection()
    cur = conn.cursor()
    total = len(chunks)
    print(f"[VectorDB] Inserting {total} chunks...")

    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["content"] for c in batch]
        vectors = emb.embed_batch(texts, batch_size=batch_size)

        for chunk, vector in zip(batch, vectors):
            cur.execute(
                """INSERT INTO document_chunks (content, source_document, page_number, chunk_index, embedding)
                VALUES (%s, %s, %s, %s, %s)""",
                (chunk["content"], chunk["source_document"], chunk["page_number"], chunk["chunk_index"], vector),
            )
        conn.commit()
        print(f"  Inserted {min(i + batch_size, total)}/{total} chunks")

    cur.close()
    conn.close()
    print(f"[VectorDB] Successfully inserted {total} chunks.")


def search(query: str, top_k: int = config.TOP_K_RESULTS) -> list[dict]:
    query_vector = emb.embed_text(query)
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """SELECT content, source_document, page_number, chunk_index,
               1 - (embedding <=> %s::vector) AS similarity
        FROM document_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s""",
        (query_vector, query_vector, top_k),
    )
    results = []
    for row in cur.fetchall():
        results.append({
            "content": row[0], "source_document": row[1],
            "page_number": row[2], "chunk_index": row[3], "similarity": float(row[4]),
        })
    cur.close()
    conn.close()
    return results


def get_chunk_count() -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM document_chunks;")
    count = cur.fetchone()[0]
    cur.close()
    conn.close()
    return count


if __name__ == "__main__":
    setup_database()
    count = get_chunk_count()
    print(f"Current chunks in database: {count}")
    if count > 0:
        results = search("What is self-attention?", top_k=3)
        for r in results:
            print(f"  [{r['similarity']:.4f}] {r['source_document']} (p.{r['page_number']})")
