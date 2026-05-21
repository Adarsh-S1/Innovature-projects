import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

async def test_groq():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ Groq API Key is missing")
        return False
        
    print(f"Testing Groq API Key (starts with {api_key[:7]}...):")
    try:
        from groq import AsyncGroq
        client = AsyncGroq(api_key=api_key, max_retries=1, timeout=10)
        
        print("  - Testing ChatCompletion (llama-3.3-70b-versatile)... ", end="")
        response = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Say 'hello'"}]
        )
        print("✅ Success!")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

async def test_local_embeddings():
    print("Testing Local Text Embeddings (all-MiniLM-L6-v2):")
    try:
        from app.ingestion.embedder import TextEmbedder
        embedder = TextEmbedder()
        print("  - Downloading/Loading model... ", end="")
        vector = embedder.embed_text("Test string")
        print(f"✅ Success! Generated vector of length {len(vector)}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

async def test_full_pipeline():
    print("Testing Full Pipeline (Groq + Local Embeddings):")
    try:
        from app.agents.graph import graph
        compiled = graph.compile()
        
        initial_state = {
            "user_query": "What are the specs?",
            "session_id": "test",
            "query_intent": "",
            "needs_image": False,
            "query_keywords": [],
            "query_concepts": [],
            "query_type": "general",
            "reformulated_query": "",
            "bm25_weight": 0.4,
            "vector_weight": 0.6,
            "retrieved_chunks": [],
            "retrieved_images": [],
            "rerank_scores": [],
            "draft_answer": "",
            "final_answer": "",
            "cited_sources": [],
            "quality_score": 0.0,
            "quality_issues": [],
            "retry_count": 0,
            "messages": [],
            "conversation_history": [],
            "response_images": [],
            "response_metadata": {},
        }
        
        print("  - Invoking pipeline... ", end="")
        final_state = compiled.invoke(initial_state)
        print("✅ Success!")
        print(f"    Intent: {final_state.get('query_intent')}")
        print(f"    Answer snippet: {final_state.get('final_answer')[:100]}...")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

async def main():
    print("=" * 50)
    print("GROQ & LOCAL EMBEDDING VERIFICATION")
    print("=" * 50)
    
    await test_groq()
    print("-" * 50)
    await test_local_embeddings()
    print("-" * 50)
    await test_full_pipeline()
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())
