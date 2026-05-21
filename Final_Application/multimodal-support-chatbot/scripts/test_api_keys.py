import os
import asyncio
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

async def test_openai():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.startswith("sk-your-openai"):
        print("❌ OpenAI API Key is missing or invalid in .env")
        return False
        
    print(f"Testing OpenAI API Key (starts with {api_key[:7]}...):")
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key, max_retries=1, timeout=10)
        
        # Test 1: Simple completion
        print("  - Testing ChatCompletion (gpt-4o-mini)... ", end="")
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'hello'"}]
        )
        print("✅ Success!")
        
        # Test 2: Embedding
        print("  - Testing Embeddings (text-embedding-3-large)... ", end="")
        response = await client.embeddings.create(
            model="text-embedding-3-large",
            input="Test string",
            dimensions=3072
        )
        print(f"✅ Success! Generated vector of length {len(response.data[0].embedding)}")
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


async def test_gemini():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Google Gemini API Key is missing in .env (Skipping)")
        return False
        
    print(f"Testing Gemini API Key (starts with {api_key[:6]}...):")
    try:
        from google import genai
        
        client = genai.Client(api_key=api_key)
        
        print("  - Testing Text Generation (gemini-2.0-flash)... ", end="")
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Say 'hello'",
        )
        print("✅ Success!")
        
        print("  - Testing Embeddings (gemini-embedding-2)... ", end="")
        response = client.models.embed_content(
            model="gemini-embedding-2",
            contents="Test string",
        )
        print(f"✅ Success! Generated vector of length {len(response.embeddings[0].values)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

async def main():
    print("=" * 50)
    print("API KEY VERIFICATION")
    print("=" * 50)
    
    openai_ok = await test_openai()
    print("-" * 50)
    gemini_ok = await test_gemini()
    
    print("=" * 50)
    if gemini_ok:
        print("🎉 Gemini is ready! Let's refactor the pipeline to use Gemini.")
    elif openai_ok:
        print("🎉 OpenAI is ready! Your pipeline will use real LLM agents.")
    else:
        print("⚠️ No APIs are ready. The pipeline will fall back to rule-based execution.")

if __name__ == "__main__":
    asyncio.run(main())
