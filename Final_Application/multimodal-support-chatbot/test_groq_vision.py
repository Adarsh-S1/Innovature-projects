import asyncio
import os
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()

async def test():
    client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
    try:
        response = await client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
        )
        print("✅ Success!")
    except Exception as e:
        print(f"❌ Failed: {e}")

asyncio.run(test())
