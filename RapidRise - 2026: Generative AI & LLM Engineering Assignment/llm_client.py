"""
llm_client.py — NVIDIA NIM API client with Few-Shot + Chain-of-Thought prompting.

Provides a core generation function that:
  1. Connects to NVIDIA Build API (OpenAI-compatible)
  2. Uses Few-Shot examples + CoT reasoning
  3. Returns structured JSON: {summary, key_entities, confidence_score}
"""

import json
import re
from openai import OpenAI
import config


def get_client() -> OpenAI:
    """Initialize the NVIDIA NIM OpenAI-compatible client."""
    return OpenAI(
        base_url=config.NVIDIA_BASE_URL,
        api_key=config.NVIDIA_API_KEY,
    )


# ─── System Prompt with Few-Shot Examples ────────────────────────────────────
SYSTEM_PROMPT = """You are an expert AI research assistant. Your job is to answer complex questions accurately and thoroughly.

## Instructions
1. Think step by step (Chain-of-Thought reasoning) before producing your final answer.
2. If context documents are provided, base your answer strictly on them and cite the source document names using [Source: document_name] format.
3. Extract key entities (people, models, techniques, concepts) mentioned in your answer.
4. Estimate your confidence in the answer from 0.0 to 1.0.

## Output Format
You MUST respond with ONLY a valid JSON object in this exact format (no markdown, no extra text):
{
  "chain_of_thought": "Your step-by-step reasoning here...",
  "summary": "Your final concise answer here...",
  "key_entities": ["entity1", "entity2", "entity3"],
  "confidence_score": 0.85
}

## Few-Shot Examples

### Example 1
User Query: "What is the main contribution of the Transformer architecture?"
Context: "The Transformer model architecture eschews recurrence and instead relies entirely on an attention mechanism to draw global dependencies between input and output. [Source: attention_is_all_you_need.pdf]"

Response:
{
  "chain_of_thought": "The user asks about the Transformer's main contribution. The context from the Attention paper states it eschews recurrence and relies entirely on attention. This means the key innovation is replacing recurrent layers with self-attention.",
  "summary": "The main contribution of the Transformer architecture is replacing recurrent neural network layers entirely with self-attention mechanisms to capture global dependencies between input and output sequences. This allows for significantly more parallelization during training. [Source: attention_is_all_you_need.pdf]",
  "key_entities": ["Transformer", "self-attention", "recurrence", "parallelization"],
  "confidence_score": 0.95
}

### Example 2
User Query: "How does LoRA achieve parameter-efficient fine-tuning?"
Context: "LoRA freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters. [Source: lora.pdf]"

Response:
{
  "chain_of_thought": "The user asks about LoRA's method. The context explains that LoRA freezes pre-trained weights and injects low-rank decomposition matrices. This means instead of updating all parameters, only small matrices are trained.",
  "summary": "LoRA (Low-Rank Adaptation) achieves parameter-efficient fine-tuning by freezing the pre-trained model weights and injecting small, trainable low-rank decomposition matrices into each Transformer layer. This dramatically reduces the number of trainable parameters while maintaining model performance. [Source: lora.pdf]",
  "key_entities": ["LoRA", "low-rank decomposition", "parameter-efficient fine-tuning", "Transformer"],
  "confidence_score": 0.93
}

### Example 3 (No context — general knowledge)
User Query: "What is the capital of France?"

Response:
{
  "chain_of_thought": "This is a straightforward factual question. The capital of France is Paris. No context documents are needed.",
  "summary": "The capital of France is Paris.",
  "key_entities": ["France", "Paris", "capital"],
  "confidence_score": 0.99
}
"""


def generate(
    query: str,
    context: str = "",
    system_prompt: str = SYSTEM_PROMPT,
) -> dict:
    """
    Generate a structured JSON response using Few-Shot + CoT prompting.

    Args:
        query: The user's question.
        context: Optional retrieved context (from RAG or web search).
        system_prompt: System prompt with few-shot examples.

    Returns:
        dict with keys: chain_of_thought, summary, key_entities, confidence_score
    """
    client = get_client()

    # Build user message
    user_message = ""
    if context:
        user_message += f"## Retrieved Context\n{context}\n\n"
    user_message += f"## User Query\n{query}"

    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
        )

        raw_content = response.choices[0].message.content.strip()

        # Try to extract JSON from the response (handle markdown code blocks)
        json_match = re.search(r'\{[\s\S]*\}', raw_content)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = json.loads(raw_content)

        # Ensure all required keys exist
        required_keys = ["summary", "key_entities", "confidence_score"]
        for key in required_keys:
            if key not in result:
                result[key] = "" if key == "summary" else ([] if key == "key_entities" else 0.0)

        return result

    except json.JSONDecodeError:
        # If JSON parsing fails, return a structured fallback
        return {
            "chain_of_thought": "Failed to parse LLM response as JSON.",
            "summary": raw_content,
            "key_entities": [],
            "confidence_score": 0.0,
        }
    except Exception as e:
        return {
            "chain_of_thought": f"Error during generation: {str(e)}",
            "summary": f"Error: {str(e)}",
            "key_entities": [],
            "confidence_score": 0.0,
        }


def classify_intent(query: str, document_topics: list[str]) -> dict:
    """
    Use the LLM to classify a query's intent for routing.

    Args:
        query: The user's question.
        document_topics: List of topics covered by the document corpus.

    Returns:
        dict with 'route' ('rag' or 'web_search') and optionally 'search_query'.
    """
    client = get_client()

    topics_str = "\n".join(f"  - {t}" for t in document_topics)

    router_prompt = f"""You are a query routing assistant. Given a user query and a list of document topics, classify the intent.

## Available Document Topics:
{topics_str}

## Rules:
- If the query is about any of the above topics, respond: {{"route": "rag"}}
- If the query is about current events, general knowledge, or topics NOT covered by the documents, respond: {{"route": "web_search", "search_query": "optimized search query"}}

## Important:
- Respond with ONLY a valid JSON object.
- Do NOT include any other text.

## User Query: {query}"""

    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "user", "content": router_prompt},
            ],
            temperature=0.1,
            max_tokens=128,
        )

        raw = response.choices[0].message.content.strip()
        json_match = re.search(r'\{[\s\S]*?\}', raw)
        if json_match:
            return json.loads(json_match.group())
        return json.loads(raw)

    except Exception as e:
        # Default to RAG on failure
        print(f"[Router] Classification failed: {e}. Defaulting to RAG.")
        return {"route": "rag"}


if __name__ == "__main__":
    # Quick test
    print("Testing LLM client...")
    result = generate("What is the Transformer architecture?")
    print(json.dumps(result, indent=2))
