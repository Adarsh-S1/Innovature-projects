"""
evaluate.py — LLM-as-a-Judge evaluation script.

Grades the RAG pipeline's answers against ground truth on:
  - Faithfulness (1-5): Is the answer grounded in the retrieved context?
  - Answer Relevance (1-5): Does it directly address the user's question?
"""

import json
import sys
import os
import re
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
import rag_pipeline
import llm_client


JUDGE_PROMPT = """You are an expert evaluator judging the quality of an AI-generated answer.

## Your Task
Compare the Generated Answer against the Ground Truth Answer and the Retrieved Context. Score on two dimensions.

## Scoring Criteria

### Faithfulness (1-5)
How well is the generated answer grounded in the retrieved context?
1 = Completely fabricated, not grounded in context at all
2 = Mostly fabricated with some vague connection to context
3 = Partially grounded, mixes context information with unsupported claims
4 = Mostly grounded in context with minor unsupported additions
5 = Fully grounded, every claim can be traced to the retrieved context

### Answer Relevance (1-5)
How well does the generated answer address the user's question?
1 = Completely irrelevant, does not address the question
2 = Tangentially related but misses the core question
3 = Partially addresses the question, missing key aspects
4 = Mostly addresses the question with minor gaps
5 = Fully and directly addresses the question

## Input
Question: {question}
Ground Truth Answer: {ground_truth}
Generated Answer: {generated_answer}
Retrieved Context: {context}

## Output Format
Respond with ONLY a valid JSON object:
{{
  "faithfulness_score": <1-5>,
  "faithfulness_reasoning": "...",
  "relevance_score": <1-5>,
  "relevance_reasoning": "..."
}}
"""


def evaluate_single(qa_pair: dict) -> dict:
    """Evaluate a single QA pair through the RAG pipeline."""
    question = qa_pair["question"]
    ground_truth = qa_pair["ground_truth_answer"]

    print(f"\n{'─'*60}")
    print(f"Evaluating Q{qa_pair['id']}: {question[:80]}...")

    # Step 1: Get RAG answer
    rag_result = rag_pipeline.generate_rag_answer(question)
    generated_answer = rag_result.get("summary", "No answer generated.")

    # Step 2: Get retrieved context
    chunks = rag_pipeline.retrieve(question)
    context = rag_pipeline.build_context(chunks) if chunks else "No context retrieved."

    # Step 3: Judge with LLM
    judge_prompt = JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        generated_answer=generated_answer,
        context=context[:3000],  # Truncate to avoid token limits
    )

    try:
        client = llm_client.get_client()
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.1,
            max_tokens=512,
        )
        raw = response.choices[0].message.content.strip()
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            scores = json.loads(json_match.group())
        else:
            scores = json.loads(raw)
    except Exception as e:
        print(f"  Judge error: {e}")
        scores = {
            "faithfulness_score": 0,
            "faithfulness_reasoning": f"Error: {e}",
            "relevance_score": 0,
            "relevance_reasoning": f"Error: {e}",
        }

    return {
        "id": qa_pair["id"],
        "question": question,
        "ground_truth": ground_truth,
        "generated_answer": generated_answer,
        "sources": rag_result.get("sources", []),
        "confidence_score": rag_result.get("confidence_score", 0),
        "faithfulness_score": scores.get("faithfulness_score", 0),
        "faithfulness_reasoning": scores.get("faithfulness_reasoning", ""),
        "relevance_score": scores.get("relevance_score", 0),
        "relevance_reasoning": scores.get("relevance_reasoning", ""),
    }


def run_evaluation():
    """Run full evaluation on all ground truth QA pairs."""
    print("=" * 70)
    print("  LLM-AS-A-JUDGE EVALUATION")
    print("=" * 70)

    # Load ground truth
    gt_path = Path(__file__).parent / "ground_truth.json"
    with open(gt_path) as f:
        qa_pairs = json.load(f)

    print(f"Loaded {len(qa_pairs)} QA pairs for evaluation.\n")

    results = []
    for qa_pair in qa_pairs:
        result = evaluate_single(qa_pair)
        results.append(result)

        print(f"\n  Q{result['id']} Scores:")
        print(f"    Faithfulness: {result['faithfulness_score']}/5 — {result['faithfulness_reasoning'][:100]}")
        print(f"    Relevance:    {result['relevance_score']}/5 — {result['relevance_reasoning'][:100]}")

    # Calculate averages
    f_scores = [r["faithfulness_score"] for r in results if r["faithfulness_score"] > 0]
    r_scores = [r["relevance_score"] for r in results if r["relevance_score"] > 0]

    avg_faithfulness = sum(f_scores) / len(f_scores) if f_scores else 0
    avg_relevance = sum(r_scores) / len(r_scores) if r_scores else 0

    # Print summary
    print(f"\n{'='*70}")
    print(f"  EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Question':<60} {'Faith':>6} {'Relev':>6}")
    print(f"  {'─'*60} {'─'*6} {'─'*6}")
    for r in results:
        q_short = r['question'][:57] + "..." if len(r['question']) > 60 else r['question']
        print(f"  {q_short:<60} {r['faithfulness_score']:>5}/5 {r['relevance_score']:>5}/5")
    print(f"  {'─'*60} {'─'*6} {'─'*6}")
    print(f"  {'AVERAGE':<60} {avg_faithfulness:>5.1f}/5 {avg_relevance:>5.1f}/5")
    print(f"{'='*70}")

    # Save results
    output = {
        "results": results,
        "summary": {
            "avg_faithfulness": round(avg_faithfulness, 2),
            "avg_relevance": round(avg_relevance, 2),
            "total_questions": len(qa_pairs),
            "evaluated": len(f_scores),
        },
    }

    results_path = Path(__file__).parent / "results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    return output


if __name__ == "__main__":
    run_evaluation()
