# Layer 3: RAG Evaluation
#
# Runs the test set through the chain and scores each answer
# on three metrics: faithfulness, context precision, answer relevance
#
# Run this after any significant change to measure if things improved

import os
import json
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# RAGAS imports
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset


def run_evaluation(chain, test_questions: list) -> dict:
    """
    Run the test set through the chain and score each answer.

    Returns a dict with scores for each metric and per-question details.
    """

    print("\n" + "="*60)
    print("RUNNING RAG EVALUATION")
    print("="*60)
    print(f"Test questions: {len(test_questions)}")
    print("Metrics: faithfulness, context_precision, answer_relevancy")
    print("="*60 + "\n")

    # ── Step 1: Run each question through the chain ───────────────────
    results = []

    for i, item in enumerate(test_questions):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"[{i+1}/{len(test_questions)}] {question[:60]}...")

        # Get retrieved chunks
        docs_with_scores = chain._vectorstore.similarity_search(
            question, k=8
        )
        contexts = [doc.page_content for doc in docs_with_scores]

        # Get answer from chain
        try:
            answer = chain.invoke(question)
        except Exception as e:
            answer = f"ERROR: {str(e)}"
            print(f"  ERROR: {str(e)}")

        print(f"  Answer: {answer[:100]}...")
        print()

        results.append({
            "question":     question,
            "answer":       answer,
            "contexts":     contexts,
            "ground_truth": ground_truth,
        })

    # ── Step 2: Build RAGAS dataset ───────────────────────────────────
    dataset = Dataset.from_list(results)

    # ── Step 3: Set up RAGAS with Groq as judge ───────────────────────────
    from ragas.run_config import RunConfig

    judge_llm = LangchainLLMWrapper(ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
    ))

    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    )

    # ── Step 4: Run RAGAS evaluation ──────────────────────────────────────
    print("Scoring with RAGAS...")
    scores = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
        ],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=RunConfig(
            max_workers=1,        # no parallel calls — avoids rate limiting
            timeout=120,          # 2 min timeout per call
            max_retries=3,        # retry on failure
        )
    )

    # ── Step 5: Display results ───────────────────────────────────────────
    def safe_score(val):
        """Handle cases where score comes back as list or None."""
        if isinstance(val, list):
            valid = [v for v in val if v is not None]
            return sum(valid) / len(valid) if valid else 0.0
        return val if val is not None else 0.0

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"  Faithfulness      : {safe_score(scores['faithfulness']):.3f}  (hallucination check)")
    print(f"  Answer Relevancy  : {safe_score(scores['answer_relevancy']):.3f}  (did it answer the question?)")
    print(f"  Context Precision : {safe_score(scores['context_precision']):.3f}  (retrieval quality)")
    print("="*60)

    # ── Step 6: Save results to file ──────────────────────────────────
    # So you can compare before/after improvements
    Path("logs").mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = Path("logs") / f"eval_{timestamp}.json"

    output = {
    "timestamp": timestamp,
    "scores": {
        "faithfulness":      safe_score(scores["faithfulness"]),
        "answer_relevancy":  safe_score(scores["answer_relevancy"]),
        "context_precision": safe_score(scores["context_precision"]),
    },
    "per_question": results,
}

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {output_path}")
    return output