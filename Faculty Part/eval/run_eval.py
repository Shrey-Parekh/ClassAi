"""
Eval harness entry point.

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --split hr
    python eval/run_eval.py --queries eval/golden/queries.jsonl --output eval/results/run.jsonl
"""

import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics import recall_at_k, mrr, check_must_include, check_must_not_include, check_grounding, aggregate

logging.basicConfig(level=logging.WARNING)


def load_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_pipeline():
    """Build the same retrieval + generation pipeline the API uses."""
    from dotenv import load_dotenv
    load_dotenv()

    from src.utils.vector_db import VectorDBClient
    from src.utils.query_embedder import QueryEmbedder
    from src.utils.llm import LLMClient
    from src.retrieval.pipeline import RetrievalPipeline
    from src.generation.answer_generator import AnswerGenerator

    vector_db = VectorDBClient()
    query_embedder = QueryEmbedder(model_name="BAAI/bge-m3")
    llm_client = LLMClient()

    retrieval_pipeline = RetrievalPipeline(
        vector_db_client=vector_db,
        embedding_model=query_embedder,
        llm_client=llm_client,
    )

    answer_generator = AnswerGenerator(llm_client)
    return retrieval_pipeline, answer_generator


def extract_answer_text(answer_result: dict) -> str:
    """Pull plain text out of the structured answer dict."""
    structured = answer_result.get("structured", {})
    if isinstance(structured, str):
        try:
            structured = json.loads(structured)
        except Exception:
            return structured

    parts = []
    if structured.get("title"):
        parts.append(structured["title"])
    if structured.get("subtitle"):
        parts.append(structured["subtitle"])
    for section in structured.get("sections", []):
        content = section.get("content", "")
        if content:
            parts.append(content)
        for item in section.get("items", []):
            parts.append(str(item))
    if structured.get("fallback"):
        parts.append(structured["fallback"])
    return " ".join(parts)


def run_eval(
    queries_path: Path,
    output_path: Path,
    split: str = None,
    dry_run: bool = False,
):
    queries = load_jsonl(queries_path)
    if split:
        queries = [q for q in queries if split in q.get("tags", [])]

    print(f"\nRunning eval on {len(queries)} queries...")
    if dry_run:
        print("(dry-run: skipping generation)")

    retrieval_pipeline, answer_generator = build_pipeline()

    results = []
    for i, q in enumerate(queries, 1):
        print(f"  [{i:02d}/{len(queries)}] {q['id']}: {q['query'][:60]}", end="", flush=True)

        t0 = time.time()
        try:
            retrieval = retrieval_pipeline.retrieve(q["query"], top_k=5)
        except Exception as e:
            print(f" ERROR (retrieval): {e}")
            results.append({"id": q["id"], "error": str(e)})
            continue
        t1 = time.time()

        answer_text = ""
        answer_result = {}
        if not dry_run and retrieval["chunks"]:
            try:
                answer_result = answer_generator.generate(
                    query=q["query"],
                    retrieved_chunks=retrieval["chunks"],
                    intent_type=retrieval["intent"],
                )
                answer_text = extract_answer_text(answer_result)
            except Exception as e:
                print(f" ERROR (generation): {e}")
        t2 = time.time()

        retrieved_ids = [str(c.get("chunk_id", "")) for c in retrieval["chunks"]]

        result = {
            "id": q["id"],
            "query": q["query"],
            "expected_intent": q["intent"],
            "detected_intent": retrieval["intent"],
            "intent_correct": retrieval["intent"] == q["intent"],
            "retrieved_chunk_ids": retrieved_ids,
            "retrieved_scores": [round(c.get("score", 0), 4) for c in retrieval["chunks"]],
            "recall_at_5": recall_at_k(retrieved_ids, q.get("gold_chunk_ids", []), 5),
            "mrr": mrr(retrieved_ids, q.get("gold_chunk_ids", [])),
            "answer_text": answer_text[:500],  # truncate for readability
            "must_include_pass": check_must_include(answer_text, q.get("must_include", [])),
            "must_not_include_pass": check_must_not_include(answer_text, q.get("must_not_include", [])),
            "answer_grounded": check_grounding(answer_text, retrieval["chunks"]),
            "latency_ms": {
                "retrieve": int((t1 - t0) * 1000),
                "generate": int((t2 - t1) * 1000),
            },
            "tags": q.get("tags", []),
        }
        results.append(result)

        status = "✓" if result["intent_correct"] else "✗"
        print(f" {status} intent={result['detected_intent']} chunks={len(retrieved_ids)}")

    summary = aggregate(results)
    summary["run_at"] = datetime.utcnow().isoformat()
    summary["queries_path"] = str(queries_path)

    write_jsonl(output_path, [summary] + results)

    # Also write to latest.jsonl
    latest = output_path.parent / "latest.jsonl"
    write_jsonl(latest, [summary] + results)

    print_table(summary)
    print(f"\nResults written to: {output_path}")
    return summary


def print_table(summary: dict):
    print("\n" + "=" * 60)
    print("EVAL SUMMARY")
    print("=" * 60)
    print(f"  Queries:              {summary.get('n', 0)}")
    print(f"  Recall@5:             {summary.get('recall_at_5', 0):.3f}")
    print(f"  MRR:                  {summary.get('mrr', 0):.3f}")
    print(f"  Intent Accuracy:      {summary.get('intent_accuracy', 0):.3f}")
    print(f"  Must-Include Pass:    {summary.get('must_include_pass_rate', 0):.3f}")
    print(f"  Grounding Pass:       {summary.get('grounding_pass_rate', 0):.3f}")
    print(f"  P50 Latency:          {summary.get('p50_latency_ms', 0)}ms")
    print(f"  P95 Latency:          {summary.get('p95_latency_ms', 0)}ms")

    per_intent = summary.get("per_intent", {})
    if per_intent:
        print("\n  Per-intent breakdown:")
        print(f"  {'Intent':<20} {'N':>4} {'Recall@5':>9} {'IntentAcc':>10} {'Grounding':>10}")
        print("  " + "-" * 58)
        for intent, m in sorted(per_intent.items()):
            print(
                f"  {intent:<20} {m['n']:>4} "
                f"{m['recall_at_5']:>9.3f} "
                f"{m['intent_accuracy']:>10.3f} "
                f"{m['grounding_pass_rate']:>10.3f}"
            )
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG eval harness")
    parser.add_argument(
        "--queries",
        default="eval/golden/queries.jsonl",
        help="Path to queries JSONL file",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (default: eval/results/YYYY-MM-DD_HH-MM.jsonl)",
    )
    parser.add_argument("--split", default=None, help="Filter by tag (e.g. 'hr', 'faculty')")
    parser.add_argument("--dry-run", action="store_true", help="Skip generation, retrieval only")
    args = parser.parse_args()

    queries_path = Path(args.queries)
    if args.output:
        output_path = Path(args.output)
    else:
        ts = datetime.utcnow().strftime("%Y-%m-%d_%H-%M")
        output_path = Path(f"eval/results/{ts}.jsonl")

    run_eval(queries_path, output_path, split=args.split, dry_run=args.dry_run)
