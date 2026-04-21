"""
Retrieval and answer quality metrics.
"""

from typing import List, Dict, Any, Optional
import re


def recall_at_k(retrieved_ids: List[str], gold_ids: List[str], k: int = 5) -> float:
    """Fraction of gold chunks found in top-k retrieved chunks."""
    if not gold_ids:
        return 1.0  # No gold defined — treat as pass
    retrieved_set = set(str(i) for i in retrieved_ids[:k])
    gold_set = set(str(i) for i in gold_ids)
    hits = len(retrieved_set & gold_set)
    return hits / len(gold_set)


def mrr(retrieved_ids: List[str], gold_ids: List[str]) -> float:
    """Mean Reciprocal Rank — rank of first gold chunk in retrieved list."""
    if not gold_ids:
        return 1.0
    gold_set = set(str(i) for i in gold_ids)
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if str(chunk_id) in gold_set:
            return 1.0 / rank
    return 0.0


def check_must_include(answer_text: str, must_include: List[str]) -> bool:
    """All must_include strings appear (case-insensitive) in answer."""
    text_lower = answer_text.lower()
    return all(s.lower() in text_lower for s in must_include)


def check_must_not_include(answer_text: str, must_not_include: List[str]) -> bool:
    """None of must_not_include strings appear in answer."""
    text_lower = answer_text.lower()
    return all(s.lower() not in text_lower for s in must_not_include)


def check_grounding(answer_text: str, chunks: List[Dict[str, Any]], threshold: float = 0.25) -> bool:
    """
    Cheap n-gram grounding check — no LLM required.

    Computes 5-gram overlap between answer and retrieved chunks.
    Returns True if coverage >= threshold.
    """
    if not answer_text or not answer_text.strip():
        return False

    def _ngrams(text: str, n: int = 5) -> set:
        tokens = re.findall(r'\b\w+\b', text.lower())
        return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}

    answer_shingles = _ngrams(answer_text)
    if not answer_shingles:
        return False

    chunk_shingles: set = set()
    for chunk in chunks:
        chunk_text = chunk.get("text", "") or chunk.get("content", "") or ""
        chunk_shingles |= _ngrams(chunk_text)

    coverage = len(answer_shingles & chunk_shingles) / len(answer_shingles)
    return coverage >= threshold


def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate metrics over all results."""
    if not results:
        return {}

    def _mean(values):
        vals = [v for v in values if v is not None]
        return sum(vals) / len(vals) if vals else 0.0

    def _percentile(values, p):
        if not values:
            return 0
        sorted_vals = sorted(values)
        idx = int(len(sorted_vals) * p / 100)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    latencies = [
        r["latency_ms"]["retrieve"] + r["latency_ms"]["generate"]
        for r in results
        if "latency_ms" in r
    ]

    summary = {
        "n": len(results),
        "recall_at_5": _mean(r.get("recall_at_5") for r in results),
        "mrr": _mean(r.get("mrr") for r in results),
        "intent_accuracy": _mean(r.get("intent_correct") for r in results),
        "must_include_pass_rate": _mean(r.get("must_include_pass") for r in results),
        "must_not_include_pass_rate": _mean(r.get("must_not_include_pass") for r in results),
        "grounding_pass_rate": _mean(r.get("answer_grounded") for r in results),
        "p50_latency_ms": _percentile(latencies, 50),
        "p95_latency_ms": _percentile(latencies, 95),
        "per_intent": _group_by_intent(results),
    }
    return summary


def _group_by_intent(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Break down metrics by expected intent."""
    from collections import defaultdict

    groups: Dict[str, list] = defaultdict(list)
    for r in results:
        groups[r.get("expected_intent", "unknown")].append(r)

    out = {}
    for intent, group in groups.items():
        out[intent] = {
            "n": len(group),
            "recall_at_5": sum(r.get("recall_at_5", 0) for r in group) / len(group),
            "intent_accuracy": sum(r.get("intent_correct", 0) for r in group) / len(group),
            "must_include_pass_rate": sum(r.get("must_include_pass", 0) for r in group) / len(group),
            "grounding_pass_rate": sum(r.get("answer_grounded", 0) for r in group) / len(group),
        }
    return out
