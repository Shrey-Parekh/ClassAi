"""
Compare two eval runs and surface regressions and wins.

Usage:
    python eval/compare.py eval/results/baseline.jsonl eval/results/latest.jsonl
    python eval/compare.py eval/results/baseline.jsonl eval/results/latest.jsonl --fail-threshold 0.02
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


def load_jsonl(path: Path) -> Tuple[dict, List[dict]]:
    """Returns (summary, results_list)."""
    with open(path, encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    if not lines:
        raise ValueError(f"Empty file: {path}")
    summary = lines[0] if "n" in lines[0] else {}
    results = lines[1:] if summary else lines
    return summary, results


PASS_KEYS = ["recall_at_5", "intent_correct", "must_include_pass", "must_not_include_pass", "answer_grounded"]


def is_passing(result: dict) -> bool:
    """A result is 'passing' if all binary checks pass and recall > 0."""
    return (
        result.get("recall_at_5", 1.0) > 0
        and result.get("intent_correct", True)
        and result.get("must_include_pass", True)
        and result.get("must_not_include_pass", True)
    )


def compare(baseline_path: Path, latest_path: Path, fail_threshold: float = 0.02) -> int:
    """
    Compare two runs. Returns exit code: 0 = ok, 1 = regression detected.
    """
    base_summary, base_results = load_jsonl(baseline_path)
    new_summary, new_results = load_jsonl(latest_path)

    base_by_id = {r["id"]: r for r in base_results if "id" in r}
    new_by_id = {r["id"]: r for r in new_results if "id" in r}

    regressions = []
    wins = []
    neutral = []

    for qid in sorted(set(base_by_id) | set(new_by_id)):
        base = base_by_id.get(qid)
        new = new_by_id.get(qid)

        if base is None or new is None:
            continue

        base_pass = is_passing(base)
        new_pass = is_passing(new)

        if base_pass and not new_pass:
            changes = []
            for k in PASS_KEYS:
                bv = base.get(k)
                nv = new.get(k)
                if bv != nv:
                    changes.append(f"{k}: {bv} → {nv}")
            regressions.append((qid, base.get("query", ""), changes))
        elif not base_pass and new_pass:
            wins.append((qid, base.get("query", "")))
        else:
            neutral.append(qid)

    # Print report
    print("\n" + "=" * 70)
    print(f"COMPARISON: {baseline_path.name}  →  {latest_path.name}")
    print("=" * 70)

    if regressions:
        print(f"\n🔴 REGRESSIONS ({len(regressions)} queries were passing, now failing):")
        for qid, query, changes in regressions:
            print(f"  {qid}: {query[:60]}")
            for c in changes:
                print(f"    • {c}")
    else:
        print("\n✅ No regressions")

    if wins:
        print(f"\n🟢 WINS ({len(wins)} queries were failing, now passing):")
        for qid, query in wins:
            print(f"  {qid}: {query[:60]}")

    print(f"\n⚪ Neutral: {len(neutral)} queries unchanged")

    # Aggregate delta
    def _get(s, k):
        return s.get(k, 0) if s else 0

    metrics = ["recall_at_5", "mrr", "intent_accuracy", "grounding_pass_rate"]
    print("\nOverall metric delta:")
    for m in metrics:
        bv = _get(base_summary, m)
        nv = _get(new_summary, m)
        delta = nv - bv
        sign = "+" if delta >= 0 else ""
        print(f"  {m:<30} {bv:.3f} → {nv:.3f}  ({sign}{delta*100:.1f}%)")

    latency_delta = _get(new_summary, "p50_latency_ms") - _get(base_summary, "p50_latency_ms")
    sign = "+" if latency_delta >= 0 else ""
    print(f"  {'p50_latency_ms':<30} {_get(base_summary, 'p50_latency_ms')}ms → {_get(new_summary, 'p50_latency_ms')}ms  ({sign}{latency_delta}ms)")

    # CI gate
    recall_delta = _get(new_summary, "recall_at_5") - _get(base_summary, "recall_at_5")
    exit_code = 0
    if recall_delta < -fail_threshold:
        print(f"\n❌ CI FAIL: Recall@5 dropped by {abs(recall_delta)*100:.1f}% (threshold: {fail_threshold*100:.1f}%)")
        exit_code = 1
    if regressions:
        print(f"\n❌ CI FAIL: {len(regressions)} regression(s) detected")
        exit_code = 1
    if exit_code == 0:
        print("\n✅ CI PASS")

    print("=" * 70)
    return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two eval runs")
    parser.add_argument("baseline", help="Baseline JSONL results file")
    parser.add_argument("latest", help="Latest JSONL results file")
    parser.add_argument(
        "--fail-threshold",
        type=float,
        default=0.02,
        help="Recall@5 drop threshold that triggers CI failure (default: 0.02 = 2%%)",
    )
    args = parser.parse_args()
    code = compare(Path(args.baseline), Path(args.latest), args.fail_threshold)
    sys.exit(code)
