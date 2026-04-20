#!/usr/bin/env python3
"""
Eval harness for the faculty RAG pipeline.

Drives a YAML ground-truth set against the retrieval + generation stack
and reports aggregate metrics — recall@k, intent accuracy, citation-
grounded ratio, and simple keyword presence — to stdout and (optionally)
a JSON report. The harness deliberately avoids any network-call or
LLM-side assertions beyond the coarse keyword checks: those are cheap to
maintain and catch regressions when chunking or retrieval change.

Usage:
    python scripts/eval_harness.py                              # default paths
    python scripts/eval_harness.py --queries data/eval/ground_truth.yaml
    python scripts/eval_harness.py --report data/eval/report.json --top-k 15
    python scripts/eval_harness.py --intent-only                # retrieval-free
    python scripts/eval_harness.py --retrieval-only             # skip LLM

Exit codes:
    0 — all metrics met their gate (intent_acc >= 0.70, recall@k >= 0.70,
        grounding_avg >= 0.50, keyword_any_hit >= 0.70).
    1 — at least one gate failed.
    2 — harness itself errored (missing file, import failure, etc).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make src.* importable
sys.path.insert(0, str(Path(__file__).parent.parent))


def _require_yaml():
    """Import PyYAML lazily — the harness is useful even on systems where
    PyYAML isn't installed if the user passes --queries pointing at JSON."""
    try:
        import yaml  # type: ignore
        return yaml
    except ImportError:
        print(
            "[ERROR] PyYAML not available. Install with `pip install pyyaml` "
            "or convert data/eval/ground_truth.yaml to JSON.",
            file=sys.stderr,
        )
        sys.exit(2)


def load_queries(path: Path) -> List[Dict[str, Any]]:
    """Load the ground-truth set from YAML or JSON based on suffix."""
    if not path.exists():
        print(f"[ERROR] queries file not found: {path}", file=sys.stderr)
        sys.exit(2)
    if path.suffix.lower() in {".yaml", ".yml"}:
        yaml = _require_yaml()
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    queries = data.get("queries", []) if isinstance(data, dict) else data
    if not isinstance(queries, list):
        print(f"[ERROR] queries root must be a list, got {type(queries).__name__}", file=sys.stderr)
        sys.exit(2)
    return queries


def _answer_text(structured: Any) -> str:
    """Flatten a StructuredResponse (or its dict form) into one string.

    We join title + subtitle + every section's content + every item text
    so the keyword checks apply uniformly across paragraph/bullets/steps
    sections. Non-string items (dicts, None) are stringified or skipped.
    """
    if structured is None:
        return ""
    if hasattr(structured, "model_dump"):
        structured = structured.model_dump()
    elif hasattr(structured, "dict"):
        structured = structured.dict()
    if not isinstance(structured, dict):
        return str(structured)

    parts: List[str] = []
    for key in ("title", "subtitle", "summary", "footer", "fallback"):
        val = structured.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val)

    for section in structured.get("sections", []) or []:
        if not isinstance(section, dict):
            continue
        heading = section.get("heading")
        if isinstance(heading, str):
            parts.append(heading)
        content = section.get("content")
        if isinstance(content, str):
            parts.append(content)
        for item in section.get("items", []) or []:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                for k in ("text", "content", "description", "body"):
                    v = item.get(k)
                    if isinstance(v, str):
                        parts.append(v)
                        break
    return " ".join(parts)


def _document_hit(chunks: List[Dict[str, Any]], needle: str) -> bool:
    """Does any retrieved chunk's document_name contain ``needle``?

    Empty ``needle`` means "no document gate" — always True so the query
    is still checked for other signals (keywords, grounding, etc.).
    """
    if not needle:
        return True
    needle_l = needle.lower()
    for c in chunks:
        name = (c.get("metadata", {}) or {}).get("document_name", "") or ""
        if needle_l in name.lower():
            return True
    return False


def _keyword_hit(text: str, keywords: Optional[List[str]], require_all: bool) -> Tuple[bool, List[str]]:
    """Check keyword presence in ``text``.

    Returns (hit, missing). For ``require_all=False`` the first match counts;
    missing is the list of keywords that didn't appear (only useful when
    require_all=True or when everything missed).
    """
    if not keywords:
        return True, []
    t = (text or "").lower()
    missing: List[str] = []
    any_hit = False
    for kw in keywords:
        if not isinstance(kw, str) or not kw:
            continue
        if kw.lower() in t:
            any_hit = True
        else:
            missing.append(kw)
    if require_all:
        return (len(missing) == 0), missing
    return any_hit, missing


def _summarise(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-query results into coarse scoreboard metrics."""
    n = len(results)
    if n == 0:
        return {"n_queries": 0}

    intent_ok = sum(1 for r in results if r.get("intent_ok"))
    doc_ok = sum(1 for r in results if r.get("document_ok"))
    kw_any_ok = sum(1 for r in results if r.get("keywords_any_ok"))
    kw_all_ok = sum(1 for r in results if r.get("keywords_all_ok", True))  # default True when not set
    grounded = [r.get("grounding_ratio") for r in results if r.get("grounding_ratio") is not None]
    errors = [r for r in results if r.get("error")]

    return {
        "n_queries": n,
        "intent_accuracy": round(intent_ok / n, 3),
        "recall_at_k": round(doc_ok / n, 3),
        "keyword_any_hit": round(kw_any_ok / n, 3),
        "keyword_all_hit": round(kw_all_ok / n, 3),
        "grounding_avg": round(sum(grounded) / len(grounded), 3) if grounded else None,
        "n_errors": len(errors),
    }


def _format_row(r: Dict[str, Any]) -> str:
    """Single-line compact per-query row for the human table."""
    marks = [
        "✓" if r.get("intent_ok") else "✗",
        "✓" if r.get("document_ok") else "✗",
        "✓" if r.get("keywords_any_ok") else "✗",
    ]
    grounding = r.get("grounding_ratio")
    gstr = f"{grounding:.2f}" if grounding is not None else "—"
    return (
        f"{r['id'][:22]:<22} "
        f"intent={r.get('intent_got','?')[:12]:<12} "
        f"doc_hit={marks[1]} "
        f"intent_hit={marks[0]} "
        f"kw_hit={marks[2]} "
        f"grounding={gstr:>4}"
    )


def run_query(
    pipeline,
    answer_generator,
    query: str,
    top_k: int,
    retrieval_only: bool,
) -> Dict[str, Any]:
    """Run one query through the pipeline and return structured results."""
    t0 = time.perf_counter()
    retrieval_result = pipeline.retrieve(query=query, top_k=top_k)
    t_retrieve = time.perf_counter() - t0

    chunks = retrieval_result.get("chunks", [])
    intent = retrieval_result.get("intent", "general")

    out: Dict[str, Any] = {
        "intent_got": intent,
        "chunks": chunks,
        "latency_retrieval_ms": round(t_retrieve * 1000, 1),
    }

    if retrieval_only or not chunks or answer_generator is None:
        out["answer_text"] = ""
        out["grounding"] = None
        return out

    t1 = time.perf_counter()
    ans = answer_generator.generate(query=query, retrieved_chunks=chunks, intent_type=intent)
    out["latency_generation_ms"] = round((time.perf_counter() - t1) * 1000, 1)
    out["answer_text"] = _answer_text(ans.get("structured"))
    out["grounding"] = ans.get("grounding")
    out["confidence"] = ans.get("confidence")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("data/eval/ground_truth.yaml"),
        help="YAML/JSON file with ground-truth queries",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path to write the full JSON report",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Retrieval top_k")
    parser.add_argument(
        "--intent-only",
        action="store_true",
        help="Skip retrieval — only measure intent classifier accuracy",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip LLM generation — only measure retrieval metrics",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Run only the first N queries (0 = all). Useful for smoke tests.",
    )
    args = parser.parse_args()

    queries = load_queries(args.queries)
    if args.limit > 0:
        queries = queries[: args.limit]
    print(f"Loaded {len(queries)} queries from {args.queries}")

    if args.intent_only:
        # Lightweight path — only touches the intent classifier.
        from src.retrieval.query_understanding import QueryAnalyzer
        qa = QueryAnalyzer()
        results: List[Dict[str, Any]] = []
        for q in queries:
            u = qa.analyze(q["query"])
            intent_ok = u.intent == q.get("intent_expected")
            results.append({
                "id": q["id"],
                "query": q["query"],
                "intent_got": u.intent,
                "intent_ok": intent_ok,
                "document_ok": True,           # gate skipped in intent-only mode
                "keywords_any_ok": True,
                "keywords_all_ok": True,
                "grounding_ratio": None,
            })
    else:
        # Full pipeline path — requires the LLM + vector DB to be up.
        try:
            from dotenv import load_dotenv  # optional
            load_dotenv()
        except ImportError:
            pass
        from src.retrieval.pipeline import RetrievalPipeline
        from src.utils.vector_db import VectorDBClient
        from src.utils.llm import LLMClient
        from src.generation.answer_generator import AnswerGenerator

        vector_db = VectorDBClient(collection_name="faculty_chunks")
        llm_client = LLMClient()
        pipeline = RetrievalPipeline(vector_db_client=vector_db, llm_client=llm_client)
        # Some deployments build the BM25 index lazily — nudge it here.
        try:
            pipeline.search_engine.build_bm25_index()
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] BM25 index build skipped: {e}")

        answer_generator = None if args.retrieval_only else AnswerGenerator(llm_client)

        results = []
        for i, q in enumerate(queries, 1):
            print(f"[{i}/{len(queries)}] {q['id']}: {q['query'][:60]!r}")
            row: Dict[str, Any] = {
                "id": q["id"],
                "query": q["query"],
                "intent_ok": False,
                "document_ok": False,
                "keywords_any_ok": False,
                "keywords_all_ok": True,
                "grounding_ratio": None,
                "error": None,
            }
            try:
                out = run_query(
                    pipeline=pipeline,
                    answer_generator=answer_generator,
                    query=q["query"],
                    top_k=args.top_k,
                    retrieval_only=args.retrieval_only,
                )
                row.update(out)
                row["intent_got"] = out["intent_got"]
                row["intent_ok"] = out["intent_got"] == q.get("intent_expected")
                row["document_ok"] = _document_hit(out["chunks"], q.get("document_expected", ""))
                row["keywords_any_ok"], row["keywords_any_missing"] = _keyword_hit(
                    out.get("answer_text", ""), q.get("keywords_any"), require_all=False
                )
                if q.get("keywords_all"):
                    row["keywords_all_ok"], row["keywords_all_missing"] = _keyword_hit(
                        out.get("answer_text", ""), q.get("keywords_all"), require_all=True
                    )
                grounding = out.get("grounding") or {}
                row["grounding_ratio"] = grounding.get("ratio")
                # Drop chunk payload from the row — keep the report lean.
                row.pop("chunks", None)
            except Exception as e:  # noqa: BLE001
                row["error"] = f"{type(e).__name__}: {e}"
                row["traceback"] = traceback.format_exc(limit=4)
                print(f"   [ERROR] {row['error']}")
            results.append(row)

    # --- Reporting -----------------------------------------------------------
    print("\n" + "=" * 88)
    print("PER-QUERY RESULTS")
    print("=" * 88)
    for r in results:
        print(_format_row(r))

    summary = _summarise(results)
    print("\n" + "=" * 88)
    print("SUMMARY")
    print("=" * 88)
    for k, v in summary.items():
        print(f"  {k:20s} {v}")
    print("=" * 88)

    if args.report:
        report = {"summary": summary, "results": results}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Wrote full report to {args.report}")

    # Gates — tuned to today's known baseline. Loosen when the corpus
    # grows; tighten when the pipeline gets better.
    gates = {
        "intent_accuracy": 0.70,
        "recall_at_k": 0.70,
        "grounding_avg": 0.50,
        "keyword_any_hit": 0.70,
    }
    failed = []
    for metric, threshold in gates.items():
        value = summary.get(metric)
        if value is None:
            continue
        if value < threshold:
            failed.append(f"{metric}={value} < {threshold}")

    if failed:
        print("\nGATES FAILED:")
        for f in failed:
            print(f"  ✗ {f}")
        return 1
    print("\nAll gates passed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(2)
