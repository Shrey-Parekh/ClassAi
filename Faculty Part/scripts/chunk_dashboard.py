#!/usr/bin/env python3
"""
Chunk-count sanity dashboard.

Walks the raw-documents directory, runs the chunker over every file, and
prints a per-document / per-chunk-type count table alongside simple sanity
thresholds. The goal is fast regression detection: a single chunker tweak
often moves these counts in either direction, and the dashboard makes that
movement visible without spinning up the embedding model or Qdrant.

Usage:
    python scripts/chunk_dashboard.py --input data/raw
    python scripts/chunk_dashboard.py --input data/raw --json

Exit codes:
    0 — all documents produced chunk counts within their expected ranges.
    1 — at least one document fell outside its expected chunk-count range.
    2 — the run crashed (bad path, broken chunker, etc.).

The expected ranges below reflect the known structure of the three seed
documents (Compendium, Employee Resource Book, Faculty Academic Guidelines)
as of 2026-04. When a new document is added the "expected" entry is None
so no sanity gating is applied — counts are still printed for review.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make `src.*` importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.document_processor import DocumentProcessor  # noqa: E402
from src.chunking.document_chunker import DocumentChunker  # noqa: E402


# Expected (min, max) total-chunk count per known document. None means no
# sanity check — we still print the counts so a human can eyeball them.
EXPECTED_TOTAL_RANGE: Dict[str, Optional[Tuple[int, int]]] = {
    "NMIMS_Faculty_Applications_Compendium.pdf": (15, 60),
    "NMIMS_Employee_Resource_Book_2024-25.pdf": (10, 200),
    "NMIMS_Faculty_Academic_Guidelines.pdf": (5, 100),
}

# Per-chunk-type minimum counts. A document is expected to produce AT LEAST
# this many chunks of the given type; falling below indicates a regression
# (e.g. dropping SECTION chunks or dropping form_section per-letter chunks).
# Chunk-type names come from document_chunker.py's metadata["chunk_type"].
EXPECTED_MIN_BY_TYPE: Dict[str, Dict[str, int]] = {
    "NMIMS_Faculty_Applications_Compendium.pdf": {
        "form_template": 5,   # 6 forms in the compendium (plus TOC)
        "form_section": 15,   # At minimum A/B across the 6 forms
    },
    "NMIMS_Employee_Resource_Book_2024-25.pdf": {
        "policy_section": 5,
    },
    "NMIMS_Faculty_Academic_Guidelines.pdf": {
        "guideline_section": 3,
    },
}


def analyze_document(
    file_path: Path,
    doc_processor: DocumentProcessor,
    chunker: DocumentChunker,
) -> Dict[str, Any]:
    """Run text extraction + chunking for one file and return summary stats."""
    summary: Dict[str, Any] = {
        "file": file_path.name,
        "source_type": None,
        "total_chunks": 0,
        "by_type": {},
        "avg_tokens": 0.0,
        "min_tokens": None,
        "max_tokens": None,
        "warnings": [],
        "error": None,
    }
    try:
        processed = doc_processor.process_document(
            file_path,
            {"doc_id": file_path.stem, "title": file_path.stem},
        )
        source_type = chunker.detect_source_type(file_path)
        chunks = chunker.chunk_document(
            text=processed["content"],
            filepath=file_path,
            doc_metadata=processed["metadata"],
        )
        type_counts: Dict[str, int] = defaultdict(int)
        token_counts: List[int] = []
        for c in chunks:
            ct = c.metadata.get("chunk_type", "unknown")
            type_counts[ct] += 1
            tokens = getattr(c, "token_count", None) or 0
            if tokens:
                token_counts.append(tokens)

        summary["source_type"] = source_type
        summary["total_chunks"] = len(chunks)
        summary["by_type"] = dict(type_counts)
        if token_counts:
            summary["avg_tokens"] = round(sum(token_counts) / len(token_counts), 1)
            summary["min_tokens"] = min(token_counts)
            summary["max_tokens"] = max(token_counts)
    except Exception as e:  # noqa: BLE001 — dashboard must keep going
        summary["error"] = f"{type(e).__name__}: {e}"
    return summary


def check_thresholds(summary: Dict[str, Any]) -> List[str]:
    """Compare a document summary against EXPECTED_* and return warnings."""
    warnings: List[str] = []
    name = summary["file"]

    if summary.get("error"):
        warnings.append(f"ERROR during chunking: {summary['error']}")
        return warnings

    # Total range check
    total_range = EXPECTED_TOTAL_RANGE.get(name)
    if total_range is not None:
        lo, hi = total_range
        total = summary["total_chunks"]
        if total < lo:
            warnings.append(
                f"total_chunks={total} is BELOW expected minimum {lo} "
                f"(regression: fewer chunks than before)"
            )
        elif total > hi:
            warnings.append(
                f"total_chunks={total} is ABOVE expected maximum {hi} "
                f"(regression: chunks may be over-fragmented)"
            )

    # Per-type minimums
    type_mins = EXPECTED_MIN_BY_TYPE.get(name, {})
    for ctype, min_n in type_mins.items():
        actual = summary["by_type"].get(ctype, 0)
        if actual < min_n:
            warnings.append(
                f"chunk_type={ctype!r}: only {actual} chunks "
                f"(expected >= {min_n})"
            )

    # Generic sanity: zero chunks is always a bug
    if summary["total_chunks"] == 0:
        warnings.append("zero chunks produced — document is empty or parsing failed")

    return warnings


def render_table(results: List[Dict[str, Any]]) -> str:
    """Produce a monospace dashboard table."""
    lines: List[str] = []
    lines.append("=" * 92)
    lines.append("CHUNK-COUNT SANITY DASHBOARD")
    lines.append("=" * 92)
    lines.append(
        f"{'Document':<44} {'Type':<22} {'Total':>6} {'Avg':>6} {'Min':>5} {'Max':>5}"
    )
    lines.append("-" * 92)
    for r in results:
        name = r["file"]
        src = (r["source_type"] or "—")[:22]
        total = r["total_chunks"]
        avg = r["avg_tokens"]
        mn = r["min_tokens"] if r["min_tokens"] is not None else "—"
        mx = r["max_tokens"] if r["max_tokens"] is not None else "—"
        lines.append(f"{name[:44]:<44} {src:<22} {total:>6} {avg:>6.1f} {str(mn):>5} {str(mx):>5}")

        # Per-type breakdown
        by_type = r.get("by_type", {})
        for ctype in sorted(by_type):
            lines.append(f"    • {ctype:<40} {by_type[ctype]:>6}")

        # Warnings
        for w in r.get("warnings", []):
            lines.append(f"    ⚠ {w}")
    lines.append("=" * 92)

    # Summary
    n_warnings = sum(len(r.get("warnings", [])) for r in results)
    n_files = len(results)
    n_ok = n_files - sum(1 for r in results if r.get("warnings"))
    lines.append(f"Files analysed : {n_files}")
    lines.append(f"Files healthy  : {n_ok}")
    lines.append(f"Warnings total : {n_warnings}")
    lines.append("=" * 92)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw"),
        help="Directory to scan (default: data/raw)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the human table.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="**/*.pdf",
        help="glob pattern relative to --input (default: **/*.pdf)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: --input path does not exist: {args.input}", file=sys.stderr)
        return 2

    dp = DocumentProcessor()
    chunker = DocumentChunker()

    files = sorted(args.input.glob(args.pattern))
    if not files:
        print(f"No files matched pattern {args.pattern!r} under {args.input}")
        return 2

    results: List[Dict[str, Any]] = []
    for fp in files:
        summary = analyze_document(fp, dp, chunker)
        summary["warnings"] = check_thresholds(summary)
        results.append(summary)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(render_table(results))

    any_warnings = any(r.get("warnings") for r in results)
    return 1 if any_warnings else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(2)
a: BLE001
        traceback.print_exc()
        sys.exit(2)
