"""Streaming audit and leakage-safe sampling of EnterpriseRAG-Bench Parquet files."""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import closing
from hashlib import sha256
import heapq
import json
from pathlib import Path
import sqlite3
import tempfile
from typing import Any

import pyarrow.parquet as pq


CORPUS_FIELDS = frozenset({"doc_id", "source_type", "title", "content"})
QUESTION_FIELDS = frozenset({
    "question_id", "question_type", "source_types", "question",
    "expected_doc_ids", "gold_answer", "answer_facts",
})
EVALUATION_ONLY_FIELDS = QUESTION_FIELDS - {"question_id", "question"}


def _file_info(path: Path) -> dict[str, Any]:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    parquet = pq.ParquetFile(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "rows_from_metadata": parquet.metadata.num_rows,
        "schema": {field.name: str(field.type) for field in parquet.schema_arrow},
        "row_groups": parquet.metadata.num_row_groups,
    }


def _require_schema(path: Path, required: frozenset[str], forbidden: frozenset[str]) -> pq.ParquetFile:
    parquet = pq.ParquetFile(path)
    fields = set(parquet.schema_arrow.names)
    missing = required - fields
    leaking = forbidden & fields
    if missing or leaking:
        raise ValueError(f"Unsafe or incomplete {path}: missing={sorted(missing)}, forbidden={sorted(leaking)}")
    return parquet


def corpus_batches(path: Path, batch_size: int = 1024):
    """Yield only retrieval fields; reject any benchmark answer or label column."""
    parquet = _require_schema(path, CORPUS_FIELDS, QUESTION_FIELDS)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=sorted(CORPUS_FIELDS)):
        yield batch.to_pylist()


def audit(documents: Path, questions: Path, batch_size: int = 1024) -> dict[str, Any]:
    """Scan every row while storing IDs and lengths in temporary SQLite, not RAM."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    doc_meta = _file_info(documents)
    question_meta = _file_info(questions)
    _require_schema(documents, CORPUS_FIELDS, QUESTION_FIELDS)
    question_file = _require_schema(questions, QUESTION_FIELDS, frozenset())
    missing_docs: Counter[str] = Counter()
    missing_questions: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    question_types: Counter[str] = Counter()
    duplicate_docs = 0
    duplicate_questions = 0
    doc_rows = 0
    question_rows = 0
    missing_expected_ids = 0
    with tempfile.TemporaryDirectory(prefix="ear-audit-") as scratch:
        with closing(sqlite3.connect(Path(scratch) / "audit.sqlite")) as db:
            db.execute("CREATE TABLE ids (doc_id TEXT PRIMARY KEY)")
            db.execute("CREATE TABLE lengths (source TEXT NOT NULL, chars INTEGER NOT NULL)")
            db.execute("CREATE TABLE question_ids (question_id TEXT PRIMARY KEY)")
            for batch in corpus_batches(documents, batch_size):
                for row in batch:
                    doc_rows += 1
                    for field in CORPUS_FIELDS:
                        if row[field] is None or row[field] == "":
                            missing_docs[field] += 1
                    doc_id = row["doc_id"]
                    if doc_id:
                        try:
                            db.execute("INSERT INTO ids VALUES (?)", (doc_id,))
                        except sqlite3.IntegrityError:
                            duplicate_docs += 1
                    source = row["source_type"] or "<missing>"
                    source_counts[source] += 1
                    db.execute("INSERT INTO lengths VALUES (?, ?)", (source, len(row["content"] or "")))
            db.commit()
            db.execute("CREATE INDEX length_order ON lengths(source, chars)")
            for batch in question_file.iter_batches(batch_size=batch_size):
                for row in batch.to_pylist():
                    question_rows += 1
                    for field in QUESTION_FIELDS:
                        if row[field] is None or row[field] == "":
                            missing_questions[field] += 1
                    if row["question_id"]:
                        try:
                            db.execute("INSERT INTO question_ids VALUES (?)", (row["question_id"],))
                        except sqlite3.IntegrityError:
                            duplicate_questions += 1
                    question_types[row["question_type"] or "<missing>"] += 1
                    for doc_id in row["expected_doc_ids"] or []:
                        if not db.execute("SELECT 1 FROM ids WHERE doc_id = ?", (doc_id,)).fetchone():
                            missing_expected_ids += 1
            lengths = {}
            for source, count in sorted(source_counts.items()):
                minimum, maximum = db.execute(
                    "SELECT MIN(chars), MAX(chars) FROM lengths WHERE source = ?", (source,)
                ).fetchone()
                def quantile(fraction: float) -> int:
                    offset = int((count - 1) * fraction)
                    return db.execute(
                        "SELECT chars FROM lengths WHERE source = ? ORDER BY chars LIMIT 1 OFFSET ?",
                        (source, offset),
                    ).fetchone()[0]
                lengths[source] = {
                    "count": count, "min": minimum, "p50": quantile(0.5),
                    "p95": quantile(0.95), "max": maximum,
                }
    return {
        "scan_complete": doc_rows == doc_meta["rows_from_metadata"] and question_rows == question_meta["rows_from_metadata"],
        "documents": {**doc_meta, "rows_scanned": doc_rows, "source_counts": dict(sorted(source_counts.items())),
                      "missing_fields": dict(sorted(missing_docs.items())), "duplicate_doc_ids": duplicate_docs,
                      "content_length_chars": lengths},
        "questions": {**question_meta, "rows_scanned": question_rows,
                      "question_types": dict(sorted(question_types.items())),
                      "missing_fields": dict(sorted(missing_questions.items())),
                      "duplicate_question_ids": duplicate_questions,
                      "expected_ids_missing_from_corpus": missing_expected_ids},
        "boundary": "Only doc_id, source_type, title, content may enter retrieval; question labels and answers remain evaluation-only.",
    }


def _score(seed: int, doc_id: str) -> int:
    return int.from_bytes(sha256(f"{seed}:{doc_id}".encode()).digest(), "big")


def create_sample(documents: Path, questions: Path, output: Path, *, per_source: int = 21,
                  dev_questions: int = 10, seed: int = 2026,
                  length_limits: dict[str, dict[str, int]] | None = None) -> dict[str, Any]:
    """Select a deterministic, source-balanced sample and a covered dev question set.

    Supporting documents are selected in a separate evaluation step. Only the
    four corpus fields are written to the runtime Parquet file.
    """
    if per_source < 3 or dev_questions < 1:
        raise ValueError("per_source and dev_questions must be positive")
    if not length_limits:
        raise ValueError("Pass measured per-source p50/p95 length limits from an audit report")
    _require_schema(documents, CORPUS_FIELDS, QUESTION_FIELDS)
    question_file = _require_schema(questions, QUESTION_FIELDS, frozenset())
    questions_by_id = {}
    for batch in question_file.iter_batches(batch_size=1024):
        for row in batch.to_pylist():
            questions_by_id[row["question_id"]] = row
    eligible = sorted(
        (q for q in questions_by_id.values() if q["expected_doc_ids"]),
        key=lambda q: (_score(seed, q["question_id"]), q["question_id"]),
    )[:dev_questions]
    required_ids = {doc_id for q in eligible for doc_id in q["expected_doc_ids"]}
    heaps: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    required_rows = {}
    for batch in corpus_batches(documents):
        for row in batch:
            doc_id, source = row["doc_id"], row["source_type"]
            if not doc_id or not source or not row["content"]:
                continue
            if doc_id in required_ids:
                required_rows[doc_id] = row
            if source not in length_limits:
                raise ValueError(f"No measured length limits for source: {source}")
            chars = len(row["content"])
            limits = length_limits[source]
            stratum = "short" if chars <= limits["p50"] else "medium" if chars <= limits["p95"] else "long"
            priority = -_score(seed, doc_id)
            heap = heaps[(source, stratum)]
            item = (priority, doc_id)
            quota = per_source // 3 + (1 if ("short", "medium", "long").index(stratum) < per_source % 3 else 0)
            if len(heap) < quota:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
    if required_ids - required_rows.keys():
        raise ValueError(f"Missing support docs for dev questions: {sorted(required_ids - required_rows.keys())[:5]}")
    selected_ids = required_ids | {doc_id for heap in heaps.values() for _, doc_id in heap}
    selected_rows = {}
    for batch in corpus_batches(documents):
        for row in batch:
            if row["doc_id"] in selected_ids:
                safe_row = {field: row[field] for field in sorted(CORPUS_FIELDS)}
                if row["doc_id"] in selected_rows and selected_rows[row["doc_id"]] != safe_row:
                    raise ValueError(f"Conflicting duplicate document ID in sample: {row['doc_id']}")
                selected_rows[row["doc_id"]] = safe_row
    output.mkdir(parents=True, exist_ok=True)
    import pyarrow as pa
    pq.write_table(pa.Table.from_pylist([selected_rows[key] for key in sorted(selected_rows)]), output / "documents.parquet")
    # The question-only file is for manual dev queries. Gold labels stay outside runtime output.
    (output / "dev_questions.json").write_text(json.dumps(
        [{"question_id": q["question_id"], "question": q["question"]} for q in eligible], indent=2
    ), encoding="utf-8")
    manifest = {
        "seed": seed, "per_source": per_source, "dev_question_limit": dev_questions,
        "length_strata": "Per-source short <= measured p50; medium <= measured p95; long > measured p95",
        "corpus_sha256": _file_info(documents)["sha256"],
        "questions_sha256": _file_info(questions)["sha256"],
        "sample_doc_ids": sorted(selected_ids),
        "source_counts": dict(sorted(Counter(r["source_type"] for r in selected_rows.values()).items())),
        "dev_question_ids": [q["question_id"] for q in eligible],
        "dev_support_coverage": {q["question_id"]: len(q["expected_doc_ids"]) for q in eligible},
        "held_out_questions": len(questions_by_id) - len(eligible),
        "limitation": "Reduced corpus; retrieval metrics do not represent full-corpus performance.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
