"""Evaluate answer completion on the existing 212-document coverage index.

Run from the project root: python scripts/evaluate_generation_v3.py
This script verifies the isolated index and does not ingest or alter documents.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import re
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from enterprise_rag.dataset import corpus_batches
from enterprise_rag.providers import (
    AnswerGenerationStatusError, OpenAIAnswerModel, PgvectorSearch,
    SentenceTransformerEmbeddings, answer_citations, configured_answer_model,
    connect_local_postgres, local_settings,
)
from scripts.evaluate_coverage import (
    DATABASE, OUTPUT_DIR, QUESTIONS, ROOT, TARGET_IDS, database_dsn,
    file_hash, read_baseline,
)


RESULTS = ROOT / "rag_test_results_v3.txt"
V2 = ROOT / "rag_test_results_v2.txt"


def read_v2() -> list[dict]:
    text = V2.read_text(encoding="utf-8")
    sections = re.split(r"(?m)^QUESTION ([123])\s*$", text)[1:]
    found = {}
    for number, section in zip(sections[::2], sections[1::2]):
        def field(name: str) -> str:
            match = re.search(rf"(?m)^{re.escape(name)}: (.*)$", section)
            return match.group(1).strip() if match else ""
        answer_match = re.search(r"Generated answer:\s*\n(.*?)\n\s*Source citations in generated answer:", section, re.S)
        hits = re.findall(r"(?m)^  Rank (\d+): (dsid_[0-9a-f]+) \|.*?similarity=([0-9.]+)", section)
        found[int(number)] = {
            "question": field("Question"),
            "answer": answer_match.group(1).strip() if answer_match else "",
            "citations": field("Source citations in generated answer"),
            "response_time": field("Response time"),
            "errors": field("Errors"),
            "hits": [(int(rank), doc_id, float(score)) for rank, doc_id, score in hits],
        }
    if set(found) != {1, 2, 3} or any(found[i]["question"] != QUESTIONS[i - 1] for i in found):
        raise ValueError("V2 does not contain the exact three evaluation questions")
    return [found[i] for i in (1, 2, 3)]


def verify_existing_index(settings: dict, embedding: SentenceTransformerEmbeddings) -> PgvectorSearch:
    corpus_path = OUTPUT_DIR / "documents.parquet"
    manifest = json.loads((OUTPUT_DIR / "manifest.json").read_text(encoding="utf-8"))
    if file_hash(corpus_path) != manifest["evaluation_parquet_sha256"]:
        raise ValueError("Evaluation corpus hash changed")
    doc_ids = [row["doc_id"] for batch in corpus_batches(corpus_path) for row in batch]
    if len(doc_ids) != 212 or len(set(doc_ids)) != 212 or set(doc_ids) != set(manifest["evaluation_doc_ids"]):
        raise ValueError("Evaluation corpus is not the expected 212 unique documents")
    dsn = database_dsn(settings["pg_dsn"], DATABASE)
    db = connect_local_postgres(dsn)
    try:
        cursor = db.cursor()
        cursor.execute("SELECT doc_id, chunk_id FROM document_vectors")
        rows = cursor.fetchall()
    finally:
        db.close()
    if len(rows) != 212 or {row[0] for row in rows} != set(doc_ids):
        raise ValueError("Isolated index does not match the evaluation corpus")
    if any(row[1] != f"{row[0]}#0" for row in rows):
        raise ValueError("Isolated index chunk IDs changed")
    return PgvectorSearch(dsn, embedding.dimension)


def evaluate(search: PgvectorSearch, embedding: SentenceTransformerEmbeddings,
             llm: OpenAIAnswerModel) -> list[dict]:
    results = []
    for question in QUESTIONS:
        started = perf_counter()
        result = {"question": question, "answer": "", "citations": [], "hits": [],
                  "generation_metadata": None, "stage_time": {}, "errors": []}
        try:
            tick = perf_counter()
            vector = embedding.encode([question])[0]
            result["stage_time"]["embedding"] = perf_counter() - tick
            tick = perf_counter()
            hits = search.search(vector, limit=5)
            result["stage_time"]["retrieval"] = perf_counter() - tick
            result["hits"] = hits
            if not hits:
                raise ValueError("No documents retrieved")
            tick = perf_counter()
            try:
                result["answer"] = llm.answer(question, hits)
            except AnswerGenerationStatusError as exc:
                result["answer"] = exc.partial_answer
                result["errors"].append(str(exc))
            finally:
                result["stage_time"]["generation"] = perf_counter() - tick
                result["generation_metadata"] = llm.last_generation
            result["citations"] = sorted(set(re.findall(r"\[([A-Za-z0-9_-]+)\]", result["answer"])))
            try:
                answer_citations(result["answer"], hits)
            except ValueError as exc:
                result["errors"].append(str(exc))
        except Exception as exc:
            # API exceptions may include sensitive request context; keep only safe fields.
            result["errors"].append(f"{type(exc).__name__}; status={getattr(exc, 'status_code', None)}")
        result["response_time"] = perf_counter() - started
        results.append(result)
    return results


def render(results: list[dict], v1: list[dict], v2: list[dict], settings: dict) -> str:
    manifest = json.loads((OUTPUT_DIR / "manifest.json").read_text(encoding="utf-8"))
    lines = [
        "RAG answer-generation evaluation (v3)",
        f"Run time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "Evaluation sample: data/derived/evaluation_coverage/documents.parquet",
        f"Evaluation sample SHA-256: {manifest['evaluation_parquet_sha256']}",
        "Index: existing isolated PostgreSQL enterprise_rag_coverage_eval, 212 unique documents; no ingestion performed",
        f"V1 result SHA-256: {file_hash(ROOT / 'rag_test_results.txt')}",
        f"V2 result SHA-256: {file_hash(V2)}",
        f"Embedding: {settings['model_id']} @ {settings['revision']}",
        f"Answer provider/model: {settings['llm_provider']} / {settings['openai_model']}",
        "Output-token limit: 1024; retrieval limit: 5; similarity = 1 - pgvector cosine distance.",
        "Prompt construction, retrieval, and evaluation corpus are unchanged from V2.",
        "Each question was submitted independently with no conversation history.",
        "Overall response time includes query embedding, retrieval, and generation; model initialization is excluded.",
        "API response time measures only the Responses API call.", "",
    ]
    for number, (result, old1, old2) in enumerate(zip(results, v1, v2), 1):
        metadata = asdict(result["generation_metadata"]) if result["generation_metadata"] else {}
        lines += ["=" * 80, f"QUESTION {number}", "=" * 80,
                  f"Question: {result['question']}", "", "Generated answer:", result["answer"] or "<none>", "",
                  "Source citations in generated answer: " + (", ".join(result["citations"]) or "None"),
                  f"Overall response time: {result['response_time']:.3f} seconds",
                  "Stage times (seconds): " + ", ".join(f"{key}={value:.3f}" for key, value in result["stage_time"].items()),
                  "Generation metadata:"]
        for label, key in (
            ("Response ID", "response_id"), ("Response status", "status"),
            ("Incomplete reason", "incomplete_reason"), ("Input tokens", "input_tokens"),
            ("Cached input tokens", "cached_input_tokens"), ("Output tokens", "output_tokens"),
            ("Reasoning tokens", "reasoning_tokens"), ("Total tokens", "total_tokens"),
            ("API response time (seconds)", "response_time_seconds"),
        ):
            value = metadata.get(key)
            lines.append(f"  {label}: {value:.3f}" if key == "response_time_seconds" and value is not None
                         else f"  {label}: {value if value is not None else 'N/A'}")
        lines += ["Errors: " + ("; ".join(result["errors"]) or "None"),
                  f"Retrieved documents ({len(result['hits'])}):"]
        for rank, hit in enumerate(result["hits"], 1):
            lines.append(f"  Rank {rank}: {hit.doc_id} | similarity={1-hit.distance:.6f} | title={hit.title}")
        lines += ["", "Comparison against V1 and V2:"]
        for label, old in (("V1", old1), ("V2", old2)):
            first = old["hits"][0] if old["hits"] else None
            lines += [f"  {label} top hit: " + (f"{first[1]} (similarity {first[2]:.6f})" if first else "None"),
                      f"  {label} response time: {old['response_time']}",
                      f"  {label} errors: {old['errors'] or 'None'}",
                      f"  {label} citations: {old['citations'] or 'None'}",
                      f"  {label} answer: {old['answer'].replace(chr(10), ' ')}"]
        first = result["hits"][0] if result["hits"] else None
        lines += ["  V3 top hit: " + (f"{first.doc_id} (similarity {1-first.distance:.6f})" if first else "None"),
                  f"  V3 response time: {result['response_time']:.3f} seconds",
                  "  V3 errors: " + ("; ".join(result["errors"]) or "None"),
                  "  V3 citations: " + (", ".join(result["citations"]) or "None"),
                  "  Target ranks (V1 / V2 / V3):"]
        targets = TARGET_IDS[:1] if number in (1, 2) else TARGET_IDS[1:]
        for doc_id in targets:
            old2_rank = next((rank for rank, key, _ in old2["hits"] if key == doc_id), None)
            new_rank = next((rank for rank, hit in enumerate(result["hits"], 1) if hit.doc_id == doc_id), None)
            lines.append(f"    {doc_id}: absent / {old2_rank or 'outside top 5'} / {new_rank or 'outside top 5'}")
        lines.append("")
    lines.append("Interpretation: V3 changes the answer output limit and completion/citation handling only. If the API marks a response incomplete, its partial answer and termination reason remain visible above as an error.")
    return "\n".join(lines) + "\n"


def main() -> None:
    v1 = read_baseline()
    v2 = read_v2()
    settings = local_settings()
    if settings["llm_provider"] != "openai":
        raise ValueError("V3 evaluation requires LLM_PROVIDER=openai")
    embedding = SentenceTransformerEmbeddings(settings["model_id"], settings["revision"])
    search = verify_existing_index(settings, embedding)
    try:
        llm = configured_answer_model(settings)
        if not isinstance(llm, OpenAIAnswerModel) or llm.max_tokens != 1024:
            raise ValueError("V3 evaluation requires OpenAI with a 1024-token output limit")
        results = evaluate(search, embedding, llm)
    finally:
        search.close()
    RESULTS.write_text(render(results, v1, v2, settings), encoding="utf-8")
    print(f"Wrote {RESULTS}; {sum(bool(r['errors']) for r in results)} question(s) had errors")


if __name__ == "__main__":
    main()
