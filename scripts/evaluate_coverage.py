"""Repeat the three-question coverage experiment without changing the baseline index.

Run from the project root: python scripts/evaluate_coverage.py
The evaluation corpus is the immutable development sample plus three raw IDs.
Only retrieval fields enter the runtime corpus; benchmark answer labels are not read.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from time import perf_counter
from urllib.parse import urlsplit, urlunsplit

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from enterprise_rag.dataset import corpus_batches
from enterprise_rag.providers import (
    AzuriteBlobStorage, Document, PgvectorSearch, SentenceTransformerEmbeddings,
    answer_citations, configured_answer_model, connect_local_postgres, local_settings,
)


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data/derived/sample/documents.parquet"
RAW = ROOT / "data/raw/data/documents/test.parquet"
OUTPUT_DIR = ROOT / "data/derived/evaluation_coverage"
RESULTS = ROOT / "rag_test_results_v2.txt"
BASELINE = ROOT / "rag_test_results.txt"
TARGET_IDS = (
    "dsid_e54ef48bae78474684a957cf613d47d5",
    "dsid_229dd48e9b1d466a81ebaffe3ec84469",
    "dsid_aeb0022d62bc43beb6549ba92e5655eb",
)
QUESTIONS = (
    "What is the purpose of the perf-canary service, and what performance metrics does it collect?",
    "If the perf-canary service needs to be immediately disabled in production, what steps should the on-call engineer take, and which dashboards should they monitor?",
    "We are onboarding an enterprise customer who requires cross-account GPU bursting and third-party vendor integration. What technical, security, billing, and operational requirements must be addressed before production deployment?",
)
DATABASE = "enterprise_rag_coverage_eval"
CONTAINER = "documents-eval"


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def database_dsn(dsn: str, database: str) -> str:
    parts = urlsplit(dsn)
    return urlunsplit((parts.scheme, parts.netloc, "/" + database, parts.query, parts.fragment))


def ensure_database(dsn: str) -> str:
    admin = connect_local_postgres(database_dsn(dsn, "postgres"))
    try:
        admin.autocommit = True
        cursor = admin.cursor()
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DATABASE,))
        if cursor.fetchone() is None:
            cursor.execute(f"CREATE DATABASE {DATABASE}")
    finally:
        admin.close()
    return database_dsn(dsn, DATABASE)


def build_corpus() -> tuple[list[Document], dict]:
    baseline = {}
    for batch in corpus_batches(SAMPLE):
        for row in batch:
            if row["doc_id"] in baseline:
                raise ValueError(f"Duplicate development ID: {row['doc_id']}")
            baseline[row["doc_id"]] = row
    if len(baseline) != 209:
        raise ValueError(f"Expected 209 development documents, found {len(baseline)}")
    if set(TARGET_IDS) & baseline.keys():
        raise ValueError("A target already occurs in the development sample")

    targets = {}
    for batch in corpus_batches(RAW):
        for row in batch:
            doc_id = row["doc_id"]
            if doc_id in TARGET_IDS:
                if doc_id in targets:
                    raise ValueError(f"Duplicate target ID in raw corpus: {doc_id}")
                targets[doc_id] = row
    if set(targets) != set(TARGET_IDS):
        raise ValueError(f"Missing raw target IDs: {sorted(set(TARGET_IDS) - targets.keys())}")
    rows = {**baseline, **targets}
    if len(rows) != 212:
        raise ValueError("Evaluation corpus must have 212 distinct documents")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    corpus_path = OUTPUT_DIR / "documents.parquet"
    ordered = [{field: rows[key][field] for field in ("doc_id", "source_type", "title", "content")}
               for key in sorted(rows)]
    if corpus_path.exists():
        existing = [row for batch in corpus_batches(corpus_path) for row in batch]
        if sorted(existing, key=lambda row: row["doc_id"]) != ordered:
            raise ValueError("Existing evaluation corpus differs; refusing to overwrite it")
    else:
        pq.write_table(pa.Table.from_pylist(ordered), corpus_path)

    manifest = {
        "purpose": "Controlled coverage test: original development sample plus three specified raw documents",
        "development_sample_path": str(SAMPLE.relative_to(ROOT)),
        "development_sample_sha256": file_hash(SAMPLE),
        "raw_corpus_path": str(RAW.relative_to(ROOT)),
        "raw_corpus_sha256": file_hash(RAW),
        "target_doc_ids": list(TARGET_IDS),
        "development_count": len(baseline),
        "evaluation_count": len(rows),
        "evaluation_doc_ids": sorted(rows),
        "source_counts": dict(sorted(Counter(row["source_type"] for row in ordered).items())),
        "evaluation_parquet_sha256": file_hash(corpus_path),
        "storage_container": CONTAINER,
        "postgres_database": DATABASE,
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Existing evaluation manifest differs; refusing to overwrite it")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    docs = [Document(row["doc_id"], row["source_type"], row["title"] or "", row["content"])
            for row in ordered]
    return docs, manifest


def read_baseline() -> list[dict]:
    text = BASELINE.read_text(encoding="utf-8")
    sections = re.split(r"(?m)^QUESTION ([123])\s*$", text)[1:]
    found = {}
    for number, section in zip(sections[::2], sections[1::2]):
        def field(name: str) -> str:
            match = re.search(rf"(?m)^{re.escape(name)}: (.*)$", section)
            return match.group(1).strip() if match else ""
        answer_match = re.search(r"Generated answer:\s*\n(.*?)\n\s*Source citations in generated answer:", section, re.S)
        hits = re.findall(
            r"(?ms)^Rank: (\d+)\s*\nDocument ID: ([^\r\n]+).*?^Relevance score \(1 - distance\): ([0-9.]+)",
            section,
        )
        found[int(number)] = {
            "question": field("Question"), "answer": answer_match.group(1).strip() if answer_match else "",
            "citations": field("Source citations in generated answer"),
            "response_time": field("Response time"), "errors": field("Errors"),
            "hits": [(int(rank), doc_id, float(score)) for rank, doc_id, score in hits],
        }
    if set(found) != {1, 2, 3} or any(found[i]["question"] != QUESTIONS[i - 1] for i in found):
        raise ValueError("Baseline does not contain the exact three evaluation questions")
    return [found[i] for i in (1, 2, 3)]


def index_and_verify(docs: list[Document], settings: dict, embedding: SentenceTransformerEmbeddings):
    eval_dsn = ensure_database(settings["pg_dsn"])
    blob = AzuriteBlobStorage(settings["azurite"], container=CONTAINER)
    search = PgvectorSearch(eval_dsn, embedding.dimension)
    try:
        db = connect_local_postgres(eval_dsn)
        try:
            cursor = db.cursor()
            cursor.execute("SELECT doc_id, chunk_id, source_type, title, content FROM document_vectors")
            indexed = {row[0]: row for row in cursor.fetchall()}
        finally:
            db.close()
        expected = {doc.doc_id: doc for doc in docs}
        if set(indexed) - expected.keys():
            raise ValueError("Evaluation database contains documents outside the controlled sample")
        for doc_id, row in indexed.items():
            doc = expected[doc_id]
            if tuple(row) != (doc_id, f"{doc_id}#0", doc.source_type, doc.title, doc.content):
                raise ValueError(f"Existing indexed document differs from evaluation corpus: {doc_id}")

        missing = [doc for doc in docs if doc.doc_id not in indexed]
        for start in range(0, len(missing), 8):
            group = missing[start:start + 8]
            vectors = embedding.encode([doc.content for doc in group])
            for doc in group:
                blob.put(doc)
            search.upsert(group, vectors)

        for doc in docs:
            stored = blob.get(doc.doc_id)
            if stored is None:
                blob.put(doc)
                stored = blob.get(doc.doc_id)
            if stored != doc:
                raise ValueError(f"Azurite document mismatch: {doc.doc_id}")
        db = connect_local_postgres(eval_dsn)
        try:
            cursor = db.cursor()
            cursor.execute("SELECT doc_id, chunk_id, source_type, title, content FROM document_vectors")
            rows = cursor.fetchall()
        finally:
            db.close()
        if len(rows) != 212 or {row[0] for row in rows} != expected.keys():
            raise ValueError("Evaluation index must contain exactly 212 unique documents")
        for doc_id in TARGET_IDS:
            doc = expected[doc_id]
            match = [row for row in rows if row[0] == doc_id]
            if len(match) != 1 or tuple(match[0]) != (doc_id, f"{doc_id}#0", doc.source_type, doc.title, doc.content):
                raise ValueError(f"Target verification failed in PostgreSQL: {doc_id}")
        return search, len(missing)
    except Exception:
        search.close()
        raise


def evaluate(search, embedding, settings: dict) -> list[dict]:
    llm = configured_answer_model(settings)
    results = []
    for question in QUESTIONS:
        started = perf_counter()
        result = {"question": question, "answer": "", "citations": [], "hits": [], "errors": []}
        stage = {}
        try:
            tick = perf_counter()
            query_vector = embedding.encode([question])[0]
            stage["embedding"] = perf_counter() - tick
            tick = perf_counter()
            hits = search.search(query_vector, limit=5)
            stage["retrieval"] = perf_counter() - tick
            result["hits"] = hits
            if not hits:
                raise ValueError("No documents retrieved")
            tick = perf_counter()
            try:
                result["answer"] = llm.answer(question, hits)
            finally:
                stage["generation"] = perf_counter() - tick
            result["citations"] = sorted(set(re.findall(r"\[([A-Za-z0-9_-]+)\]", result["answer"])))
            try:
                answer_citations(result["answer"], hits)
            except ValueError as exc:
                result["errors"].append(str(exc))
        except Exception as exc:
            # API exceptions can contain request details; record only safe diagnostics.
            result["errors"].append(f"{type(exc).__name__}; status={getattr(exc, 'status_code', None)}")
        result["response_time"] = perf_counter() - started
        result["stage_time"] = stage
        results.append(result)
    return results


def render(results: list[dict], baseline: list[dict], manifest: dict, settings: dict, new_count: int) -> str:
    lines = [
        "Controlled RAG coverage evaluation (v2)",
        f"Run time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "Evaluation sample: data/derived/evaluation_coverage/documents.parquet",
        f"Development documents preserved: {manifest['development_count']}",
        f"Added raw documents: {', '.join(TARGET_IDS)}",
        f"Evaluation documents: {manifest['evaluation_count']} unique IDs; newly indexed this run: {new_count}",
        f"Development sample SHA-256: {manifest['development_sample_sha256']}",
        f"Evaluation sample SHA-256: {manifest['evaluation_parquet_sha256']}",
        f"Baseline result SHA-256: {file_hash(BASELINE)}",
        f"Index: local PostgreSQL database {DATABASE}, exactly 212 unique rows",
        f"Blob storage: local Azurite container {CONTAINER}; all 212 documents verified",
        f"Embedding: {embedding_name(settings)} @ {settings['revision']}; content only, one vector per document",
        f"Answer provider/model: {settings['llm_provider']} / {settings['openai_model']}",
        "Retrieval limit: 5. Similarity score = 1 - pgvector cosine distance.",
        "Each question was submitted independently with no conversation history.",
        "Response time includes query embedding, retrieval, and generation; corpus construction, indexing, and model initialization are excluded.",
        "Baseline retrieval used the original shared index (209 development documents plus one fixture); v2 uses an isolated 212-document index.",
        "",
    ]
    for number, (result, prior) in enumerate(zip(results, baseline), 1):
        lines += ["=" * 80, f"QUESTION {number}", "=" * 80,
                  f"Question: {result['question']}", "", "Generated answer:", result["answer"] or "<none>", "",
                  "Source citations in generated answer: " + (", ".join(result["citations"]) or "None"),
                  f"Response time: {result['response_time']:.3f} seconds",
                  "Stage times (seconds): " + ", ".join(f"{key}={value:.3f}" for key, value in result["stage_time"].items()),
                  "Errors: " + ("; ".join(result["errors"]) or "None"),
                  f"Retrieved documents ({len(result['hits'])}):"]
        for rank, hit in enumerate(result["hits"], 1):
            lines += [f"  Rank {rank}: {hit.doc_id} | chunk={hit.chunk_id} | similarity={1-hit.distance:.6f} | distance={hit.distance:.6f} | title={hit.title}"]
        baseline_hit = prior["hits"][0] if prior["hits"] else None
        v2_hit = result["hits"][0] if result["hits"] else None
        lines += ["", "Comparison with rag_test_results.txt:",
                  "  Baseline top hit: " + (f"{baseline_hit[1]} (similarity {baseline_hit[2]:.6f})" if baseline_hit else "None"),
                  "  V2 top hit: " + (f"{v2_hit.doc_id} (similarity {1-v2_hit.distance:.6f})" if v2_hit else "None"),
                  f"  Baseline citations: {prior['citations'] or 'None'}",
                  f"  V2 citations: {', '.join(result['citations']) or 'None'}",
                  f"  Baseline response time: {prior['response_time']}",
                  f"  V2 response time: {result['response_time']:.3f} seconds",
                  f"  Baseline errors: {prior['errors'] or 'None'}",
                  f"  V2 errors: {'; '.join(result['errors']) or 'None'}",
                  "  Baseline answer: " + prior["answer"].replace("\n", " "),
                  "  V2 answer: " + result["answer"].replace("\n", " "),
                  "  Target document ranks (baseline -> v2):"]
        applicable = TARGET_IDS[:1] if number in (1, 2) else TARGET_IDS[1:]
        for doc_id in applicable:
            hit_rank = next(((rank, hit) for rank, hit in enumerate(result["hits"], 1)
                             if hit.doc_id == doc_id), None)
            lines.append(f"    {doc_id}: absent -> " +
                         (f"rank {hit_rank[0]}, similarity {1-hit_rank[1].distance:.6f}" if hit_rank else "outside top 5"))
        lines.append("")
    lines += ["Interpretation: This run changes corpus coverage only. Answer wording and timing can vary across independent OpenAI calls; the retrieved IDs and scores are the direct coverage comparison."]
    observations = []
    for number, result in enumerate(results, 1):
        if re.search(r"\[[A-Za-z0-9_-]*$", result["answer"].rstrip()):
            observations.append(f"Q{number} ends with an unclosed citation ID")
    missing_topics = [topic for topic in ("security", "billing", "operational")
                      if topic not in results[2]["answer"].lower()]
    if missing_topics:
        observations.append("Q3 does not mention these requested topics: " + ", ".join(missing_topics))
    if observations:
        lines.append("Output review: " + "; ".join(observations) + ". The provider raised no exception. Answer generation was not changed in this evaluation.")
    return "\n".join(lines) + "\n"


def embedding_name(settings: dict) -> str:
    return settings["model_id"]


def main() -> None:
    baseline = read_baseline()
    docs, manifest = build_corpus()
    settings = local_settings()
    embedding = SentenceTransformerEmbeddings(settings["model_id"], settings["revision"])
    search, new_count = index_and_verify(docs, settings, embedding)
    try:
        results = evaluate(search, embedding, settings)
    finally:
        search.close()
    RESULTS.write_text(render(results, baseline, manifest, settings, new_count), encoding="utf-8")
    print(f"Wrote {RESULTS}; indexed {new_count} missing documents; {sum(bool(r['errors']) for r in results)} question(s) had errors")


if __name__ == "__main__":
    main()
