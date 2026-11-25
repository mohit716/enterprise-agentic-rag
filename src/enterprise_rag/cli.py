"""Reproducible commands for the benchmark audit and local development slice."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .dataset import audit, create_sample, corpus_batches


DEFAULT_DOCUMENTS = Path("data/raw/data/documents/test.parquet")
DEFAULT_QUESTIONS = Path("data/raw/data/questions/test.parquet")
DEFAULT_AUDIT = Path("reports/dataset-audit.json")
DEFAULT_SAMPLE = Path("data/derived/sample")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit_command = sub.add_parser("audit", help="Scan the pinned corpus and question Parquet files")
    audit_command.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS)
    audit_command.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    audit_command.add_argument("--output", type=Path, default=DEFAULT_AUDIT)
    sample_command = sub.add_parser("sample", help="Build a safe, source and length stratified development sample")
    sample_command.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS)
    sample_command.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    sample_command.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    sample_command.add_argument("--output", type=Path, default=DEFAULT_SAMPLE)
    sample_command.add_argument("--seed", type=int, default=2026)
    sample_command.add_argument("--per-source", type=int, default=21)
    sample_command.add_argument("--dev-questions", type=int, default=10)
    sub.add_parser("embedding-smoke", help="Embed two real strings with the pinned local model")
    sub.add_parser("infra-smoke", help="Round-trip Azurite and verify PostgreSQL + pgvector")
    sub.add_parser("llm-smoke", help="Make one bounded OpenAI request using synthetic evidence")
    rag_command = sub.add_parser("rag-smoke", help="Index the bounded sample and run a cited question")
    rag_command.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE / "documents.parquet")
    rag_command.add_argument("--question", required=True)
    rag_command.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    if args.command == "audit":
        result = audit(args.documents, args.questions)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"scan_complete": result["scan_complete"],
                          "document_rows": result["documents"]["rows_scanned"],
                          "question_rows": result["questions"]["rows_scanned"]}))
    elif args.command == "sample":
        report = json.loads(args.audit.read_text(encoding="utf-8"))
        if not report["scan_complete"]:
            raise ValueError("Cannot sample from an incomplete audit")
        result = create_sample(args.documents, args.questions, args.output,
                               per_source=args.per_source, dev_questions=args.dev_questions,
                               seed=args.seed,
                               length_limits=report["documents"]["content_length_chars"])
        print(json.dumps({"documents": len(result["sample_doc_ids"]),
                          "dev_questions": len(result["dev_question_ids"]),
                          "held_out_questions": result["held_out_questions"]}))
    elif args.command == "embedding-smoke":
        from .providers import SentenceTransformerEmbeddings
        from dotenv import load_dotenv
        load_dotenv()
        model = SentenceTransformerEmbeddings(
            os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            os.environ.get("EMBEDDING_REVISION", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"))
        vectors = model.encode(["A support document", "A query about support"])
        print(json.dumps({"model": model.model_id, "revision": model.revision,
                          "dimension": model.dimension, "vectors": len(vectors)}))
    elif args.command == "infra-smoke":
        from .providers import AzuriteBlobStorage, Document, connect_local_postgres, local_settings
        settings = local_settings()
        blob = AzuriteBlobStorage(settings["azurite"])
        document = Document("infra-smoke", "test", "Health check", "Local blob round trip")
        blob.put(document)
        if blob.get(document.doc_id) != document or blob.get("missing-infra-smoke") is not None:
            raise AssertionError("Azurite round trip or missing-record behavior failed")
        db = connect_local_postgres(settings["pg_dsn"])
        try:
            cursor = db.cursor()
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            version = cursor.fetchone()
            if not version:
                raise AssertionError("pgvector extension is unavailable")
            db.commit()
        finally:
            db.close()
        print("Azurite and pgvector round trips passed")
    elif args.command == "llm-smoke":
        from .providers import SearchHit, configured_answer_model, local_settings
        settings = local_settings()
        if settings["llm_provider"] != "openai":
            raise ValueError("llm-smoke requires LLM_PROVIDER=openai")
        model = configured_answer_model(settings)
        hit = SearchHit("smoke-doc", "smoke-doc#0", "test", "Synthetic fixture",
                        "The synthetic project code name is Alder.", 0.0)
        try:
            answer = model.answer("What is the synthetic project code name?", [hit])
        except Exception as exc:
            # Do not print exception bodies: API failures can carry sensitive context.
            status = getattr(exc, "status_code", None)
            raise SystemExit(f"OpenAI smoke failed: {type(exc).__name__}, status={status}") from None
        if "Alder" not in answer or "[smoke-doc]" not in answer:
            raise SystemExit("OpenAI smoke response lacked the expected fact or citation")
        print(json.dumps({"provider": "openai", "model": model.model,
                          "answer": answer, "citation_valid": True}))
    elif args.command == "rag-smoke":
        from .providers import (AzuriteBlobStorage, Document, PgvectorSearch,
                                SentenceTransformerEmbeddings, answer_citations,
                                configured_answer_model,
                                local_settings)
        settings = local_settings()
        embedding = SentenceTransformerEmbeddings(settings["model_id"], settings["revision"])
        blob = AzuriteBlobStorage(settings["azurite"])
        search = PgvectorSearch(settings["pg_dsn"], embedding.dimension)
        try:
            documents = []
            for batch in corpus_batches(args.sample, batch_size=32):
                documents.extend(Document(row["doc_id"], row["source_type"],
                                          row["title"] or "", row["content"]) for row in batch)
            for start in range(0, len(documents), 8):
                group = documents[start:start + 8]
                vectors = embedding.encode([doc.content for doc in group])
                for doc in group:
                    blob.put(doc)
                search.upsert(group, vectors)
            hits = search.search(embedding.encode([args.question])[0], limit=args.limit)
            if not hits:
                raise AssertionError("No documents retrieved")
            llm = configured_answer_model(settings)
            answer = llm.answer(args.question, hits)
            cited_ids = answer_citations(answer, hits)
            print(json.dumps({"question": args.question, "answer": answer,
                              "citations": cited_ids,
                              "retrieved_doc_ids": [hit.doc_id for hit in hits],
                              "embedding_model": embedding.model_id,
                              "embedding_revision": embedding.revision,
                              "embedding_dimension": embedding.dimension,
                              "llm_provider": settings["llm_provider"],
                              "llm_model": llm.model}, indent=2))
        finally:
            search.close()


if __name__ == "__main__":
    main()
