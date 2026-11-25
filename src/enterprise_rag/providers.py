"""Narrow local provider contracts and implementations for one document-level slice."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import re
from time import perf_counter
from typing import Protocol, Sequence
from urllib.parse import unquote, urlsplit


@dataclass(frozen=True)
class Document:
    doc_id: str
    source_type: str
    title: str
    content: str

    def __post_init__(self):
        if not self.doc_id or not self.source_type or not self.content:
            raise ValueError("Document needs a doc_id, source_type, and nonempty content")


@dataclass(frozen=True)
class SearchHit:
    doc_id: str
    chunk_id: str
    source_type: str
    title: str
    content: str
    distance: float


@dataclass(frozen=True)
class GenerationMetadata:
    response_id: str | None
    status: str | None
    incomplete_reason: str | None
    input_tokens: int | None
    cached_input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    total_tokens: int | None
    response_time_seconds: float


class AnswerGenerationStatusError(RuntimeError):
    """A non-completed API response, with its partial text and diagnostics."""

    def __init__(self, metadata: GenerationMetadata, partial_answer: str):
        self.metadata = metadata
        self.partial_answer = partial_answer
        super().__init__(f"OpenAI response status={metadata.status}; "
                         f"incomplete_reason={metadata.incomplete_reason}")


class Embeddings(Protocol):
    dimension: int
    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class BlobStorage(Protocol):
    def put(self, document: Document) -> None: ...
    def get(self, doc_id: str) -> Document | None: ...


class VectorSearch(Protocol):
    def upsert(self, documents: Sequence[Document], vectors: Sequence[Sequence[float]]) -> None: ...
    def search(self, vector: Sequence[float], *, limit: int = 5,
               source_type: str | None = None) -> list[SearchHit]: ...


class AnswerModel(Protocol):
    def answer(self, question: str, hits: Sequence[SearchHit]) -> str: ...


def validate_vectors(documents: Sequence[Document], vectors: Sequence[Sequence[float]], dimension: int) -> None:
    if len(documents) != len(vectors):
        raise ValueError("Document and vector counts differ")
    if any(len(vector) != dimension for vector in vectors):
        raise ValueError(f"Expected {dimension} dimensions for every vector")
    if any(not math.isfinite(float(value)) for vector in vectors for value in vector):
        raise ValueError("Vectors must contain only finite numbers")


class SentenceTransformerEmbeddings:
    def __init__(self, model_id: str, revision: str, batch_size: int = 16):
        from sentence_transformers import SentenceTransformer
        if not revision or batch_size < 1:
            raise ValueError("Pin a model revision and use a positive batch size")
        self.model_id, self.revision, self.batch_size = model_id, revision, batch_size
        self._model = SentenceTransformer(model_id, revision=revision)
        self.dimension = self._model.get_embedding_dimension()
        if not self.dimension:
            raise ValueError("Embedding model did not expose its vector dimension")

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(list(texts), batch_size=self.batch_size,
                                     normalize_embeddings=True, show_progress_bar=False)
        result = vectors.tolist()
        if any(len(vector) != self.dimension for vector in result):
            raise ValueError("Model returned unexpected vector dimension")
        return result


class AzuriteBlobStorage:
    """Azurite-only blob adapter; the local default never targets Azure."""

    def __init__(self, connection_string: str, container: str = "documents"):
        from azure.storage.blob import BlobServiceClient
        if connection_string != "UseDevelopmentStorage=true" and "127.0.0.1:10000" not in connection_string and "localhost:10000" not in connection_string:
            raise ValueError("Local storage requires an Azurite endpoint")
        # Azurite 3.36 supports this service version; newer SDK defaults can outpace it.
        self._client = BlobServiceClient.from_connection_string(
            connection_string, api_version="2025-11-05"
        )
        self._container = self._client.get_container_client(container)
        self._container.create_container() if not self._container.exists() else None

    def put(self, document: Document) -> None:
        import json
        self._container.upload_blob(document.doc_id, json.dumps(document.__dict__), overwrite=True)

    def get(self, doc_id: str) -> Document | None:
        import json
        from azure.core.exceptions import ResourceNotFoundError
        try:
            data = self._container.download_blob(doc_id).readall()
        except ResourceNotFoundError:
            return None
        return Document(**json.loads(data))


class PgvectorSearch:
    def __init__(self, dsn: str, dimension: int):
        if dimension < 1:
            raise ValueError("Vector dimension must be positive")
        self.dimension = dimension
        self._db = connect_local_postgres(dsn)
        try:
            cursor = self._db.cursor()
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute(f"""CREATE TABLE IF NOT EXISTS document_vectors (
                doc_id TEXT PRIMARY KEY,
                chunk_id TEXT NOT NULL,
                source_type TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding VECTOR({dimension}) NOT NULL
            )""")
            cursor.execute("""SELECT atttypmod FROM pg_attribute
                WHERE attrelid = 'document_vectors'::regclass AND attname = 'embedding'""")
            actual = cursor.fetchone()[0]
            if actual != dimension:
                raise ValueError(f"Index dimension {actual} differs from model dimension {dimension}")
            self._db.commit()
        except Exception:
            self._db.rollback()
            self._db.close()
            raise

    def close(self) -> None:
        self._db.close()

    def upsert(self, documents: Sequence[Document], vectors: Sequence[Sequence[float]]) -> None:
        validate_vectors(documents, vectors, self.dimension)
        cursor = self._db.cursor()
        try:
            for doc, vector in zip(documents, vectors):
                cursor.execute("""INSERT INTO document_vectors
                    (doc_id, chunk_id, source_type, title, content, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s::vector)
                    ON CONFLICT (doc_id) DO UPDATE SET source_type=EXCLUDED.source_type,
                    title=EXCLUDED.title, content=EXCLUDED.content, embedding=EXCLUDED.embedding""",
                    (doc.doc_id, f"{doc.doc_id}#0", doc.source_type, doc.title,
                     doc.content, _vector_text(vector)))
            self._db.commit()
        except Exception:
            self._db.rollback()
            raise

    def search(self, vector: Sequence[float], *, limit: int = 5,
               source_type: str | None = None) -> list[SearchHit]:
        if len(vector) != self.dimension or limit < 1 or limit > 100:
            raise ValueError("Invalid query vector dimension or limit")
        cursor = self._db.cursor()
        cursor.execute("""SELECT doc_id, chunk_id, source_type, title, content,
            embedding <=> %s::vector AS distance FROM document_vectors
            WHERE (%s::text IS NULL OR source_type = %s::text)
            ORDER BY embedding <=> %s::vector LIMIT %s""",
            (_vector_text(vector), source_type, source_type, _vector_text(vector), limit))
        rows = cursor.fetchall()
        return [SearchHit(*row[:5], float(row[5])) for row in rows]


def _vector_text(vector: Sequence[float]) -> str:
    return "[" + ",".join(str(float(value)) for value in vector) + "]"


def connect_local_postgres(dsn: str):
    """Open a loopback PostgreSQL connection with a pure Python driver."""
    import pg8000.dbapi
    parsed = urlsplit(dsn)
    if parsed.scheme not in {"postgresql", "postgres"} or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("Local mode requires a loopback PostgreSQL URL")
    return pg8000.dbapi.connect(user=unquote(parsed.username or ""),
                                password=unquote(parsed.password or ""),
                                host=parsed.hostname, port=parsed.port or 5432,
                                database=parsed.path.lstrip("/"), timeout=5)


class OllamaAnswerModel:
    def __init__(self, url: str, model: str, timeout: int = 60, max_tokens: int = 256):
        if not url.startswith(("http://127.0.0.1:", "http://localhost:")):
            raise ValueError("Local mode requires a loopback Ollama URL")
        if timeout < 1 or max_tokens < 1:
            raise ValueError("Timeout and token limit must be positive")
        self.url, self.model, self.timeout, self.max_tokens = url.rstrip("/"), model, timeout, max_tokens

    def answer(self, question: str, hits: Sequence[SearchHit]) -> str:
        import requests
        context = "\n\n".join(f"[{hit.doc_id}] {hit.title}\n{hit.content[:1800]}" for hit in hits)
        prompt = ("Answer using only the supplied documents. Cite each claim with [doc_id]. "
                  "If the documents do not support an answer, say you do not know.\n\n"
                  f"Documents:\n{context}\n\nQuestion: {question}")
        response = requests.post(f"{self.url}/api/generate", json={
            "model": self.model, "prompt": prompt, "stream": False,
            "options": {"num_predict": self.max_tokens}}, timeout=self.timeout)
        response.raise_for_status()
        answer = response.json().get("response")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Ollama returned an empty answer")
        return answer.strip()


class OpenAIAnswerModel:
    """OpenAI Responses API adapter with bounded output and no stored response."""

    def __init__(self, model: str, timeout: int = 60, max_tokens: int = 2048,
                 max_chars_per_hit: int = 12000, client=None):
        if not model or timeout < 1 or max_tokens < 1 or max_chars_per_hit < 1:
            raise ValueError("Model, timeout, output limit, and context limit are required")
        if client is None:
            from openai import OpenAI
            client = OpenAI(timeout=timeout, max_retries=0)
        self.model, self.timeout, self.max_tokens = model, timeout, max_tokens
        self.max_chars_per_hit = max_chars_per_hit
        self._client = client
        self.last_generation: GenerationMetadata | None = None

    def answer(self, question: str, hits: Sequence[SearchHit]) -> str:
        context = "\n\n".join(
            f"[{hit.doc_id}] {hit.title}\n{hit.content[:self.max_chars_per_hit]}"
            for hit in hits
        )
        self.last_generation = None
        started = perf_counter()
        response = self._client.responses.create(
            model=self.model,
            instructions=("Answer using only the supplied documents. Cite every supported claim "
                          "with [doc_id]. If the documents do not support an answer, say you do not know."),
            input=f"Documents:\n{context}\n\nQuestion: {question}",
            reasoning={"effort": "none"},
            max_output_tokens=self.max_tokens,
            store=False,
        )
        usage = response.usage
        input_details = getattr(usage, "input_tokens_details", None)
        output_details = getattr(usage, "output_tokens_details", None)
        incomplete_details = response.incomplete_details
        self.last_generation = GenerationMetadata(
            response_id=response.id,
            status=response.status,
            incomplete_reason=getattr(incomplete_details, "reason", None),
            input_tokens=getattr(usage, "input_tokens", None),
            cached_input_tokens=getattr(input_details, "cached_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            reasoning_tokens=getattr(output_details, "reasoning_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            response_time_seconds=perf_counter() - started,
        )
        answer = response.output_text
        if response.status != "completed":
            raise AnswerGenerationStatusError(
                self.last_generation, answer if isinstance(answer, str) else ""
            )
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("OpenAI returned an empty answer")
        return answer.strip()


def answer_citations(answer: str, hits: Sequence[SearchHit]) -> list[str]:
    """Return cited original IDs and reject citations outside retrieved documents."""
    cited = set()
    for reference in re.findall(r"\[[^\[\]]*\]|\[|\]", answer):
        if not (reference.startswith("[") and reference.endswith("]")
                and re.fullmatch(r"[A-Za-z0-9_-]+", reference[1:-1])):
            raise ValueError(f"Malformed or unclosed citation reference: {reference!r}")
        cited.add(reference[1:-1])
    cited = sorted(cited)
    unknown = set(cited) - {hit.doc_id for hit in hits}
    if unknown:
        raise ValueError(f"Answer cited documents absent from retrieval: {sorted(unknown)}")
    return cited


def configured_answer_model(settings: dict[str, str]) -> AnswerModel:
    """Select exactly one answer backend; never fall back to a paid service."""
    timeout = int(os.environ.get("LLM_TIMEOUT_SECONDS", "60"))
    max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "2048"))
    if settings["llm_provider"] == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        return OpenAIAnswerModel(
            settings["openai_model"], timeout, max_tokens,
            int(os.environ.get("OPENAI_CONTEXT_CHARS_PER_HIT", "12000")),
        )
    if settings["llm_provider"] == "ollama":
        return OllamaAnswerModel(settings["ollama_url"], settings["ollama_model"],
                                 timeout, max_tokens)
    raise ValueError(f"Unsupported LLM_PROVIDER: {settings['llm_provider']}")


def local_settings() -> dict[str, str]:
    """Keep storage and search local while selecting the answer backend explicitly."""
    from dotenv import load_dotenv
    load_dotenv()
    if os.environ.get("PROVIDER_MODE", "local") != "local":
        raise ValueError("Only local providers are implemented; Azure requires an explicit adapter")
    llm_provider = os.environ.get("LLM_PROVIDER", "openai")
    if llm_provider not in {"openai", "ollama"}:
        raise ValueError(f"Unsupported LLM_PROVIDER: {llm_provider}")
    pg_dsn = os.environ["PG_DSN"]
    if urlsplit(pg_dsn).hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("Local mode requires a loopback PostgreSQL host")
    return {
        "pg_dsn": pg_dsn,
        "azurite": os.environ.get("AZURITE_CONNECTION_STRING", "UseDevelopmentStorage=true"),
        "model_id": os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        "revision": os.environ.get("EMBEDDING_REVISION", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"),
        "llm_provider": llm_provider,
        "openai_model": os.environ.get("OPENAI_MODEL", "gpt-6-luna"),
        "ollama_url": os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"),
        "ollama_model": os.environ.get("OLLAMA_MODEL", "llama3.2:3b"),
    }
