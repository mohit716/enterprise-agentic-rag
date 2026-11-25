# Enterprise Agentic RAG: validated local V4 pipeline

This repository contains a working, local-first retrieval and answer-generation
pipeline evaluated on a bounded sample of EnterpriseRAG-Bench. It is the V4
baseline, with a 2,048-token OpenAI output limit and strict citation checks.
Downloaded Parquet and derived samples stay outside Git. Runtime retrieval reads
only document fields and never loads benchmark answers, answer facts, or
expected IDs. The repository includes regression tests and sanitized evaluation
reports; it does not claim full-corpus or production search quality.

## Architecture and implemented capabilities

`src/enterprise_rag/dataset.py` audits the pinned benchmark snapshot and builds
a deterministic, question-label-free development sample. `cli.py` provides the
audit, sampling, and smoke commands. `providers.py` defines the document,
embedding, Blob storage, vector search, and answer contracts. The local path is:

```text
Pinned EnterpriseRAG-Bench Parquet -> document-only sample
  -> Sentence Transformers (all-MiniLM-L6-v2, pinned revision, 384 dimensions)
  -> Azurite Blob storage + PostgreSQL/pgvector (one vector per document)
  -> cosine-distance top-five retrieval -> OpenAI Responses API
  -> answer with validated retrieved-document citations
```

The OpenAI request uses `store=False`, a bounded context per hit, a timeout,
and an output-token limit. The adapter records response status, token usage,
latency, and incomplete-response details. Ollama is an explicit local alternative
for answers. There is no automatic provider fallback, document chunking,
production Azure Search adapter, or end-to-end held-out benchmark score yet.

## Dataset audit and sample

Install Python 3.12 or newer. Create a virtual environment, then run:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[test]"
.\.venv\Scripts\python -c "from huggingface_hub import snapshot_download; snapshot_download('onyx-dot-app/EnterpriseRAG-Bench', repo_type='dataset', revision='69916e31c68aa5963c00248fd7f0bc12d04fd235', allow_patterns=['data/documents/test.parquet','data/questions/test.parquet'], local_dir='data/raw')"
.\.venv\Scripts\python -m enterprise_rag.cli audit
.\.venv\Scripts\python -m enterprise_rag.cli sample
.\.venv\Scripts\python -m pytest -q
```

The download command fetches about 1.41 GB. `audit` writes a full-scan JSON
report in `reports/dataset-audit.json`. The checked-in summary is
[reports/dataset-audit.md](reports/dataset-audit.md). `sample` writes 209
document rows and ten question-only development prompts under
`data/derived/sample/`, plus a manifest with the seed, hashes, source counts,
and coverage. It selects per-source short, medium, and long documents using
measured p50/p95 content lengths; documents needed by selected development
questions are added. The remaining 490 questions stay held out. This sample is
for integration work, not a replacement for full-corpus retrieval evaluation.
The observed local integration results are in
[reports/local-smoke.md](reports/local-smoke.md).

## Local services and OpenAI answers

Install Docker Engine with Compose. For a fresh setup, copy `.env.example` to
`.env`, replace the PostgreSQL password placeholder in both
`POSTGRES_PASSWORD` and `PG_DSN`, and set `OPENAI_API_KEY`. The host PostgreSQL
port is 55432 by default; change `POSTGRES_PORT` and the port in `PG_DSN`
together if needed. Keep `.env` local;
it is ignored by Git. Azurite and PostgreSQL bind only to loopback and persist
data in Docker volumes. OpenAI is the configured answer provider. Its default
model is `gpt-6-luna`, and each call has a timeout and output token limit.
`OPENAI_CONTEXT_CHARS_PER_HIT` bounds how much of each retrieved document is
sent to the model; its default is 12,000 characters.

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
python -m enterprise_rag.cli llm-smoke
docker compose up -d
docker compose ps
python -m enterprise_rag.cli infra-smoke
python -m enterprise_rag.cli embedding-smoke
python -m enterprise_rag.cli rag-smoke --question "What are the multipart upload limits?"
docker compose down
```

`llm-smoke` makes one small OpenAI API request using a synthetic document and
checks its answer and citation. It does not need Docker or benchmark data.
`infra-smoke` checks an Azurite Blob write/read and missing record, PostgreSQL
connectivity, and the pgvector extension. `embedding-smoke` records the pinned
Sentence Transformers model, revision, output dimensions, and a real two-item
batch. `rag-smoke` indexes the bounded local sample, retrieves documents from
pgvector, and asks OpenAI for a cited answer. The sample's document IDs remain
the citation IDs. This initial slice embeds each entire document as one unit;
the model may truncate long content, so retrieval quality is not a benchmark
result. Chunking belongs to EAR-4. Pass an unanswerable question to inspect
that path manually; the prompt instructs the model to admit missing evidence.

To repeat the controlled coverage evaluation, run `python scripts/evaluate_coverage.py`
from the project root while Compose is healthy and `.env` contains the local
PostgreSQL and OpenAI settings. This writes a separate 212-document corpus and
manifest in `data/derived/evaluation_coverage/` by preserving all 209 original
development documents and adding three specified raw documents. It uses the
local PostgreSQL database `enterprise_rag_coverage_eval` and Azurite container
`documents-eval`, verifies document IDs and contents in both, and writes
`rag_test_results_v2.txt` with three independent questions and a comparison
against the redacted baseline `rag_test_results.txt`. Existing records are checked before indexing,
so rerunning it does not add duplicate vectors. It makes three OpenAI calls.

To rerun the same three questions after changing answer-generation handling,
run `python scripts/evaluate_generation_v3.py`. It verifies the existing
212-document evaluation index without ingesting records and writes
`rag_test_results_v3.txt` with response status, token usage, citations, and
comparisons with V1 and V2. That run used a 1,024-token output limit, and its
Question 3 response was incomplete. The current default is 2,048 tokens. Run
`python scripts/evaluate_generation_v4.py` for the current limit; it writes
`rag_test_results_v4.txt` with API usage and comparisons against V1 through
V3. Incomplete API responses are reported with their termination reason and
partial text. The versioned reports preserve questions, answers, retrieval
ranks, citations, timings, and API status. The V1 baseline omits all 15 full
retrieved document bodies while retaining the fields consumed by the evaluation
scripts. Historical SHA-256 references in the V2–V4 reports refer to the original
V1 file before that redaction.

## Recorded evaluation results

- The full dataset audit scanned 511,962 documents and 500 questions. The
  deterministic development sample has 209 documents and ten question-only
  prompts; 490 questions remain held out.
- A separate 212-document coverage evaluation added three specified supporting
  documents. Its three-question V4 run used the existing isolated index, with
  no new ingestion. All three API responses completed and all three generated
  answers passed citation validation. The third answer included cited technical,
  security, billing, and operational sections.
- V3's third response had been incomplete at the 1,024-token limit. V4 used
  2,048 tokens and completed. These are three targeted checks, not an aggregate
  RAG benchmark score. See [V4 results](rag_test_results_v4.txt) and the
  [local integration report](reports/local-smoke.md).

To use Ollama instead, set `LLM_PROVIDER=ollama` in `.env`, install Ollama, and
run `ollama pull llama3.2:3b`. There is no automatic fallback between models.

To clear local database and Blob data, run `docker compose down -v` after
stopping the services. This deletes both Docker volumes. Raw Parquet and the
sample under `data/` are separate files and are not changed by that command.

For the opt-in live storage and search contract test, set
`$env:RUN_LOCAL_INTEGRATION='1'` and run
`python -m pytest tests/test_local_integration.py -q` while Compose is up.
Remove the environment variable afterward.

## Contracts and boundaries

- `Document`, `SearchHit`, and the storage, embedding, search, and answer
  protocols are in `src/enterprise_rag/providers.py`. Local adapters use
  Azurite, PostgreSQL + pgvector, the pinned `all-MiniLM-L6-v2` revision,
  and OpenAI by default. Provider selection is explicit; there is no Azure or
  Ollama fallback.
- `corpus_batches()` rejects Parquet containing any benchmark question/label
  column. The sampled document Parquet contains only `doc_id`, `source_type`,
  `title`, and `content`. Gold answers remain in ignored raw question data.
- The local storage, PostgreSQL, and Ollama adapters reject non-loopback
  endpoints. The OpenAI adapter uses the official Responses API with
  `store=False`. Cloud search integration and ranking parity require separate
  tests.
- Default unit tests make no paid network calls. `llm-smoke` and `rag-smoke`
  make explicit OpenAI requests. No Azure account is required.
