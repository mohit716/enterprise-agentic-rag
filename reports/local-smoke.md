# Local provider smoke check — 2026-11-24 (America/New_York)

The bounded development sample was tested against real local services and the
configured OpenAI answer provider. This is an integration check, not a
full-corpus benchmark score.

- `docker compose up -d --wait`: Azurite 3.36.0 and PostgreSQL 17 with
  pgvector 0.8.7 reached healthy status. PostgreSQL uses host port 55432
  because 5432 was already occupied on this machine.
- `python -m enterprise_rag.cli infra-smoke`: Azurite Blob upload/download and
  missing-record checks passed; PostgreSQL was reachable with pgvector enabled.
- `RUN_LOCAL_INTEGRATION=1 python -m pytest -q`: 8 tests passed, including the
  real Blob round trip, vector search, and metadata filter contract.
- `python -m enterprise_rag.cli llm-smoke`: one bounded OpenAI Responses API
  call succeeded with a synthetic answer and valid citation.
- `rag-smoke` with development question `qst_0380`: 209 sample documents were
  indexed. Post-run evaluation confirmed that the top two retrieved documents
  matched both support IDs. The OpenAI answer cited both,
  identified the `x-redwood-template-tags` header and opt-in default, and
  described how disabling telemetry lets a request proceed through a proxy.
  Post-run comparison with the evaluation reference found that it omitted the
  full `rw.template.*` field list and the explicit potential 4xx response.
- `rag-smoke` with an out-of-domain blue-teapot question returned an explicit
  unknown answer and no citations.

The first development run used only 1,800 characters per retrieved document
and missed the header specification at character 4,115 of a supporting
document. The OpenAI adapter now sends at most 12,000 characters per hit and
checks that any cited document ID was retrieved. This fixed the observed
truncation failure while bounding request size. These results do not establish
retrieval quality across the held-out questions or equivalence with Azure
Search. The two local containers remain running for development.
