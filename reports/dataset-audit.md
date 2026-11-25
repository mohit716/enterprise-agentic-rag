# EnterpriseRAG-Bench audit — 2026-11-24

This report records measured findings for Jira EAR-19 and EAR-20. The source is
the official [Onyx dataset](https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench)
at commit `69916e31c68aa5963c00248fd7f0bc12d04fd235` (MIT license, per its
dataset card). The Jira story refers to `architecture.md`, but that file was
not present in this project workspace when the audit was run. No comparison
against it was possible.

## Measured files and scan

- `data/documents/test.parquet`: 1,409,893,131 bytes, SHA-256
  `6b0747bf160af9427b12101537d53056ac592ada9831c1a98ae01fa50a8d2a9f`,
  511,962 metadata rows and 511,962 scanned rows, one row group. Fields:
  `doc_id`, `source_type`, `title`, `content`, all strings.
- `data/questions/test.parquet`: 408,737 bytes, SHA-256
  `e25066f4eff3843dd0f3df0d1348113471e072e75007ffe390a0aa83f2a80af2`,
  500 metadata rows and 500 scanned rows, one row group. Fields:
  `question_id`, `question_type`, `source_types`, `question`,
  `expected_doc_ids`, `gold_answer`, and `answer_facts`.
- The full scan completed. Missing corpus values: one `content`, 15 `title`.
  Four repeated `doc_id` rows were found. No missing required question values
  or duplicate `question_id` values were found. Every expected document ID
  referenced by the questions exists in the corpus.

Measured source rows: Slack 285,605; Gmail 121,390; Linear 35,308; Google
Drive 25,108; HubSpot 15,017; Fireflies 10,173; GitHub 8,052; Jira 6,120;
Confluence 5,189. These sum to 511,962. The official dataset card's source
counts match the measured counts; its headline rounds this to 500,000+.

Content lengths below are characters, not tokens. Each entry is
`source: p50 / p95 / max` from the complete scan:

- Slack: 3,239 / 5,524 / 12,665
- Gmail: 7,036 / 9,403 / 17,435
- Linear: 5,041 / 7,712 / 17,311
- Google Drive: 7,005 / 9,902 / 22,666
- HubSpot: 3,025 / 4,789 / 8,338
- Fireflies: 11,080 / 16,327 / 41,317
- GitHub: 4,616 / 6,918 / 13,773
- Jira: 5,557 / 7,202 / 10,435
- Confluence: 9,579 / 14,232 / 24,851

The question type counts also match the dataset card: basic 175, semantic 125,
intra-document reasoning 40, project-related 40, constrained 30,
conflicting-info 20, completeness 20, miscellaneous 20, high-level 10, and
info-not-found 20. The report distinguishes these measurements from the card's
claims; the card supplied the license and dataset description, not these audit
results. The command uses PyArrow batches and a temporary SQLite file for IDs
and lengths. The Parquet files have one large row group each, so peak decoder
memory can still depend on PyArrow's page handling.

## Development boundary and next decisions

`python -m enterprise_rag.cli sample` generated 209 sampled document IDs with
seed 2026: Confluence 26, Fireflies 22, GitHub 23, Gmail 23, Google Drive 22,
HubSpot 21, Jira 26, Linear 23, Slack 23. The base quota is 21 per source,
split across short, medium, and long strata using measured p50 and p95 values;
additional documents ensure coverage for ten selected development questions.
The sample has ten question-only prompts and 490 held-out questions. Its
manifest records corpus hashes, IDs, source counts, and support coverage.

The runtime corpus accepts only the four document fields; a schema guard and
tests reject gold answers, answer facts, expected IDs, and other question
columns. The raw questions file remains ignored under `data/raw/`, outside
runtime input paths. The small sample limits retrieval conclusions. Four
duplicate IDs and one missing body should be resolved by explicit ingestion
policy before full indexing. Lengths argue for a measured chunking experiment
in EAR-4; this audit does not select a production chunk size or Azure tier.
