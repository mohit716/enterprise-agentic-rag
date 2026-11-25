import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from enterprise_rag.dataset import audit, corpus_batches, create_sample


def _parquet(path, rows):
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_audit_and_sample_keep_labels_out_of_retrieval(tmp_path):
    docs = tmp_path / "documents.parquet"
    questions = tmp_path / "questions.parquet"
    rows = []
    for source in ("slack", "jira"):
        for number, length in enumerate((5, 10, 20, 30, 40, 50)):
            rows.append({"doc_id": f"{source}-{number}", "source_type": source,
                         "title": f"{source} {number}", "content": "x" * length})
    rows.append(dict(rows[0]))  # A duplicate is reported, not silently counted as unique.
    _parquet(docs, rows)
    _parquet(questions, [{"question_id": "q1", "question_type": "basic", "source_types": ["jira"],
                          "question": "What is in Jira?", "expected_doc_ids": ["jira-0"],
                          "gold_answer": "secret answer", "answer_facts": ["secret fact"]}])
    report = audit(docs, questions, batch_size=2)
    assert report["scan_complete"]
    assert report["documents"]["duplicate_doc_ids"] == 1
    assert report["questions"]["expected_ids_missing_from_corpus"] == 0
    sample_dir = tmp_path / "sample"
    sample = create_sample(docs, questions, sample_dir, per_source=3, dev_questions=1,
                           length_limits=report["documents"]["content_length_chars"])
    assert sample["dev_support_coverage"] == {"q1": 1}
    assert "jira-0" in sample["sample_doc_ids"]
    assert set(pq.ParquetFile(sample_dir / "documents.parquet").schema_arrow.names) == {
        "doc_id", "source_type", "title", "content"}
    assert json.loads((sample_dir / "dev_questions.json").read_text()) == [
        {"question_id": "q1", "question": "What is in Jira?"}]
    for path in sample_dir.iterdir():
        assert b"secret answer" not in path.read_bytes()
        assert b"secret fact" not in path.read_bytes()
    second = tmp_path / "same-seed"
    create_sample(docs, questions, second, per_source=3, dev_questions=1,
                  length_limits=report["documents"]["content_length_chars"])
    assert (sample_dir / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()


def test_rejects_benchmark_fields_on_runtime_corpus(tmp_path):
    path = tmp_path / "leaky.parquet"
    _parquet(path, [{"doc_id": "d", "source_type": "jira", "title": "t",
                     "content": "body", "gold_answer": "do not retrieve"}])
    with pytest.raises(ValueError, match="forbidden"):
        list(corpus_batches(path))
