"""Run the three original questions with a 2048-token limit on the V2 index.

Run from the project root: python scripts/evaluate_generation_v4.py
The evaluation corpus and PostgreSQL vectors are verified, never reingested.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from enterprise_rag.providers import (
    OpenAIAnswerModel, SearchHit, SentenceTransformerEmbeddings,
    answer_citations, configured_answer_model, local_settings,
)
from scripts.evaluate_coverage import OUTPUT_DIR, QUESTIONS, ROOT, TARGET_IDS, file_hash, read_baseline
from scripts.evaluate_generation_v3 import evaluate, read_v2, verify_existing_index


V3 = ROOT / "rag_test_results_v3.txt"
RESULTS = ROOT / "rag_test_results_v4.txt"
HISTORICAL_COMPLETENESS = {
    "V1": (
        "No metrics answer; retrieved evidence was insufficient.",
        "No disable procedure or dashboards answer; retrieved evidence was insufficient.",
        "Text ends mid-sentence and omits several requested requirements.",
    ),
    "V2": (
        "Answers both purpose and metrics.",
        "Text ends inside a citation ID.",
        "Stops after technical requirements; security, billing, and operations are missing.",
    ),
    "V3": (
        "API completed; answers both purpose and metrics.",
        "API completed; answers disable procedure and dashboards.",
        "API status incomplete at the 1024-token limit; answer ends inside a citation.",
    ),
}


def read_v3() -> list[dict]:
    text = V3.read_text(encoding="utf-8")
    sections = re.split(r"(?m)^QUESTION ([123])\s*$", text)[1:]
    found = {}
    for number, section in zip(sections[::2], sections[1::2]):
        def field(name: str, *, indent: str = "") -> str:
            match = re.search(rf"(?m)^{re.escape(indent + name)}: (.*)$", section)
            return match.group(1).strip() if match else ""
        answer_match = re.search(r"Generated answer:\s*\n(.*?)\n\s*Source citations in generated answer:", section, re.S)
        hits = re.findall(r"(?m)^  Rank (\d+): (dsid_[0-9a-f]+) \|.*?similarity=([0-9.]+)", section)
        found[int(number)] = {
            "question": field("Question"),
            "answer": answer_match.group(1).strip() if answer_match else "",
            "citations": field("Source citations in generated answer"),
            "response_time": field("Overall response time"),
            "errors": field("Errors"),
            "hits": [(int(rank), doc_id, float(score)) for rank, doc_id, score in hits],
            "status": field("Response status", indent="  "),
            "incomplete_reason": field("Incomplete reason", indent="  "),
            "input_tokens": field("Input tokens", indent="  "),
            "cached_input_tokens": field("Cached input tokens", indent="  "),
            "output_tokens": field("Output tokens", indent="  "),
            "reasoning_tokens": field("Reasoning tokens", indent="  "),
        }
    if set(found) != {1, 2, 3} or any(found[i]["question"] != QUESTIONS[i - 1] for i in found):
        raise ValueError("V3 does not contain the exact three evaluation questions")
    return [found[i] for i in (1, 2, 3)]


def citation_audit(answer: str, ids: list[str]) -> str:
    hits = [SearchHit(doc_id, f"{doc_id}#0", "", "", "", 0.0) for doc_id in ids]
    try:
        answer_citations(answer, hits)
    except ValueError as exc:
        return f"invalid ({exc})"
    return "valid"


def category_check(answer: str) -> dict[str, str]:
    """Check for substantive cited sections; human review still checks meaning."""
    sections = re.findall(r"(?ms)^### ([^\n]+)\n(.*?)(?=^### |\Z)", answer)
    headings = {
        "technical": r"technical",
        "security": r"security",
        "billing": r"billing|cost controls",
        "operational": r"operations|operational",
    }
    result = {}
    for category, pattern in headings.items():
        match = next(((heading, body) for heading, body in sections
                      if re.search(pattern, heading, re.I)), None)
        if match is None:
            result[category] = "MISSING section"
            continue
        heading, body = match
        bullets = len(re.findall(r"(?m)^- ", body))
        citations = len(re.findall(r"\[[A-Za-z0-9_-]+\]", body))
        outcome = "PRESENT" if bullets >= 2 and citations >= 1 else "INSUFFICIENT"
        result[category] = f"{outcome}: {heading}; {bullets} bullets; {citations} citations"
    return result


def render(results: list[dict], history: dict[str, list[dict]], settings: dict) -> str:
    manifest = json.loads((OUTPUT_DIR / "manifest.json").read_text(encoding="utf-8"))
    lines = [
        "RAG answer-generation evaluation (v4)",
        f"Run time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "Evaluation sample: data/derived/evaluation_coverage/documents.parquet",
        f"Evaluation sample SHA-256: {manifest['evaluation_parquet_sha256']}",
        "Index: existing isolated PostgreSQL enterprise_rag_coverage_eval, 212 unique documents; no ingestion performed",
        f"V1 result SHA-256: {file_hash(ROOT / 'rag_test_results.txt')}",
        f"V2 result SHA-256: {file_hash(ROOT / 'rag_test_results_v2.txt')}",
        f"V3 result SHA-256: {file_hash(V3)}",
        f"Embedding: {settings['model_id']} @ {settings['revision']}",
        f"Answer provider/model: {settings['llm_provider']} / {settings['openai_model']}",
        "Output-token limit: 2048; retrieval limit: 5; similarity = 1 - pgvector cosine distance.",
        "Prompt construction, retrieval, and evaluation corpus are unchanged from V3.",
        "Each question was submitted independently with no conversation history.",
        "Overall response latency includes query embedding, retrieval, and generation; model initialization is excluded.",
        "V1/V2 did not record API usage or status; N/A below means unavailable, not zero.",
        "Citation validity for historical answers is rechecked with the current strict validator.", "",
    ]
    for number, result in enumerate(results, 1):
        meta = asdict(result["generation_metadata"]) if result["generation_metadata"] else {}
        ids = [hit.doc_id for hit in result["hits"]]
        validity = citation_audit(result["answer"], ids)
        complete = meta.get("status") == "completed" and validity == "valid" and not result["errors"]
        lines += ["=" * 80, f"QUESTION {number}", "=" * 80,
                  f"Question: {result['question']}", "", "Generated answer:", result["answer"] or "<none>", "",
                  "Answer completion: " + ("API completed; citations valid" if complete else "INCOMPLETE OR INVALID; see status and errors"),
                  "Citation validity: " + validity,
                  "Source citations in generated answer: " + (", ".join(result["citations"]) or "None"),
                  f"Overall response latency: {result['response_time']:.3f} seconds",
                  "Stage latencies (seconds): " + ", ".join(f"{key}={value:.3f}" for key, value in result["stage_time"].items()),
                  "Generation metadata:"]
        for label, key in (
            ("Response ID", "response_id"), ("API response status", "status"),
            ("Incomplete reason", "incomplete_reason"), ("Input tokens", "input_tokens"),
            ("Cached input tokens", "cached_input_tokens"), ("Output tokens", "output_tokens"),
            ("Reasoning tokens", "reasoning_tokens"), ("Total tokens", "total_tokens"),
            ("API response latency (seconds)", "response_time_seconds"),
        ):
            value = meta.get(key)
            lines.append(f"  {label}: {value:.3f}" if key == "response_time_seconds" and value is not None
                         else f"  {label}: {value if value is not None else 'N/A'}")
        lines += ["Errors: " + ("; ".join(result["errors"]) or "None"),
                  f"Retrieved documents ({len(result['hits'])}):"]
        for rank, hit in enumerate(result["hits"], 1):
            lines.append(f"  Rank {rank}: {hit.doc_id} | similarity={1-hit.distance:.6f} | title={hit.title}")
        if number == 3:
            lines += ["", "Question 3 category structure (manual content review also required):"]
            for category, evidence in category_check(result["answer"]).items():
                lines.append(f"  {category}: {evidence}")
        lines += ["", "Comparison against V1, V2, and V3:"]
        for label in ("V1", "V2", "V3"):
            old = history[label][number - 1]
            first = old["hits"][0] if old["hits"] else None
            old_ids = [doc_id for _, doc_id, _ in old["hits"]]
            lines += [f"  {label} answer completeness: {HISTORICAL_COMPLETENESS[label][number - 1]}",
                      f"  {label} expected document ranks: " + ", ".join(
                          f"{doc_id}={next((rank for rank, key, _ in old['hits'] if key == doc_id), 'absent')}"
                          for doc_id in (TARGET_IDS[:1] if number in (1, 2) else TARGET_IDS[1:])),
                      f"  {label} top hit: " + (f"{first[1]} (similarity {first[2]:.6f})" if first else "None"),
                      f"  {label} citation validity: {citation_audit(old['answer'], old_ids)}",
                      f"  {label} citations: {old['citations'] or 'None'}",
                      f"  {label} API status: {old.get('status', 'N/A')}",
                      f"  {label} tokens (input/cached/output/reasoning): " + "/".join(
                          old.get(key, "N/A") for key in
                          ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens")),
                      f"  {label} response latency: {old['response_time']}",
                      f"  {label} recorded errors: {old['errors'] or 'None'}"]
        lines += ["  V4 expected document ranks: " + ", ".join(
            f"{doc_id}={next((rank for rank, hit in enumerate(result['hits'], 1) if hit.doc_id == doc_id), 'absent')}"
            for doc_id in (TARGET_IDS[:1] if number in (1, 2) else TARGET_IDS[1:])),
            f"  V4 answer completeness: {'API completed and citations valid' if complete else 'INCOMPLETE OR INVALID'}",
            f"  V4 citation validity: {validity}",
            f"  V4 API status: {meta.get('status', 'N/A')}",
            "  V4 tokens (input/cached/output/reasoning): " + "/".join(
                str(meta.get(key, "N/A")) for key in
                ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens")),
            f"  V4 response latency: {result['response_time']:.3f} seconds",
            "  V4 errors: " + ("; ".join(result["errors"]) or "None"), ""]
    lines.append("Interpretation: The only changed request parameter versus V3 is max_output_tokens=2048. A completed API status and valid citations do not by themselves prove factual completeness; Question 3's four categories require content review.")
    return "\n".join(lines) + "\n"


def main() -> None:
    history = {"V1": read_baseline(), "V2": read_v2(), "V3": read_v3()}
    settings = local_settings()
    if settings["llm_provider"] != "openai":
        raise ValueError("V4 evaluation requires LLM_PROVIDER=openai")
    embedding = SentenceTransformerEmbeddings(settings["model_id"], settings["revision"])
    search = verify_existing_index(settings, embedding)
    try:
        llm = configured_answer_model(settings)
        if not isinstance(llm, OpenAIAnswerModel) or llm.max_tokens != 2048:
            raise ValueError("V4 evaluation requires OpenAI with a 2048-token output limit")
        results = evaluate(search, embedding, llm)
    finally:
        search.close()
    RESULTS.write_text(render(results, history, settings), encoding="utf-8")
    print(f"Wrote {RESULTS}; {sum(bool(r['errors']) for r in results)} question(s) had errors")


if __name__ == "__main__":
    main()
