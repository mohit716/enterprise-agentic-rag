from unittest.mock import Mock, patch
from types import SimpleNamespace

import pytest

from enterprise_rag.providers import (AnswerGenerationStatusError, Document, OllamaAnswerModel, OpenAIAnswerModel,
                                      SearchHit, answer_citations, configured_answer_model,
                                      local_settings, validate_vectors)


def test_vector_contract_and_document_validation():
    document = Document("d1", "jira", "Title", "Body")
    validate_vectors([document], [[0.1, 0.2]], 2)
    with pytest.raises(ValueError, match="counts"):
        validate_vectors([document], [], 2)
    with pytest.raises(ValueError, match="dimensions"):
        validate_vectors([document], [[0.1]], 2)
    with pytest.raises(ValueError, match="finite"):
        validate_vectors([document], [[float("nan"), 0.2]], 2)
    with pytest.raises(ValueError, match="nonempty"):
        Document("d1", "jira", "Title", "")


def test_ollama_prompt_and_failures():
    llm = OllamaAnswerModel("http://127.0.0.1:11434", "local-test", timeout=2, max_tokens=20)
    hit = SearchHit("d1", "d1#0", "jira", "Title", "Body", 0.1)
    response = Mock()
    response.json.return_value = {"response": "Supported claim [d1]"}
    with patch("requests.post", return_value=response) as post:
        assert llm.answer("Question?", [hit]) == "Supported claim [d1]"
        args = post.call_args
        assert args.kwargs["timeout"] == 2
        assert args.kwargs["json"]["options"]["num_predict"] == 20
        assert "[d1]" in args.kwargs["json"]["prompt"]
        assert "do not know" in args.kwargs["json"]["prompt"]
        response.json.return_value = {"response": ""}
        with pytest.raises(ValueError, match="empty answer"):
            llm.answer("Question?", [hit])
    with pytest.raises(ValueError, match="loopback"):
        OllamaAnswerModel("https://api.example.com", "paid-model")


def test_openai_responses_contract_and_failures():
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        id="resp-completed", status="completed", incomplete_details=None,
        output_text="Alder [smoke-doc]",
        usage=SimpleNamespace(input_tokens=30, output_tokens=8, total_tokens=38,
                              input_tokens_details=SimpleNamespace(cached_tokens=5),
                              output_tokens_details=SimpleNamespace(reasoning_tokens=0)),
    )
    llm = OpenAIAnswerModel("gpt-6-luna", timeout=3, max_tokens=128, client=client)
    hit = SearchHit("smoke-doc", "smoke-doc#0", "test", "Fixture",
                    "x" * 4000 + " Alder", 0.0)
    assert llm.answer("Name?", [hit]) == "Alder [smoke-doc]"
    sent = client.responses.create.call_args.kwargs
    assert sent["model"] == "gpt-6-luna"
    assert sent["max_output_tokens"] == 128
    assert sent["reasoning"] == {"effort": "none"}
    assert sent["store"] is False
    assert "[smoke-doc]" in sent["input"]
    assert "Alder" in sent["input"]
    assert llm.last_generation.response_id == "resp-completed"
    assert llm.last_generation.status == "completed"
    assert llm.last_generation.input_tokens == 30
    assert llm.last_generation.cached_input_tokens == 5
    assert llm.last_generation.output_tokens == 8
    assert llm.last_generation.reasoning_tokens == 0
    assert llm.last_generation.total_tokens == 38
    assert llm.last_generation.response_time_seconds >= 0
    client.responses.create.return_value.output_text = ""
    with pytest.raises(ValueError, match="empty answer"):
        llm.answer("Name?", [hit])
    client.responses.create.side_effect = RuntimeError("network failure")
    with pytest.raises(RuntimeError, match="network failure"):
        llm.answer("Name?", [hit])
    assert llm.last_generation is None
    assert answer_citations("Alder [smoke-doc]", [hit]) == ["smoke-doc"]
    with pytest.raises(ValueError, match="absent from retrieval"):
        answer_citations("Alder [invented-doc]", [hit])


def test_openai_incomplete_response_preserves_reason_and_partial_answer():
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        id="resp-incomplete", status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        output_text="Partial [smoke-doc",
        usage=SimpleNamespace(input_tokens=900, output_tokens=1024, total_tokens=1924,
                              input_tokens_details=SimpleNamespace(cached_tokens=800),
                              output_tokens_details=SimpleNamespace(reasoning_tokens=0)),
    )
    llm = OpenAIAnswerModel("gpt-6-luna", client=client)
    hit = SearchHit("smoke-doc", "smoke-doc#0", "test", "Fixture", "Body", 0.0)
    with pytest.raises(AnswerGenerationStatusError, match="max_output_tokens") as caught:
        llm.answer("Question?", [hit])
    assert caught.value.partial_answer == "Partial [smoke-doc"
    assert caught.value.metadata.response_id == "resp-incomplete"
    assert caught.value.metadata.status == "incomplete"
    assert caught.value.metadata.incomplete_reason == "max_output_tokens"
    assert caught.value.metadata.output_tokens == 1024


@pytest.mark.parametrize("answer", [
    "Claim [smoke-doc", "Claim [bad citation]", "Claim []",
    "Claim [smoke-doc]]", "Claim [[smoke-doc]]", "Claim [smoke-doc] [other",
])
def test_citation_validator_rejects_unclosed_or_malformed_references(answer):
    hit = SearchHit("smoke-doc", "smoke-doc#0", "test", "Fixture", "Body", 0.0)
    with pytest.raises(ValueError, match="Malformed or unclosed citation"):
        answer_citations(answer, [hit])


def test_answer_provider_selection_requires_key(monkeypatch):
    settings = {"llm_provider": "openai", "openai_model": "gpt-6-luna"}
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        configured_answer_model(settings)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MAX_TOKENS", "2048")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "60")
    monkeypatch.setenv("OPENAI_CONTEXT_CHARS_PER_HIT", "12000")
    with patch("enterprise_rag.providers.OpenAIAnswerModel") as model:
        assert configured_answer_model(settings) is model.return_value
        model.assert_called_once_with("gpt-6-luna", 60, 2048, 12000)


def test_local_settings_reject_cloud_db(monkeypatch):
    monkeypatch.setenv("PROVIDER_MODE", "local")
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("PG_DSN", "postgresql://u:p@db.example.com:5432/app")
    with pytest.raises(ValueError, match="loopback"):
        local_settings()
