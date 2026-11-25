"""Opt-in contract checks against the real Docker Compose services."""

import os

import pytest

from enterprise_rag.providers import (AzuriteBlobStorage, Document, PgvectorSearch,
                                      local_settings)


@pytest.mark.skipif(os.environ.get("RUN_LOCAL_INTEGRATION") != "1",
                    reason="Set RUN_LOCAL_INTEGRATION=1 after docker compose up -d")
def test_storage_and_search_contract():
    settings = local_settings()
    blob = AzuriteBlobStorage(settings["azurite"], container="contract-test")
    document = Document("contract-doc", "contract_fixture", "Title", "A bounded contract fixture")
    blob.put(document)
    assert blob.get(document.doc_id) == document
    assert blob.get("unknown-contract-doc") is None

    search = PgvectorSearch(settings["pg_dsn"], 384)
    try:
        vector = [1.0] + [0.0] * 383
        search.upsert([document], [vector])
        assert search.search(vector, source_type="contract_fixture")[0].doc_id == document.doc_id
        assert search.search(vector, source_type="never_contract_fixture") == []
        with pytest.raises(ValueError, match="dimension"):
            search.search([1.0])
    finally:
        search.close()
