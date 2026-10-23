"""When answering fails, the member gets a generic error and the log says it failed.

The pipeline catches its own errors and hands back an apology as the "answer", often with
the exception's text in it (a host and port, a file path). These tests run the real graph
behind the real API and break the language model or the search index on purpose (see
pipeline_fakes.py), so no Ollama, GPU or network is needed.
"""

import importlib
import logging

import pytest
from fastapi.testclient import TestClient

from pipeline_fakes import (QUESTION, FakeDocumentRetriever, FakeGenerator, FakeVerifier,
                            api_settings, ask, node_module)
from sfsa_rag_assistant.api import create_app

# What a real outage says: internals a member must never see.
OUTAGE = "HTTPConnectionPool(host='127.0.0.1', port=11434): Max retries exceeded with url: /api/chat"


@pytest.fixture
def client(fake_pipeline):
    with TestClient(create_app(api_settings(), verifier=FakeVerifier())) as client:
        yield client


# ── behind the API ───────────────────────────────────────────────────────────
def test_a_failing_model_gives_a_generic_error_not_the_exception_text(client, caplog):
    caplog.set_level(logging.INFO)
    FakeGenerator.generation_error = RuntimeError(OUTAGE)

    response = ask(client)

    assert response.status_code == 500
    assert "ref " in response.text                              # something to quote to the admins
    for internal in ("11434", "HTTPConnectionPool", "apologize"):
        assert internal not in response.text
    assert "chat failed" in caplog.text and "pipeline:generation" in caplog.text
    assert "chat ok" not in caplog.text                         # the audit trail no longer calls it a success


def test_a_failed_request_does_not_lock_the_member_out(client):
    FakeGenerator.generation_error = RuntimeError(OUTAGE)
    assert ask(client).status_code == 500

    FakeGenerator.reset()                                       # the model is back
    assert ask(client).status_code == 200


def test_a_search_index_failure_is_an_error_even_though_the_model_still_writes_an_answer(client, caplog):
    """Without its sources the answer is not trustworthy, so it is not delivered."""
    caplog.set_level(logging.INFO)
    FakeDocumentRetriever.error = FileNotFoundError("/home/ubuntu/sfsa-rag-assistant/data/vectordb/index.faiss")

    response = ask(client)

    assert response.status_code == 500 and "ref " in response.text
    assert "index.faiss" not in response.text and QUESTION not in response.text
    assert "pipeline:retrieval" in caplog.text and "chat ok" not in caplog.text


def test_a_model_hiccup_the_pipeline_recovers_from_is_not_an_error(client, caplog):
    """The first attempt fails; the validator asks for a retry; the retry works."""
    caplog.set_level(logging.INFO)
    FakeGenerator.generation_error = RuntimeError(OUTAGE)
    FakeGenerator.fail_on = {1}

    response = ask(client)

    assert response.status_code == 200
    assert QUESTION in response.json()["response"]
    assert "apologize" not in response.json()["response"]
    assert "chat ok" in caplog.text and "chat failed" not in caplog.text


# ── the signal itself ────────────────────────────────────────────────────────
def test_run_workflow_reports_where_it_failed_and_keeps_its_apology_for_command_line_users(fake_pipeline):
    run_workflow = importlib.import_module("sfsa_rag_assistant.graph").run_workflow
    FakeGenerator.generation_error = RuntimeError(OUTAGE)

    result = run_workflow(user_query=QUESTION)

    assert result["metadata"]["failure"] == "generation"
    assert "apologize" in result["response"]                    # unchanged for the CLI; the API withholds it


def test_run_workflow_reports_a_crash_and_a_missing_output_as_failures(monkeypatch):
    graph = importlib.import_module("sfsa_rag_assistant.graph")

    def crash():
        raise RuntimeError("boom")

    monkeypatch.setattr(graph, "build_graph", crash)
    assert graph.run_workflow(user_query=QUESTION)["metadata"]["failure"] == "workflow"

    class NoOutput:
        def invoke(self, state):
            return {}

    monkeypatch.setattr(graph, "build_graph", lambda: NoOutput())
    assert graph.run_workflow(user_query=QUESTION)["metadata"]["failure"] == "no_output"


def test_generate_response_flags_what_went_wrong(fake_pipeline):
    generate = node_module("generate_response").generate_response
    ready = {"user_query": "q", "retrieved_formatted": "some context"}

    assert generate(ready)["failure"] is None

    FakeGenerator.generation_error = RuntimeError(OUTAGE)
    assert generate(ready)["failure"] == "generation"

    assert generate({"user_query": "q"})["failure"] == "no_context"
    assert generate({"retrieved_formatted": "some context"})["failure"] == "no_query"


def test_a_generation_retry_that_works_clears_a_generation_failure_but_not_a_retrieval_failure(fake_pipeline):
    generate = node_module("generate_response").generate_response
    ready = {"user_query": "q", "retrieved_formatted": "some context"}

    assert generate({**ready, "failure": "generation"})["failure"] is None          # a retry that works clears it
    assert generate({**ready, "failure": "retrieval"})["failure"] == "retrieval"    # an answer without sources stays flagged

    FakeGenerator.generation_error = RuntimeError(OUTAGE)
    assert generate({**ready, "failure": "retrieval"})["failure"] == "retrieval"    # ...even if generation fails as well


def test_retrieve_context_flags_a_failed_search(fake_pipeline):
    retrieve = node_module("retrieve_context").retrieve_context
    FakeDocumentRetriever.error = FileNotFoundError("index.faiss")

    result = retrieve({"user_query": "q"})

    assert result["failure"] == "retrieval"
    assert result["retrieved_docs"] == []
