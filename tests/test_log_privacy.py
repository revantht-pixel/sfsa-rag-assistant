"""Member content must stay out of the server's logs.

test_api.py replaces the whole pipeline with a stand-in, so it cannot see what the
pipeline itself logs (which is how question text reached the server log unnoticed).
These tests run the real graph behind the real API; only the language model and the
search index are faked, so no Ollama, GPU or network is needed.
"""

import importlib
import json
import logging

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from sfsa_rag_assistant.api import _configure_logging, create_app
from sfsa_rag_assistant.app_config import Settings
from sfsa_rag_assistant.wiki_auth import Identity

ORIGIN = "https://rag.example.org"
COOKIE = "wiki_db_session=goodsession; wiki_dbUserID=42"

# Stand-ins for member content. The fake model echoes them back the way a real one
# paraphrases a question, so they show up wherever the pipeline logs model output too.
QUESTION = "ZX-CONFIDENTIAL-QUESTION-7731"
EARLIER_TURN = "ZX-CONFIDENTIAL-EARLIER-TURN-4412"


class FakeVerifier:
    def verify(self, raw_cookie):
        return Identity(42, "Alice Member") if "goodsession" in (raw_cookie or "") else None


class FakeGenerator:
    """Stands in for the Ollama-backed TextGenerator."""

    validations = 0

    def __init__(self, *args, **kwargs):
        pass

    def load_model(self):
        pass

    def invoke(self, prompt):
        return f"The answer to {QUESTION}. " * 20

    def with_structured_output(self, schema):
        return _Structured(schema)


class _Structured:
    def __init__(self, schema):
        self.schema = schema

    def invoke(self, prompt):
        name = self.schema.__name__
        if name == "ContextualizedQuery":
            return self.schema(contextualized_query=f"Rewritten: {QUESTION}", reasoning=f"Used {EARLIER_TURN}")
        if name == "ContextSufficiencyCheck":
            return self.schema(decision="insufficient", reasoning=f"Nothing about {QUESTION} in the context")
        if name == "ResponseValidation":
            FakeGenerator.validations += 1
            if FakeGenerator.validations == 1:      # make the pipeline take its refine-and-regenerate path
                return self.schema(decision="needs_refinement", reasoning=f"Too thin on {QUESTION}",
                                   refined_query=f"More detail on {QUESTION}?")
            return self.schema(decision="satisfactory", reasoning="Fine.", refined_query=None)
        raise AssertionError(f"unexpected structured output: {name}")


class FakeDocumentRetriever:
    """Stands in for the FAISS index."""

    def __init__(self, *args, **kwargs):
        pass

    def get_retriever(self, *args, **kwargs):
        return self

    def invoke(self, query):
        return [Document(page_content="Wiki text about steel casting.", metadata={"source": "wikidocs/Rr008.pdf", "page": 4})]


def node_module(name):
    # The nodes package re-exports each function under its module's name, so a dotted
    # path would find the function; import_module returns the module itself.
    return importlib.import_module(f"sfsa_rag_assistant.nodes.{name}")


@pytest.fixture
def fake_pipeline(monkeypatch):
    FakeGenerator.validations = 0
    for node in ("check_context_sufficiency", "contextualize_query", "generate_response",
                 "validate_response", "web_search"):
        monkeypatch.setattr(node_module(node), "TextGenerator", FakeGenerator)
    monkeypatch.setattr(node_module("retrieve_context"), "DocumentRetriever", FakeDocumentRetriever)
    monkeypatch.setattr(node_module("web_search").settings, "web_search_enabled", False)


@pytest.fixture(autouse=True)
def restore_package_loggers():
    names = ["sfsa_rag_assistant", "sfsa_rag_assistant.api"]
    saved = {n: (logging.getLogger(n).level, list(logging.getLogger(n).handlers)) for n in names}
    yield
    for n, (level, handlers) in saved.items():
        logger = logging.getLogger(n)
        logger.setLevel(level)
        logger.handlers[:] = handlers


def api_settings():
    return Settings(
        _env_file=None,
        allowed_origins=ORIGIN,
        frame_ancestors="https://wiki.example.org",
        citation_hosts="wiki.sfsa.org",
        max_question_chars=500,
        max_history_chars=500,
        max_history_turns=4,
    )


def ask(client):
    body = {
        "question": QUESTION,
        "conversation_history": [
            {"role": "user", "content": EARLIER_TURN},
            {"role": "assistant", "content": "An earlier answer."},
        ],
    }
    return client.post(
        "/chat",
        content=json.dumps(body),
        headers={"Origin": ORIGIN, "Cookie": COOKIE, "Content-Type": "application/json"},
    )


# ── the real pipeline, behind the real API ───────────────────────────────────
def test_the_real_pipeline_never_writes_member_text_to_the_log(fake_pipeline, caplog):
    caplog.set_level(logging.DEBUG)                # as permissive as a host can make it
    with TestClient(create_app(api_settings(), verifier=FakeVerifier())) as client:   # startup applies the log policy
        response = ask(client)

    assert response.status_code == 200
    assert QUESTION in response.json()["response"]            # the fake model's answer: the pipeline really ran
    assert QUESTION not in caplog.text
    assert EARLIER_TURN not in caplog.text
    assert "The answer to" not in caplog.text                 # no answer text either
    assert "chat ok" in caplog.text and "Alice Member" in caplog.text    # the audit trail is intact


def test_control_the_fakes_do_reach_the_pipelines_log_calls(fake_pipeline, caplog):
    """Without the policy this harness does see the text -- so the test above is not vacuous."""
    caplog.set_level(logging.DEBUG)
    with TestClient(create_app(api_settings(), verifier=FakeVerifier())) as client:
        logging.getLogger("sfsa_rag_assistant").setLevel(logging.INFO)     # undo the policy after startup
        ask(client)

    assert QUESTION in caplog.text


# ── the policy itself ────────────────────────────────────────────────────────
def test_only_the_api_audit_logger_may_write_info_lines():
    _configure_logging("privacy_pkg_a", root=logging.Logger("bare-root"))

    assert logging.getLogger("privacy_pkg_a.api").isEnabledFor(logging.INFO)
    for module in ("graph", "generation", "retrieval", "nodes.generate_response", "nodes.some_future_node"):
        child = logging.getLogger(f"privacy_pkg_a.{module}")
        assert not child.isEnabledFor(logging.INFO), module     # new modules inherit the restriction
        assert child.isEnabledFor(logging.WARNING), module      # failures stay visible


def test_the_policy_applies_even_when_the_host_already_configured_logging():
    configured_root = logging.Logger("configured-root")
    configured_root.addHandler(logging.NullHandler())

    _configure_logging("privacy_pkg_b", root=configured_root)

    assert logging.getLogger("privacy_pkg_b").level == logging.WARNING
    assert logging.getLogger("privacy_pkg_b.api").level == logging.INFO
    assert logging.getLogger("privacy_pkg_b").handlers == []     # the host's own handlers are left alone
