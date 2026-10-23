"""Member content must stay out of the server's logs.

test_api.py replaces the whole pipeline with a stand-in, so it cannot see what the
pipeline itself logs (which is how question text reached the server log unnoticed).
These tests run the real graph behind the real API; only the language model and the
search index are faked (see pipeline_fakes.py), so no Ollama, GPU or network is needed.
"""

import logging

import pytest
from fastapi.testclient import TestClient

from pipeline_fakes import EARLIER_TURN, QUESTION, FakeVerifier, api_settings, ask
from sfsa_rag_assistant.api import _configure_logging, create_app


@pytest.fixture(autouse=True)
def restore_package_loggers():
    names = ["sfsa_rag_assistant", "sfsa_rag_assistant.api"]
    saved = {n: (logging.getLogger(n).level, list(logging.getLogger(n).handlers)) for n in names}
    yield
    for n, (level, handlers) in saved.items():
        logger = logging.getLogger(n)
        logger.setLevel(level)
        logger.handlers[:] = handlers


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
