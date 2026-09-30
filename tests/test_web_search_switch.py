"""
The web-search agent is the one place a member's question could leave the server
(it sends a search query to Tavily). SFSA's rule is that it stays off, so these
tests pin that: off by default, off even when a key is present, and only on when
deliberately switched on. Tavily is always faked here; nothing touches a network.
"""

import importlib
import sys
import types

import pytest

from sfsa_rag_assistant.app_config import Settings

ws = importlib.import_module("sfsa_rag_assistant.nodes.web_search")

SECRET_QUESTION = "What is our proprietary alloy X pour temperature?"
PLACEHOLDER_KEY = "your_tavily_api_key_here"   # what .env.example used to ship


class FakeTavily:
    """Records every search that would have left the server."""

    searches = []

    def __init__(self, api_key=None):
        self.api_key = api_key

    def search(self, query, **kwargs):
        FakeTavily.searches.append(query)
        return {"results": [{"title": "T", "url": "https://example.org/t", "content": "C", "score": 0.9}]}


@pytest.fixture(autouse=True)
def fake_tavily(monkeypatch):
    FakeTavily.searches = []
    module = types.ModuleType("tavily")
    module.TavilyClient = FakeTavily
    monkeypatch.setitem(sys.modules, "tavily", module)
    # Reformulation uses the local LLM; replace it so the tests never need a model.
    monkeypatch.setattr(ws, "_reformulate_query", lambda query, context: f"reformulated: {query}")


def use(monkeypatch, enabled=False, key=None):
    # tavily_api_key is read through the TAVILY_API_KEY alias; passing it by field name
    # is silently ignored, which would make every "a key is present" test vacuous.
    cfg = Settings(_env_file=None, web_search_enabled=enabled, TAVILY_API_KEY=key)
    assert cfg.tavily_api_key == key
    monkeypatch.setattr(ws, "settings", cfg)
    return cfg


def test_web_search_is_off_by_default():
    assert Settings(_env_file=None).web_search_enabled is False


def test_disabled_node_sends_nothing_and_skips_the_llm(monkeypatch):
    use(monkeypatch, key="tvly-a-real-looking-key")   # a real key must not matter
    monkeypatch.setattr(ws, "_reformulate_query", lambda *a: pytest.fail("LLM was called"))
    out = ws.web_search({"user_query": SECRET_QUESTION, "retrieved_formatted": "ctx"})
    assert FakeTavily.searches == []
    assert out["web_search_results"] == []
    assert out["web_search_formatted"] == "No web search results found."
    assert out["error"] is None


def test_a_placeholder_key_can_no_longer_leak_a_query(monkeypatch):
    use(monkeypatch, key=PLACEHOLDER_KEY)
    ws.web_search({"user_query": SECRET_QUESTION})
    assert FakeTavily.searches == []


def test_the_search_function_refuses_on_its_own_when_disabled(monkeypatch):
    use(monkeypatch, key="tvly-a-real-looking-key")
    assert ws._search_tavily(SECRET_QUESTION) == []
    assert FakeTavily.searches == []


def test_enabling_the_switch_is_what_turns_search_on(monkeypatch):
    # Positive control: proves the tests above pass because of the switch,
    # not because the search path is broken.
    use(monkeypatch, enabled=True, key="tvly-a-real-looking-key")
    out = ws.web_search({"user_query": SECRET_QUESTION, "retrieved_formatted": "ctx"})
    assert FakeTavily.searches == [f"reformulated: {SECRET_QUESTION}"]
    assert out["web_search_results"][0]["url"] == "https://example.org/t"


def test_enabled_but_no_key_still_sends_nothing(monkeypatch):
    use(monkeypatch, enabled=True, key=None)
    out = ws.web_search({"user_query": SECRET_QUESTION})
    assert FakeTavily.searches == []
    assert out["web_search_results"] == []


def test_the_switch_reads_the_environment_variable(monkeypatch):
    monkeypatch.setenv("SFSA_WEB_SEARCH_ENABLED", "true")
    assert Settings(_env_file=None).web_search_enabled is True
    monkeypatch.setenv("SFSA_WEB_SEARCH_ENABLED", "false")
    assert Settings(_env_file=None).web_search_enabled is False


# ── the package itself is not installed (project decision, 2026-09-28) ───────
# The fixture above stubs `sys.modules["tavily"]`, which proves the *switch* works but
# would pass even if the real package were missing entirely -- it never exercises the
# actual ImportError path. These two tests bypass that stub to prove the package's
# absence is itself harmless: no crash, no network attempt, just a clear log line.
def test_the_real_package_is_not_installed(monkeypatch):
    monkeypatch.delitem(sys.modules, "tavily", raising=False)  # undo the autouse stub above
    with pytest.raises(ImportError):
        import tavily  # noqa: F401


def test_missing_package_fails_closed_even_if_the_switch_were_mistakenly_on(monkeypatch, fake_tavily):
    monkeypatch.delitem(sys.modules, "tavily", raising=False)  # undo the fixture's stub
    use(monkeypatch, enabled=True, key="tvly-a-real-looking-key")
    assert ws._search_tavily(SECRET_QUESTION) == []
