"""Stand-ins for the language model and the search index, so tests can run the real graph.

Not a test file: tests/conftest.py installs these (the `fake_pipeline` fixture) and the
tests import the constants and helpers they need. Nothing here needs Ollama, a GPU or
the network.
"""

import importlib
import json

from langchain_core.documents import Document

from sfsa_rag_assistant.app_config import Settings
from sfsa_rag_assistant.wiki_auth import Identity

ORIGIN = "https://rag.example.org"
COOKIE = "wiki_db_session=goodsession; wiki_dbUserID=42"

# Stand-ins for member content. The fake model echoes them back the way a real one
# paraphrases a question, so they show up wherever the pipeline handles model output too.
QUESTION = "ZX-CONFIDENTIAL-QUESTION-7731"
EARLIER_TURN = "ZX-CONFIDENTIAL-EARLIER-TURN-4412"


class FakeVerifier:
    def verify(self, raw_cookie):
        return Identity(42, "Alice Member") if "goodsession" in (raw_cookie or "") else None


class FakeGenerator:
    """Stands in for the Ollama-backed TextGenerator."""

    validations = 0
    generation_calls = 0
    generation_error = None     # an exception for invoke() to raise
    fail_on = None              # which invoke() calls raise it (1-based); None = every call

    @classmethod
    def reset(cls):
        cls.validations = 0
        cls.generation_calls = 0
        cls.generation_error = None
        cls.fail_on = None

    def __init__(self, *args, **kwargs):
        pass

    def load_model(self):
        pass

    def invoke(self, prompt):
        type(self).generation_calls += 1
        if self.generation_error is not None and (self.fail_on is None or self.generation_calls in self.fail_on):
            raise self.generation_error
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

    error = None                # an exception for invoke() to raise

    @classmethod
    def reset(cls):
        cls.error = None

    def __init__(self, *args, **kwargs):
        pass

    def get_retriever(self, *args, **kwargs):
        return self

    def invoke(self, query):
        if self.error is not None:
            raise self.error
        return [Document(page_content="Wiki text about steel casting.", metadata={"source": "wikidocs/Rr008.pdf", "page": 4})]


def node_module(name):
    # The nodes package re-exports each function under its module's name, so a dotted
    # path would find the function; import_module returns the module itself.
    return importlib.import_module(f"sfsa_rag_assistant.nodes.{name}")


def install(monkeypatch):
    """Replace the model and the index in the real graph's nodes (undone by monkeypatch)."""
    FakeGenerator.reset()
    FakeDocumentRetriever.reset()
    for node in ("check_context_sufficiency", "contextualize_query", "generate_response",
                 "validate_response", "web_search"):
        monkeypatch.setattr(node_module(node), "TextGenerator", FakeGenerator)
    monkeypatch.setattr(node_module("retrieve_context"), "DocumentRetriever", FakeDocumentRetriever)
    monkeypatch.setattr(node_module("web_search").settings, "web_search_enabled", False)


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
