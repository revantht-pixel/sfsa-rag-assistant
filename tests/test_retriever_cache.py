"""The search index is loaded once per process, not once per question.

Building a DocumentRetriever loads the embedding model and the whole index (about 17 s on the
production box). These tests count constructions with a stand-in, so nothing is loaded.
"""

import importlib
import threading
import time

import pytest
from langchain_core.documents import Document

retrieve_module = importlib.import_module("sfsa_rag_assistant.nodes.retrieve_context")


class CountingRetriever:
    """Stands in for DocumentRetriever and records how it is used."""

    built = []                  # (index path, model) for each successful construction
    fail_next_build = False
    build_seconds = 0.0
    search_seconds = 0.0
    active = 0                  # searches running right now
    max_active = 0              # the most that ever ran at once
    _guard = threading.Lock()

    @classmethod
    def reset(cls):
        cls.built = []
        cls.fail_next_build = False
        cls.build_seconds = 0.0
        cls.search_seconds = 0.0
        cls.active = 0
        cls.max_active = 0

    def __init__(self, vectordb_path, embedding_model_name):
        time.sleep(type(self).build_seconds)
        if type(self).fail_next_build:
            type(self).fail_next_build = False
            raise FileNotFoundError("index.faiss")
        type(self).built.append((str(vectordb_path), embedding_model_name))

    def get_retriever(self, search_type="similarity", search_kwargs=None):
        return self

    def invoke(self, query):
        cls = type(self)
        with cls._guard:
            cls.active += 1
            cls.max_active = max(cls.max_active, cls.active)
        time.sleep(cls.search_seconds)
        with cls._guard:
            cls.active -= 1
        return [Document(page_content=f"Wiki text for {query}", metadata={"source": "a.pdf", "page": 1})]


@pytest.fixture(autouse=True)
def counting_retriever(monkeypatch):
    CountingRetriever.reset()
    retrieve_module.clear_retriever_cache()
    monkeypatch.setattr(retrieve_module, "DocumentRetriever", CountingRetriever)
    yield CountingRetriever
    retrieve_module.clear_retriever_cache()          # nothing built here may leak into other tests


def ask(question="What is steel casting?"):
    return retrieve_module.retrieve_context({"user_query": question})


def run_in_threads(count, target):
    threads = [threading.Thread(target=target) for _ in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)


def test_one_retriever_is_built_and_reused_for_every_question(counting_retriever):
    for question in ("first", "second", "third"):
        assert ask(question)["retrieved_docs"]

    assert len(counting_retriever.built) == 1


def test_each_question_still_searches_with_its_own_text(counting_retriever):
    assert "alloys" in ask("alloys")["retrieved_docs"][0].page_content
    assert "casting" in ask("casting")["retrieved_docs"][0].page_content      # not the first question's result


def test_a_different_model_or_index_gets_its_own_retriever(monkeypatch, counting_retriever):
    first_model = retrieve_module.settings.embedding_model
    ask()
    monkeypatch.setattr(retrieve_module.settings, "embedding_model", "some-other-model")
    ask()
    monkeypatch.setattr(type(retrieve_module.settings), "get_vectordb_path", lambda self: "/some/other/index")
    ask()

    assert [model for _, model in counting_retriever.built] == [first_model, "some-other-model", "some-other-model"]
    assert counting_retriever.built[-1][0] == "/some/other/index"


def test_concurrent_first_questions_build_only_one_retriever(counting_retriever):
    counting_retriever.build_seconds = 0.2          # long enough for every thread to arrive while it is building
    results = []
    run_in_threads(8, lambda: results.append(ask()))

    assert len(results) == 8 and all(result["retrieved_docs"] for result in results)
    assert len(counting_retriever.built) == 1


def test_searches_take_turns(counting_retriever):
    ask()                                            # build first, so the threads below only search
    counting_retriever.search_seconds = 0.05
    results = []
    run_in_threads(8, lambda: results.append(ask()))

    assert len(results) == 8
    assert counting_retriever.max_active == 1


def test_a_failed_build_is_not_cached(counting_retriever):
    counting_retriever.fail_next_build = True

    first = ask()
    assert first["retrieved_docs"] == [] and "Retrieval failed" in first["error"]      # as before: an empty result, flagged

    second = ask()                                   # the next question tries again, and works
    assert second["retrieved_docs"]
    assert len(counting_retriever.built) == 1


def test_clearing_the_cache_forces_a_rebuild(counting_retriever):
    ask()
    retrieve_module.clear_retriever_cache()
    ask()

    assert len(counting_retriever.built) == 2
