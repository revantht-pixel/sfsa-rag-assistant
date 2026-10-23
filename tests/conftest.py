import pytest

import pipeline_fakes


@pytest.fixture
def fake_pipeline(monkeypatch):
    """The real graph with a fake language model and a fake search index (see pipeline_fakes.py)."""
    pipeline_fakes.install(monkeypatch)
