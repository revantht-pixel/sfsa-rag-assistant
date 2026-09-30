"""Tests for wiki_auth.py. Each test guards one of the properties in its docstring."""

import logging

import httpx
import pytest

from sfsa_rag_assistant.wiki_auth import (
    Identity,
    WikiSessionVerifier,
    WikiUnavailable,
    filter_wiki_cookies,
)

PREFIX = "wiki_db"
URL = "https://wiki.example.org/api.php"
COOKIE = "wiki_db_session=abc123; wiki_dbUserID=42"

LOGGED_IN = {"batchcomplete": "", "query": {"userinfo": {"id": 42, "name": "Alice Member", "groups": ["*", "user"]}}}
# What SFSA's private wiki actually returns to a logged-out caller (checked live).
DENIED = {"error": {"code": "readapidenied", "info": "You need read permission to use this module."}}


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def make(handler, **kwargs):
    """A verifier wired to a fake wiki; returns (verifier, list of requests it received)."""
    calls = []

    def wrapped(request):
        calls.append(request)
        return handler(request)

    client = httpx.Client(transport=httpx.MockTransport(wrapped))
    return WikiSessionVerifier(URL, PREFIX, client=client, **kwargs), calls


def json_reply(payload, status=200):
    return lambda request: httpx.Response(status, json=payload)


# ── cookie filtering: only the wiki's own cookies are ever forwarded ─────────
def test_only_wiki_cookies_are_kept_and_normalised():
    raw = "_ga=GA1.2.3; wiki_db_session=abc123; other=1; wiki_dbUserID=42"
    assert filter_wiki_cookies(raw, PREFIX) == "wiki_dbUserID=42; wiki_db_session=abc123"


@pytest.mark.parametrize("raw", [None, "", "_ga=1; x=2", "nocookies"])
def test_nothing_to_forward_means_none(raw):
    assert filter_wiki_cookies(raw, PREFIX) is None


@pytest.mark.parametrize(
    "raw",
    [
        "wiki_db_session=ab cd",            # whitespace inside a value
        "wiki_db_session=ab,cd",            # comma
        'wiki_db_session="quoted"',         # quotes
        "wiki_db_session=ab\r\nX-Evil: 1",  # header injection attempt
        "wiki_db_session=" + "a" * 9000,    # oversized
    ],
)
def test_malformed_or_oversized_wiki_cookie_is_denied_not_guessed(raw):
    assert filter_wiki_cookies(raw, PREFIX) is None


def test_one_malformed_wiki_cookie_poisons_the_whole_header():
    # A tampered or corrupted wiki cookie is a denial for the whole request; we
    # do not quietly drop it and forward whatever is left.
    assert filter_wiki_cookies("wiki_db_session=ab cd; wiki_dbUserID=42", PREFIX) is None
    assert filter_wiki_cookies("wiki_dbUserID=42; wiki_db_session=a,b", PREFIX) is None


# ── what is sent to the wiki ─────────────────────────────────────────────────
def test_verifies_member_and_forwards_only_wiki_cookies():
    def handler(request):
        assert request.url.host == "wiki.example.org"
        assert request.headers["cookie"] == "wiki_dbUserID=42; wiki_db_session=abc123"  # no _ga
        assert request.url.params["action"] == "query"
        assert request.url.params["meta"] == "userinfo"
        return httpx.Response(200, json=LOGGED_IN)

    verifier, calls = make(handler)
    identity = verifier.verify("_ga=1; wiki_db_session=abc123; wiki_dbUserID=42")
    assert identity == Identity(user_id=42, name="Alice Member")
    assert len(calls) == 1


def test_no_wiki_cookie_means_no_call_to_the_wiki():
    verifier, calls = make(json_reply(LOGGED_IN))
    assert verifier.verify("_ga=1; something=else") is None
    assert calls == []


# ── deny by default ──────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "payload",
    [
        DENIED,                                                              # private wiki, logged out
        {"query": {"userinfo": {"id": 0, "name": "203.0.113.5", "anon": ""}}},  # public wiki, anonymous
        {"query": {"userinfo": {"id": 5, "name": "x", "anon": ""}}},
        {"query": {"userinfo": {"name": "NoId"}}},
        {"query": {"userinfo": {"id": "42", "name": "StringId"}}},
        {"query": {"userinfo": {"id": True, "name": "BoolId"}}},
        {"query": {"userinfo": {"id": -3, "name": "Negative"}}},
        {"query": {"userinfo": {"id": 42, "name": ""}}},
        {"query": {"userinfo": {"id": 42}}},
        {"query": {"userinfo": "nope"}},
        {"query": []},
        {},
        [],
        "string",
        None,
    ],
)
def test_anything_but_a_clear_member_is_denied(payload):
    verifier, _ = make(json_reply(payload))
    assert verifier.verify(COOKIE) is None


def test_non_json_reply_is_denied():
    verifier, _ = make(lambda request: httpx.Response(200, text="<html>not json</html>"))
    assert verifier.verify(COOKIE) is None


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_client_error_statuses_are_denials(status):
    verifier, _ = make(lambda request: httpx.Response(status, json=LOGGED_IN))
    assert verifier.verify(COOKIE) is None


# ── an unreachable wiki is an outage, not a denial, and is never cached ──────
@pytest.mark.parametrize("status", [301, 302, 500, 502, 503])
def test_redirects_and_server_errors_are_outages(status):
    verifier, calls = make(lambda request: httpx.Response(status, headers={"location": "https://evil.example/"}))
    with pytest.raises(WikiUnavailable):
        verifier.verify(COOKIE)
    assert len(calls) == 1  # the redirect was not followed


def test_network_failure_is_an_outage_and_does_not_leak_the_cookie():
    def handler(request):
        raise httpx.ConnectTimeout("timed out", request=request)

    verifier, _ = make(handler)
    with pytest.raises(WikiUnavailable) as caught:
        verifier.verify(COOKIE)
    assert "abc123" not in str(caught.value) and "abc123" not in repr(caught.value)


def test_outages_are_not_cached():
    responses = [httpx.Response(503), httpx.Response(200, json=LOGGED_IN)]
    verifier, calls = make(lambda request: responses.pop(0))
    with pytest.raises(WikiUnavailable):
        verifier.verify(COOKIE)
    assert verifier.verify(COOKIE) == Identity(42, "Alice Member")
    assert len(calls) == 2


# ── caching: brief, bounded, and shorter for denials ─────────────────────────
def test_successful_check_is_cached_then_expires():
    clock = FakeClock()
    verifier, calls = make(json_reply(LOGGED_IN), cache_ttl=60, clock=clock)
    verifier.verify(COOKIE)
    verifier.verify(COOKIE)
    assert len(calls) == 1
    clock.advance(61)
    verifier.verify(COOKIE)
    assert len(calls) == 2


def test_denials_are_remembered_only_briefly():
    clock = FakeClock()
    verifier, calls = make(json_reply(DENIED), negative_ttl=10, clock=clock)
    verifier.verify(COOKIE)
    verifier.verify(COOKIE)
    assert len(calls) == 1
    clock.advance(11)
    verifier.verify(COOKIE)
    assert len(calls) == 2


def test_cache_is_bounded():
    verifier, calls = make(json_reply(LOGGED_IN), max_entries=2)
    for n in range(3):
        verifier.verify(f"wiki_db_session=s{n}")
    assert len(calls) == 3
    verifier.verify("wiki_db_session=s0")  # oldest was evicted, so the wiki is asked again
    assert len(calls) == 4


def test_cache_is_keyed_by_hash_not_by_the_cookie():
    verifier, _ = make(json_reply(LOGGED_IN))
    verifier.verify(COOKIE)
    assert all("abc123" not in key for key in verifier._cache)


# ── configuration safety ─────────────────────────────────────────────────────
@pytest.mark.parametrize("url", ["http://wiki.example.org/api.php", "ftp://wiki.example.org/api.php", "wiki.example.org/api.php"])
def test_refuses_to_send_cookies_over_anything_but_https(url):
    with pytest.raises(ValueError):
        WikiSessionVerifier(url, PREFIX)


def test_plain_http_is_allowed_only_for_loopback():
    WikiSessionVerifier("http://127.0.0.1:9999/api.php", PREFIX)
    WikiSessionVerifier("http://localhost:9999/api.php", PREFIX)


def test_default_client_never_follows_redirects_or_uses_env_proxies():
    verifier = WikiSessionVerifier(URL, PREFIX)
    assert verifier._client.follow_redirects is False
    assert verifier._client.trust_env is False


# ── cookies are credentials: never logged ────────────────────────────────────
def test_cookie_values_never_reach_the_logs(caplog):
    caplog.set_level(logging.DEBUG)
    secret = "TOPSECRETSESSIONVALUE"
    raw = f"wiki_db_session={secret}"
    for reply in (json_reply(LOGGED_IN), json_reply(DENIED), lambda request: httpx.Response(503)):
        verifier, _ = make(reply)
        try:
            verifier.verify(raw)
        except WikiUnavailable:
            pass
    assert secret not in caplog.text
