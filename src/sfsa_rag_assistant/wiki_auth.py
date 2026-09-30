# src/sfsa_rag_assistant/wiki_auth.py
"""
Server-side verification of a member's MediaWiki login.

The chat API keeps no passwords and no accounts of its own. SFSA's wiki sets
its login cookie for the whole ``sfsa.org`` domain, so a member's browser sends
it to ``rag.sfsa.org`` as well. This module forwards *only the wiki's own
cookies* to the wiki's API and asks "who is this?". Anything other than a
clear, well-formed "this is member N" answer is a denial.

Each rule below is a security property and has a test in tests/test_wiki_auth.py:

* Deny by default. Errors, malformed replies, anonymous sessions and the
  ``readapidenied`` error a private wiki gives logged-out callers all mean
  "not a member".
* Cookies are credentials. They are forwarded over HTTPS only (plain http is
  refused except on loopback), never followed across redirects, never logged,
  and the result cache is keyed by a hash rather than by the cookie itself.
* Only the wiki's own cookies are forwarded (unrelated cookies on the shared
  parent domain, e.g. analytics ones, are dropped).
* An unreachable wiki is reported as such (``WikiUnavailable``), not as a
  denial, and is never cached.
* A successful check is trusted for a short time only (``cache_ttl``), which
  is therefore also the longest a revoked login keeps working here.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}
MAX_COOKIE_HEADER = 8192

# RFC 6265: cookie-name is an HTTP token; cookie-value is a restricted octet
# set (no controls, whitespace, quotes, commas, semicolons or backslashes).
_COOKIE_NAME_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_COOKIE_VALUE_RE = re.compile(r"^[\x21\x23-\x2b\x2d-\x3a\x3c-\x5b\x5d-\x7e]*$")


@dataclass(frozen=True)
class Identity:
    """A verified wiki member."""

    user_id: int
    name: str


class WikiUnavailable(Exception):
    """The wiki could not give a usable answer (network error, 5xx, redirect)."""


def filter_wiki_cookies(raw_cookie_header: Optional[str], prefix: str) -> Optional[str]:
    """
    Reduce a browser ``Cookie`` header to just the wiki's own cookies.

    Returns a normalised ``name=value; ...`` string, or ``None`` when there is
    nothing to forward or a wiki cookie is malformed (which we treat as a
    denial rather than guessing).
    """
    if not raw_cookie_header or len(raw_cookie_header) > MAX_COOKIE_HEADER:
        return None
    kept = []
    for part in raw_cookie_header.split(";"):
        name, sep, value = part.strip().partition("=")
        if not sep or not name.startswith(prefix):
            continue
        if not _COOKIE_NAME_RE.match(name) or not _COOKIE_VALUE_RE.match(value):
            return None
        kept.append(f"{name}={value}")
    return "; ".join(sorted(kept)) or None


def _parse_userinfo(data: Any) -> Optional[Identity]:
    """Turn the wiki's reply into an Identity, or None for anything but a clear member."""
    if not isinstance(data, dict):
        return None
    query = data.get("query")
    info = query.get("userinfo") if isinstance(query, dict) else None
    if not isinstance(info, dict) or "anon" in info:
        return None
    user_id, name = info.get("id"), info.get("name")
    if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id <= 0:
        return None
    if not isinstance(name, str) or not name:
        return None
    return Identity(user_id=user_id, name=name)


class WikiSessionVerifier:
    """Asks the wiki's own API who a request's login cookie belongs to."""

    def __init__(
        self,
        api_url: str,
        cookie_prefix: str,
        *,
        timeout: float = 5.0,
        tls_server_name: Optional[str] = None,
        cache_ttl: int = 60,
        negative_ttl: int = 10,
        max_entries: int = 1024,
        client: Optional[httpx.Client] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        parsed = urlparse(api_url)
        secure = parsed.scheme == "https"
        loopback_http = parsed.scheme == "http" and parsed.hostname in _LOOPBACK_HOSTS
        if not (secure or loopback_http):
            raise ValueError(
                "wiki_api_url must be https:// (plain http is allowed only for "
                "loopback), because session cookies are sent to it."
            )
        self._api_url = api_url
        self._prefix = cookie_prefix
        self._tls_server_name = tls_server_name
        self._cache_ttl = cache_ttl
        self._negative_ttl = negative_ttl
        self._max_entries = max_entries
        self._clock = clock
        self._lock = threading.Lock()
        self._cache: "OrderedDict[str, tuple[float, Optional[Identity]]]" = OrderedDict()
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(timeout, connect=min(3.0, timeout)),
            follow_redirects=False,   # never carry the cookie to another host
            trust_env=False,          # never route it through an env-configured proxy
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )

    def verify(self, raw_cookie_header: Optional[str]) -> Optional[Identity]:
        """
        Return the member the cookie belongs to, or None if it isn't a valid
        member session. Raises WikiUnavailable if the wiki couldn't answer.
        """
        cookie = filter_wiki_cookies(raw_cookie_header, self._prefix)
        if cookie is None:
            return None

        key = hashlib.sha256(cookie.encode("utf-8")).hexdigest()
        now = self._clock()
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None and hit[0] > now:
                self._cache.move_to_end(key)
                return hit[1]

        identity = self._ask_wiki(cookie)   # may raise WikiUnavailable (not cached)

        ttl = self._cache_ttl if identity is not None else self._negative_ttl
        with self._lock:
            self._cache[key] = (now + ttl, identity)
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_entries:
                self._cache.popitem(last=False)
        return identity

    def _ask_wiki(self, cookie: str) -> Optional[Identity]:
        extensions = {"sni_hostname": self._tls_server_name} if self._tls_server_name else {}
        try:
            response = self._client.get(
                self._api_url,
                params={"action": "query", "meta": "userinfo", "uiprop": "groups", "format": "json"},
                headers={
                    "Cookie": cookie,
                    "Accept": "application/json",
                    "User-Agent": "sfsa-rag-assistant/session-verifier",
                },
                extensions=extensions,
            )
        except httpx.HTTPError as exc:
            # Log the exception class only: never the request (it carries the cookie).
            raise WikiUnavailable(type(exc).__name__) from None

        if 300 <= response.status_code < 400 or response.status_code >= 500:
            # A redirect usually means a mis-set wiki_api_url; surface it as an
            # outage rather than silently denying every member.
            raise WikiUnavailable(f"HTTP {response.status_code}")
        if response.status_code != 200:
            return None
        try:
            data = response.json()
        except ValueError:
            return None
        return _parse_userinfo(data)

    def close(self) -> None:
        self._client.close()
