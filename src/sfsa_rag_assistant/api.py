# src/sfsa_rag_assistant/api.py
"""
Chat API for the SFSA RAG assistant (production-shaped; not yet streaming).

Trust model
-----------
* Authentication is the member's own MediaWiki login (wiki_auth.py). Every
  data endpoint denies by default; only /health, the widget page and its
  static assets are open.
* The chat UI is served from this same origin and embedded in the wiki in an
  iframe, so its calls are same-origin. CORS is deliberately NOT enabled:
  browsers therefore refuse to let any other page (for example scripts on a
  wiki page) read these responses. Unsafe requests must also carry an
  allow-listed Origin and, in browsers that send it, ``Sec-Fetch-Site:
  same-origin``. That stops other pages under sfsa.org -- whose requests the
  browser *will* attach the wiki login cookie to -- from driving the assistant
  on a member's behalf.
* Cookies are credentials: never logged, never echoed.
* Question and answer text is never logged; only who, when and how long.
"""

from __future__ import annotations

import logging
import re
import secrets
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ValidationError

from .app_config import Settings, settings as default_settings
from .wiki_auth import Identity, WikiSessionVerifier, WikiUnavailable
from .wiki_utils import construct_wiki_url

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent / "web"
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


# ─── Models ──────────────────────────────────────────────────────────────────
class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str
    conversation_history: List[Turn] = []


class Source(BaseModel):
    label: str
    url: Optional[str] = None
    kind: Literal["wiki", "web"] = "wiki"


class ChatResponse(BaseModel):
    response: str
    sources: List[Source]


# ─── Helpers ─────────────────────────────────────────────────────────────────
class InflightLimiter:
    """One question at a time per member, and a cap on simultaneous questions."""

    def __init__(self, global_max: int) -> None:
        self._global_max = global_max
        self._lock = threading.Lock()
        self._active: set = set()

    def acquire(self, user_id: int) -> Optional[str]:
        """Return None on success, else why it was refused: 'user' or 'server'."""
        with self._lock:
            if user_id in self._active:
                return "user"
            if len(self._active) >= self._global_max:
                return "server"
            self._active.add(user_id)
            return None

    def release(self, user_id: int) -> None:
        with self._lock:
            self._active.discard(user_id)


def _public_sources(raw_sources: Optional[List[Dict[str, Any]]]) -> List[Source]:
    """
    Convert workflow sources into what the widget may show.

    The index stores 0-based page numbers (PyMuPDF), but PDF viewers'
    ``#page=`` anchors and readers count from 1, so both the label and the link
    use ``page + 1``. (The inherited wiki_url field does not, and would land
    one page early.)
    """
    out: List[Source] = []
    for s in raw_sources or []:
        kind = s.get("source_type")
        if kind == "vector_db":
            filename = str(s.get("source", "")).replace("\\", "/").rsplit("/", 1)[-1] or "document"
            page = s.get("page")
            has_page = isinstance(page, int) and not isinstance(page, bool) and page >= 0
            label = f"{filename} (p.{page + 1})" if has_page else filename
            url = construct_wiki_url(filename, page + 1 if has_page else None)
            out.append(Source(label=label, url=url, kind="wiki"))
        elif kind == "web_search":
            out.append(Source(label=str(s.get("title") or "Web result"), url=s.get("url"), kind="web"))
    return out


def _configure_logging(package: str = "sfsa_rag_assistant", root: Optional[logging.Logger] = None) -> None:
    """
    Make this package's INFO logs (the audit trail: who asked, when, how long)
    actually appear when the API runs under uvicorn.

    uvicorn configures only its own loggers, so without this the audit lines
    are silently dropped. It does nothing if the host process (a test run, the
    CLI) has already set logging up.
    """
    root = root or logging.getLogger()
    pkg = logging.getLogger(package)
    if root.handlers or pkg.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    pkg.addHandler(handler)
    pkg.setLevel(logging.INFO)


def _safe_name(name: str) -> str:
    """A wiki username made safe to put in a log line."""
    return re.sub(r"[^\w .@'-]", "?", name)[:64]


def _reject(status: int, detail: str, **headers: str) -> HTTPException:
    return HTTPException(status_code=status, detail=detail, headers=headers or None)


# ─── App factory ─────────────────────────────────────────────────────────────
def create_app(
    cfg: Optional[Settings] = None,
    verifier: Optional[WikiSessionVerifier] = None,
    workflow: Optional[Callable[..., Dict[str, Any]]] = None,
) -> FastAPI:
    """
    Build the API. ``verifier`` and ``workflow`` can be injected for tests;
    by default they are the real wiki verifier and the LangGraph workflow.
    """
    cfg = cfg or default_settings
    allowed_origins = set(cfg.get_allowed_origins())
    frame_ancestors = cfg.get_frame_ancestors()
    citation_hosts = cfg.get_citation_hosts()
    verifier = verifier or WikiSessionVerifier(
        api_url=cfg.wiki_api_url,
        cookie_prefix=cfg.wiki_cookie_prefix,
        timeout=cfg.wiki_request_timeout,
        tls_server_name=cfg.wiki_tls_server_name,
        cache_ttl=cfg.session_cache_ttl,
        negative_ttl=cfg.session_negative_ttl,
    )
    limiter = InflightLimiter(cfg.max_inflight)

    def run_workflow(**kwargs: Any) -> Dict[str, Any]:
        if workflow is not None:
            return workflow(**kwargs)
        from .graph import run_workflow as real_run_workflow  # heavy import, on first use
        return real_run_workflow(**kwargs)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        _configure_logging()
        yield

    # No interactive docs / OpenAPI: less surface, nothing to explore anonymously.
    app = FastAPI(title="SFSA RAG Assistant API", docs_url=None, redoc_url=None, openapi_url=None,
                  lifespan=lifespan)

    @app.middleware("http")
    async def harden(request: Request, call_next):
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > cfg.max_body_bytes:
            response: Response = JSONResponse({"detail": "Request too large."}, status_code=413)
        else:
            response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers.setdefault("Cache-Control", "no-store")   # answers are members-only content
        if request.url.scheme == "https":
            response.headers.setdefault("Strict-Transport-Security", "max-age=31536000")
        return response

    def authorize(request: Request) -> Identity:
        """Cheap request checks first, then who the member is. Deny by default."""
        site = request.headers.get("sec-fetch-site")
        unsafe = request.method not in ("GET", "HEAD")
        if unsafe:
            origin = request.headers.get("origin")
            if site not in (None, "same-origin") or origin not in allowed_origins:
                logger.warning("blocked %s %s origin=%r site=%r", request.method, request.url.path,
                               (origin or "")[:100], site)
                raise _reject(403, "Forbidden.")
            content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
            if content_type != "application/json":
                raise _reject(415, "Unsupported media type.")
        elif site not in (None, "same-origin", "none"):
            logger.warning("blocked %s %s site=%r", request.method, request.url.path, site)
            raise _reject(403, "Forbidden.")

        cookie = request.headers.get("cookie")
        if not cookie:
            raise _reject(401, "Please log in to the SFSA wiki first.")
        try:
            identity = verifier.verify(cookie)
        except WikiUnavailable as exc:
            logger.error("wiki session check unavailable: %s", exc)
            raise _reject(503, "The login service is unavailable right now. Please try again shortly.")
        if identity is None:
            raise _reject(401, "Please log in to the SFSA wiki first.")
        return identity

    # ── Open endpoints ──────────────────────────────────────────────────────
    @app.api_route("/health", methods=["GET", "HEAD"])
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    def _asset(filename: str, media_type: str, cache_control: str, csp: bool = False) -> FileResponse:
        headers = {"Cache-Control": cache_control}
        if csp:
            ancestors = " ".join(frame_ancestors) or "'none'"
            headers["Content-Security-Policy"] = (
                "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
                f"connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors {ancestors}"
            )
        return FileResponse(WEB_DIR / filename, media_type=media_type, headers=headers)

    @app.api_route("/widget", methods=["GET", "HEAD"])
    def widget_page() -> FileResponse:
        return _asset("widget.html", "text/html; charset=utf-8", "no-cache", csp=True)

    @app.api_route("/widget.js", methods=["GET", "HEAD"])
    def widget_js() -> FileResponse:
        return _asset("widget.js", "text/javascript; charset=utf-8", "public, max-age=300")

    @app.api_route("/widget.css", methods=["GET", "HEAD"])
    def widget_css() -> FileResponse:
        return _asset("widget.css", "text/css; charset=utf-8", "public, max-age=300")

    # ── Members-only endpoints ──────────────────────────────────────────────
    @app.get("/session")
    async def session(request: Request) -> Dict[str, Any]:
        identity = await run_in_threadpool(authorize, request)
        return {
            "authenticated": True,
            "user": identity.name,
            "citation_hosts": citation_hosts,
            "max_question_chars": cfg.max_question_chars,
        }

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: Request, response: Response) -> ChatResponse:
        request_id = secrets.token_hex(6)
        response.headers["X-Request-ID"] = request_id
        started = time.monotonic()
        identity = await run_in_threadpool(authorize, request)
        who = _safe_name(identity.name)

        try:
            payload = ChatRequest.model_validate_json(await request.body())
            question = _CONTROL_CHARS.sub("", payload.question).strip()
            if (
                not question
                or len(question) > cfg.max_question_chars
                or len(payload.conversation_history) > cfg.max_history_turns
                or any(len(t.content) > cfg.max_history_chars for t in payload.conversation_history)
            ):
                raise ValueError("out of bounds")
        except (ValidationError, ValueError):
            raise _reject(400, "That question couldn't be processed.")

        refused = limiter.acquire(identity.user_id)
        if refused == "user":
            raise _reject(429, "You already have a question in progress. Please wait for it to finish.",
                          **{"Retry-After": "15"})
        if refused == "server":
            raise _reject(503, "The assistant is busy right now. Please try again in a moment.",
                          **{"Retry-After": "15"})
        try:
            result = await run_in_threadpool(
                run_workflow,
                user_query=question,
                conversation_history=[t.model_dump() for t in payload.conversation_history],
                max_validation_attempts=cfg.max_validation_attempts,
            )
        except Exception as exc:  # noqa: BLE001 -- never leak internals to the client
            logger.error("chat failed rid=%s user=%s error=%s", request_id, who, type(exc).__name__)
            raise _reject(500, f"Something went wrong answering that. Please try again. (ref {request_id})")
        finally:
            limiter.release(identity.user_id)

        logger.info("chat ok rid=%s user=%s ms=%d q_chars=%d", request_id, who,
                    int((time.monotonic() - started) * 1000), len(question))
        return ChatResponse(response=str(result.get("response", "")), sources=_public_sources(result.get("sources")))

    return app


# The instance uvicorn serves: `uvicorn sfsa_rag_assistant.api:app`
app = create_app()
