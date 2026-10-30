"""FastAPI application entry point for HADOTA."""

from __future__ import annotations

from pathlib import Path

import asyncio
import json
import logging
import threading
import secrets

from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from urllib.parse import urlsplit
from pydantic import ValidationError

from app.cli import create_conversation_service_with_runtime
from app.config import load_settings, validate_public_mode_settings
from app.runtime.exceptions import RuntimeErrorBase
from app.runtime.ollama_runtime import OllamaRuntime
from app.web.schemas import (
    ConversationResponse,
    CreateConversationRequest,
    MessageResponse,
    SendMessageRequest,
    SessionResponse,
)
from app.web.sessions import (
    ConversationBusyError,
    ConversationLimitError,
    SessionLimitError,
    SessionManager,
    SessionNotFoundError,
    SessionRateLimitError,
)

ServiceFactory = Callable[..., tuple[Any, Any]]

SESSION_COOKIE = "hadota_session"
SESSION_MAX_AGE = 14400
SESSION_CLEANUP_INTERVAL_SECONDS = 60
DEFAULT_MAX_REQUEST_BODY_BYTES = 65536
logger = logging.getLogger("hadota.web")


class RequestBodyLimitMiddleware:
    """Bound API request bodies before FastAPI parses or buffers them."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or not scope.get("path", "").startswith("/api/"):
            await self.app(scope, receive, send)
            return

        # Starlette places the owning application in scope['app']. Prefer its
        # live state so tests and deployments can override the body cap; retain
        # the lifespan scope state as a fallback for ASGI wrappers.
        application = scope.get("app")
        application_state = getattr(application, "state", None)
        settings = getattr(application_state, "settings", None)
        if settings is None:
            scope_state = scope.get("state") or {}
            settings = (
                scope_state.get("settings")
                if isinstance(scope_state, dict)
                else None
            )
        limit = getattr(
            settings,
            "max_request_body_bytes",
            DEFAULT_MAX_REQUEST_BODY_BYTES,
        )
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                if int(content_length) > limit:
                    await self._send_too_large(send, scope.get("method", "unknown"))
                    return
            except ValueError:
                await self._send_too_large(send, scope.get("method", "unknown"))
                return

        # Buffer only the bounded API body, then replay the received ASGI events
        # to FastAPI. This also covers chunked requests without Content-Length.
        events = []
        total = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            events.append(message)
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > limit:
                    await self._send_too_large(send, scope.get("method", "unknown"))
                    return
                if not message.get("more_body", False):
                    break
            else:
                break

        async def replay_receive():
            if events:
                return events.pop(0)
            return {"type": "http.request", "body": b"", "more_body": False}

        await self.app(scope, replay_receive, send)

    @staticmethod
    async def _send_too_large(send, method="unknown"):
        body = json.dumps({"error": "Request body too large."}).encode("utf-8")
        request_id = secrets.token_hex(12)
        logger.warning(
            "http_request_rejected request_id=%s method=%s status=413",
            request_id,
            method,
        )
        await send({
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json; charset=utf-8"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"x-content-type-options", b"nosniff"),
                (b"referrer-policy", b"no-referrer"),
                (b"x-request-id", request_id.encode("ascii")),
            ],
        })
        await send({"type": "http.response.body", "body": body})


async def run_sync_operation_to_completion(function, *args):
    """Run sync inference off-loop and keep ownership until its worker exits.

    Cancelling an asyncio task does not stop a worker thread. Shielding and
    joining that thread before propagating cancellation prevents the route from
    releasing conversation/global-generation locks while inference is still
    mutating the conversation.
    """
    worker = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        try:
            await asyncio.shield(worker)
        except BaseException:
            # Preserve the caller's cancellation; worker errors are not relevant
            # to the already-cancelled request, but cleanup waits for completion.
            pass
        raise


def create_app(
    service_factory: ServiceFactory | None = None,
    *,
    conversation_service_factory: Callable[[], Any] | None = None,
) -> FastAPI:
    """Create the HADOTA API application."""

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        runtime = None
        session_manager = None
        cleanup_task = None
        stop_cleanup = asyncio.Event()

        async def cleanup_expired_sessions():
            while not stop_cleanup.is_set():
                try:
                    await asyncio.wait_for(
                        stop_cleanup.wait(),
                        timeout=SESSION_CLEANUP_INTERVAL_SECONDS,
                    )
                except asyncio.TimeoutError:
                    await asyncio.to_thread(
                        session_manager.cleanup_expired
                    )

        try:
            if service_factory is None:
                settings = load_settings()
                validate_public_mode_settings(settings)
                application.state.settings = settings
            else:
                settings = getattr(application.state, "settings", None)

            session_manager = SessionManager(
                max_sessions=getattr(settings, "max_active_sessions", 25),
                max_conversations_per_session=getattr(
                    settings, "max_conversations_per_session", 5
                ),
                idle_ttl_seconds=getattr(settings, "session_idle_ttl_seconds", 900),
                absolute_ttl_seconds=getattr(
                    settings, "session_absolute_ttl_seconds", SESSION_MAX_AGE
                ),
            )
            application.state.session_manager = session_manager
            application.state.generation_semaphore = threading.BoundedSemaphore(
                getattr(settings, "max_active_generations", 1)
            )
            cleanup_task = asyncio.create_task(cleanup_expired_sessions())

            if service_factory is None:
                runtime = OllamaRuntime(settings=settings)
                application.state.runtime = runtime
                application.state.create_conversation_service = (
                    lambda: create_conversation_service_with_runtime(runtime)
                )
            else:
                runtime, service = service_factory()
                application.state.runtime = runtime
                application.state.conversation_service = service
                # Production already creates a fresh service per conversation.
                # Tests/integrations may inject that factory explicitly so each
                # conversation gets its own mutable history rather than sharing
                # the single service returned by the setup factory.
                application.state.create_conversation_service = (
                    conversation_service_factory
                    if conversation_service_factory is not None
                    else lambda: service
                )

            yield
        finally:
            stop_cleanup.set()

            if cleanup_task is not None:
                await cleanup_task

            if session_manager is not None:
                await asyncio.to_thread(session_manager.close_all)

            application.state.runtime = None
            application.state.session_manager = None
            application.state.generation_semaphore = None

            for name in (
                "settings",
                "create_conversation_service",
                "conversation_service",
            ):
                if hasattr(application.state, name):
                    setattr(application.state, name, None)

            if runtime is not None:
                runtime.close()

    application = FastAPI(
        title="HADOTA - Local AI Assistant",
        version="0.2.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    application.add_middleware(RequestBodyLimitMiddleware)

    @application.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        """Attach a server-generated correlation ID without trusting input IDs."""
        request_id = secrets.token_hex(12)
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        if response.status_code >= 400:
            # Log operational metadata only: never log prompts, answers, cookie
            # values, session IDs, request bodies, or full conversation paths.
            logger.warning(
                "http_request_failed request_id=%s method=%s status=%s",
                request_id,
                request.method,
                response.status_code,
            )
        return response

    @application.middleware("http")
    async def validate_request_origin(request: Request, call_next):
        """Reject disallowed browser origins on state-changing API requests."""

        if (
            request.url.path.startswith("/api/")
            and request.method in {"POST", "PUT", "PATCH", "DELETE"}
        ):
            origin = request.headers.get("origin")
            settings = getattr(request.app.state, "settings", None)
            public_mode = getattr(settings, "public_mode", False)

            if public_mode and not origin:
                return JSONResponse(
                    status_code=403,
                    content={"error": "Origin header required."},
                )

            if origin:
                configured_origins = getattr(settings, "allowed_origins", "")
                def parse_origin(value: str):
                    """Return a normalized HTTP(S) origin, or None if invalid."""
                    try:
                        parsed = urlsplit(value)

                        if (
                            parsed.scheme.lower() not in {"http", "https"}
                            or not parsed.hostname
                            or parsed.username is not None
                            or parsed.password is not None
                            or parsed.path not in {"", "/"}
                            or parsed.query
                            or parsed.fragment
                        ):
                            return None

                        port = parsed.port
                        scheme = parsed.scheme.lower()
                        hostname = parsed.hostname.lower()

                        if port is None:
                            port = 443 if scheme == "https" else 80

                        return scheme, hostname, port
                    except ValueError:
                        return None

                request_origin = parse_origin(origin)
                allowed_origins = {
                    parsed
                    for item in configured_origins.split(",")
                    if item.strip()
                    for parsed in [parse_origin(item.strip())]
                    if parsed is not None
                }
                # In local mode, permit the app's own origin when no
                # explicit allowlist is configured. Foreign origins remain blocked.
                if not public_mode and not configured_origins.strip():
                    same_origin = parse_origin(
                        f"{request.url.scheme}://{request.url.netloc}"
                    )
                    if same_origin is not None:
                        allowed_origins.add(same_origin)


                if (
                    request_origin is None
                    or request_origin not in allowed_origins
                ):
                    return JSONResponse(
                        status_code=403,
                        content={"error": "Origin not allowed."},
                    )

        return await call_next(request)

    def get_session_id(request: Request) -> str:
        session_id = request.cookies.get(SESSION_COOKIE)

        if not session_id:
            from fastapi import HTTPException

            raise HTTPException(
                status_code=401,
                detail="A valid session is required.",
            )

        return session_id

    def enforce_generation_rate_limit(request: Request, session_id: str) -> None:
        settings = getattr(request.app.state, "settings", None)
        manager: SessionManager = request.app.state.session_manager
        manager.enforce_request_rate_limit(
            session_id,
            max_requests=getattr(settings, "session_rate_limit_requests", 10),
            window_seconds=getattr(settings, "session_rate_limit_window_seconds", 60),
        )
        manager.enforce_global_generation_rate_limit(
            max_requests=getattr(
                settings, "global_generation_rate_limit_requests", 20
            ),
            window_seconds=getattr(
                settings, "global_generation_rate_limit_window_seconds", 60
            ),
        )

    def public_error(status_code: int, message: str) -> JSONResponse:
        return JSONResponse(
            status_code=status_code,
            content={"error": message},
        )

    @application.exception_handler(RequestValidationError)
    async def handle_request_validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        """Return a stable validation error without echoing submitted input."""
        return public_error(422, "Invalid request payload.")

    @application.exception_handler(SessionNotFoundError)
    async def handle_session_not_found(
        request: Request,
        exc: SessionNotFoundError,
    ) -> JSONResponse:
        return public_error(404, "Session or conversation not found.")

    @application.exception_handler(SessionLimitError)
    async def handle_session_limit(
        request: Request,
        exc: SessionLimitError,
    ) -> JSONResponse:
        return public_error(429, "The session limit has been reached.")

    @application.exception_handler(ConversationLimitError)
    async def handle_conversation_limit(
        request: Request,
        exc: ConversationLimitError,
    ) -> JSONResponse:
        return public_error(429, "The conversation limit has been reached.")

    @application.exception_handler(ConversationBusyError)
    async def handle_conversation_busy(
        request: Request,
        exc: ConversationBusyError,
    ) -> JSONResponse:
        return public_error(409, "This conversation is busy. Please try again shortly.")

    @application.exception_handler(SessionRateLimitError)
    async def handle_session_rate_limit(
        request: Request,
        exc: SessionRateLimitError,
    ) -> JSONResponse:
        return public_error(429, "Too many requests. Please wait before trying again.")

    @application.get("/api/health")
    async def health() -> JSONResponse:
        """Liveness: the web process is responding, regardless of Ollama state."""
        return JSONResponse(status_code=200, content={"status": "ok"})

    @application.get("/api/ready")
    async def ready(request: Request) -> JSONResponse:
        """Readiness: Ollama is reachable and the configured model is installed."""
        runtime = getattr(request.app.state, "runtime", None)
        if runtime is None:
            return JSONResponse(
                status_code=503,
                content={"status": "unavailable", "reason": "runtime_unavailable"},
            )

        settings = getattr(request.app.state, "settings", None)
        try:
            healthy = await asyncio.to_thread(runtime.health)
            if not healthy:
                raise RuntimeErrorBase("Runtime is unavailable.")
            model_name = getattr(settings, "model_name", None)
            if model_name:
                available_models = await asyncio.to_thread(runtime.list_models)
                if model_name not in available_models:
                    return JSONResponse(
                        status_code=503,
                        content={"status": "unavailable", "reason": "model_unavailable"},
                    )
        except Exception:
            return JSONResponse(
                status_code=503,
                content={"status": "unavailable", "reason": "runtime_unavailable"},
            )

        return JSONResponse(status_code=200, content={"status": "ready"})

    @application.post("/api/session", response_model=SessionResponse)
    async def create_session(request: Request, response: Response):
        """Create an anonymous session and set its opaque cookie."""

        manager: SessionManager = request.app.state.session_manager

        try:
            session = manager.create_session()
        except SessionLimitError:
            return public_error(429, "The session limit has been reached.")

        settings = getattr(request.app.state, "settings", None)
        response.set_cookie(
            key=SESSION_COOKIE,
            value=session.session_id,
            max_age=getattr(settings, "session_absolute_ttl_seconds", SESSION_MAX_AGE),
            httponly=True,
            secure=getattr(
                getattr(request.app.state, "settings", None),
                "cookie_secure",
                False,
            ),
            samesite="lax",
            path="/",
        )

        return SessionResponse()

    @application.post("/api/session/close")
    async def request_session_close(request: Request):
        """Best-effort page-close signal with grace so a refresh can resume."""
        manager: SessionManager = request.app.state.session_manager
        session_id = get_session_id(request)
        scheduled = manager.request_session_close(session_id)
        return {"status": "closing" if scheduled else "expired"}

    @application.delete("/api/session")
    async def delete_session(request: Request, response: Response):
        """Delete the current session and clear its cookie."""

        manager: SessionManager = request.app.state.session_manager
        session_id = request.cookies.get(SESSION_COOKIE)

        if session_id:
            await asyncio.to_thread(manager.delete_session, session_id)

        response.delete_cookie(
            key=SESSION_COOKIE,
            path="/",
            httponly=True,
            secure=getattr(
                getattr(request.app.state, "settings", None),
                "cookie_secure",
                False,
            ),
            samesite="lax",
        )

        return {"status": "ok"}

    @application.post(
        "/api/conversations",
        response_model=ConversationResponse,
        status_code=201,
    )
    async def create_conversation(
        payload: CreateConversationRequest,
        request: Request,
    ):
        """Create a conversation owned by the current session."""

        session_id = get_session_id(request)
        manager: SessionManager = request.app.state.session_manager

        entry = manager.create_conversation(
            session_id,
            request.app.state.create_conversation_service,
        )
        entry.title = payload.title

        return ConversationResponse(
            conversation_id=entry.conversation_id,
            title=payload.title,
        )

    @application.get("/api/conversations")
    async def list_conversations(request: Request):
        """List conversations belonging to the current session."""

        session_id = get_session_id(request)
        manager: SessionManager = request.app.state.session_manager
        session = manager.get_session(session_id)

        with session.lock:
            conversations = [
                {
                    "conversation_id": entry.conversation_id,
                    "title": getattr(entry, "title", None),
                }
                for entry in session.conversations.values()
                if not entry.closing
            ]

        return {"conversations": conversations}

    @application.get("/api/conversations/{conversation_id}")
    async def get_conversation(
        conversation_id: str,
        request: Request,
    ):
        """Return the currently available conversation history without waiting for generation."""

        session_id = get_session_id(request)
        manager: SessionManager = request.app.state.session_manager

        def read_conversation():
            # Retrieve the entry without acquiring its generation lock.
            entry = manager.get_conversation(session_id, conversation_id)

            # Read the currently available messages while generation may still
            # be in progress. The generation operation retains its own lock.
            messages = entry.service.conversation.get_messages(
                include_system=False,
            )

            return {
                "conversation_id": conversation_id,
                "title": getattr(entry, "title", None),
                "continuation_available": entry.service.continuation_available,
                "messages": [
                    {
                        "role": message.role,
                        "content": message.content,
                    }
                    for message in messages
                ],
            }

        return await asyncio.to_thread(read_conversation)

    @application.delete("/api/conversations/{conversation_id}")
    async def delete_conversation(
        conversation_id: str,
        request: Request,
    ):
        """Delete a conversation owned by the current session."""

        session_id = get_session_id(request)
        manager: SessionManager = request.app.state.session_manager

        try:
            deleted = await asyncio.to_thread(
                manager.delete_conversation,
                session_id,
                conversation_id,
                blocking=False,
            )
        except ConversationBusyError:
            return public_error(409, "This conversation is generating a response; try deleting it again later.")

        if not deleted:
            return public_error(404, "Conversation not found.")

        return {"status": "ok"}

    @application.post(
        "/api/conversations/{conversation_id}/continue",
        response_model=MessageResponse,
    )
    async def continue_message(
        conversation_id: str,
        request: Request,
    ):
        """Continue a response that stopped at the generation token limit."""

        session_id = get_session_id(request)
        manager: SessionManager = request.app.state.session_manager

        enforce_generation_rate_limit(request, session_id)
        semaphore = request.app.state.generation_semaphore
        if not semaphore.acquire(blocking=False):
            return public_error(429, "The assistant is busy. Please wait and try again.")
        try:
            with manager.conversation_operation(
                session_id,
                conversation_id,
                blocking=False,
            ) as entry:
                result = await run_sync_operation_to_completion(
                    entry.service.continue_message
                )

                if not result.done:
                    return public_error(
                        503,
                        "The assistant could not complete the continuation.",
                    )

                return MessageResponse(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=result.text,
                    done=True,
                    continuation_available=(
                        entry.service.continuation_available
                        if isinstance(entry.service.continuation_available, bool)
                        else False
                    ),
                )
        except RuntimeErrorBase:
            return public_error(
                503,
                "The assistant is temporarily unavailable.",
            )
        except ValueError:
            return public_error(
                409,
                "There is no incomplete response available to continue.",
            )
        finally:
            semaphore.release()

    @application.post(
        "/api/conversations/{conversation_id}/messages",
        response_model=MessageResponse,
    )
    async def send_message(
        conversation_id: str,
        payload: SendMessageRequest,
        request: Request,
    ):
        """Generate a complete assistant response."""

        session_id = get_session_id(request)
        manager: SessionManager = request.app.state.session_manager

        settings = getattr(request.app.state, "settings", None)
        if len(payload.content) > getattr(settings, "max_message_chars", 4000):
            return public_error(422, "The message exceeds the configured length limit.")

        enforce_generation_rate_limit(request, session_id)
        semaphore = request.app.state.generation_semaphore
        if not semaphore.acquire(blocking=False):
            return public_error(429, "The assistant is busy. Please wait and try again.")

        try:
            with manager.conversation_operation(
                session_id,
                conversation_id,
                blocking=False,
            ) as entry:
                result = await run_sync_operation_to_completion(
                    entry.service.send_message,
                    payload.content,
                )

                if not result.done:
                    return public_error(
                        503,
                        "The assistant could not complete the response.",
                    )

                # Derive a short title from the first user message; no extra model call.
                if not getattr(entry, "title", None):
                    cleaned = " ".join(payload.content.split()).strip()
                    if cleaned:
                        words = cleaned.split()
                        title = " ".join(words[:7])
                        if len(title) > 55:
                            shortened = title[:52].rsplit(" ", 1)[0]
                            title = (shortened or title[:52]) + "..."
                        entry.title = title

                return MessageResponse(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=result.text,
                    title=getattr(entry, "title", None),
                    done=True,
                    continuation_available=(
                        entry.service.continuation_available
                        if isinstance(entry.service.continuation_available, bool)
                        else False
                    ),
                )
        except RuntimeErrorBase:
            return public_error(
                503,
                "The assistant is temporarily unavailable.",
            )
        except ValueError:
            return public_error(422, "The message could not be processed.")
        finally:
            semaphore.release()

    frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
    application.mount(
        "/",
        StaticFiles(directory=str(frontend_dir), html=True),
        name="frontend",
    )

    return application


app = create_app()
