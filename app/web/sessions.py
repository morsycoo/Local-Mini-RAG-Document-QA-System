"""In-memory anonymous session lifecycle management."""

from __future__ import annotations

import secrets
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator


class SessionError(Exception):
    """Base exception for anonymous session management."""


class SessionLimitError(SessionError):
    """Raised when the active-session limit has been reached."""


class SessionNotFoundError(SessionError):
    """Raised when a session does not exist or has expired."""


class ConversationLimitError(SessionError):
    """Raised when a session has reached its conversation limit."""


class ConversationBusyError(SessionError):
    """Raised when a conversation is already being modified or generated."""


class SessionRateLimitError(SessionError):
    """Raised when a session exceeds its request-rate allowance."""


@dataclass
class ConversationEntry:
    """A conversation and its associated service."""

    conversation_id: str
    service: Any
    lock: threading.Lock = field(default_factory=threading.Lock)
    closing: bool = False


@dataclass
class AnonymousSession:
    """An anonymous session stored only in process memory."""

    session_id: str
    created_at: float
    last_activity_at: float
    conversations: dict[str, ConversationEntry] = field(default_factory=dict)
    request_timestamps: deque[float] = field(default_factory=deque)
    close_requested_at: float | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)


class SessionManager:
    """Manage anonymous sessions with bounded lifetime and capacity."""

    def __init__(
        self,
        *,
        max_sessions: int = 25,
        max_conversations_per_session: int = 5,
        idle_ttl_seconds: float = 900.0,
        absolute_ttl_seconds: float = 14400.0,
        close_grace_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions must be at least 1.")
        if max_conversations_per_session < 1:
            raise ValueError(
                "max_conversations_per_session must be at least 1."
            )
        if idle_ttl_seconds <= 0:
            raise ValueError("idle_ttl_seconds must be positive.")
        if absolute_ttl_seconds <= 0:
            raise ValueError("absolute_ttl_seconds must be positive.")
        if close_grace_seconds <= 0:
            raise ValueError("close_grace_seconds must be positive.")

        self.max_sessions = max_sessions
        self.max_conversations_per_session = max_conversations_per_session
        self.idle_ttl_seconds = idle_ttl_seconds
        self.absolute_ttl_seconds = absolute_ttl_seconds
        self.close_grace_seconds = close_grace_seconds
        self._clock = clock
        self._sessions: dict[str, AnonymousSession] = {}
        self._global_generation_timestamps: deque[float] = deque()
        self._lock = threading.RLock()

    def create_session(self) -> AnonymousSession:
        """Create a session with an unpredictable opaque identifier."""

        now = self._clock()

        with self._lock:
            expired = self._pop_expired_locked(now)

            if len(self._sessions) >= self.max_sessions:
                session = None
            else:
                while True:
                    session_id = secrets.token_urlsafe(32)
                    if session_id not in self._sessions:
                        break

                session = AnonymousSession(
                    session_id=session_id,
                    created_at=now,
                    last_activity_at=now,
                )
                self._sessions[session_id] = session

        # Session creation is called from an async API route. Do not let
        # expired-session cleanup wait for a running generation on the event loop.
        self._close_sessions(expired, wait=False)

        if session is None:
            raise SessionLimitError("The active session limit has been reached.")

        return session

    def get_session(self, session_id: str) -> AnonymousSession:
        """Return a live session and update its last-activity timestamp."""

        now = self._clock()
        expired_session = None

        with self._lock:
            session = self._sessions.get(session_id)

            if session is None:
                raise SessionNotFoundError("The session was not found.")

            if self._is_expired(session, now):
                expired_session = self._sessions.pop(session_id)
                session = None
            else:
                session.last_activity_at = now
                # A refresh/new request inside the grace interval cancels the
                # best-effort page-close request and preserves the temp session.
                session.close_requested_at = None

        if expired_session is not None:
            # This method can be called directly from an async route. Defer any
            # wait on a running conversation worker to a cleanup thread.
            self._close_sessions([expired_session], wait=False)
            raise SessionNotFoundError("The session was not found.")

        if session is None:
            raise SessionNotFoundError("The session was not found.")

        return session

    def request_session_close(self, session_id: str) -> bool:
        """Schedule best-effort session cleanup after a short grace period.

        Browsers fire pagehide on refresh as well as close. Delaying deletion lets
        a refreshed page make its next request and preserve the temporary session.
        If no further activity arrives, the normal cleanup sweep expires it.
        """
        now = self._clock()
        expired_session = None
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            if self._is_expired(session, now):
                expired_session = self._sessions.pop(session_id)
            else:
                session.close_requested_at = now
                return True

        if expired_session is not None:
            self._close_sessions([expired_session], wait=False)
        return False

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and close its conversation services immediately."""

        with self._lock:
            session = self._sessions.pop(session_id, None)

        if session is None:
            return False

        self._close_sessions([session])
        return True

    def enforce_request_rate_limit(
        self,
        session_id: str,
        *,
        max_requests: int,
        window_seconds: float,
    ) -> None:
        """Enforce a sliding-window request limit for one anonymous session."""
        if max_requests < 1 or window_seconds <= 0:
            raise ValueError("Rate-limit settings must be positive.")

        session = self.get_session(session_id)
        now = self._clock()
        cutoff = now - window_seconds

        with session.lock:
            while session.request_timestamps and session.request_timestamps[0] <= cutoff:
                session.request_timestamps.popleft()

            if len(session.request_timestamps) >= max_requests:
                raise SessionRateLimitError(
                    "The session request rate limit has been reached."
                )

            session.request_timestamps.append(now)

    def enforce_global_generation_rate_limit(
        self,
        *,
        max_requests: int,
        window_seconds: float,
    ) -> None:
        """Apply a process-wide generation rate limit across all sessions."""
        if max_requests < 1 or window_seconds <= 0:
            raise ValueError("Rate-limit settings must be positive.")

        now = self._clock()
        cutoff = now - window_seconds
        with self._lock:
            while (
                self._global_generation_timestamps
                and self._global_generation_timestamps[0] <= cutoff
            ):
                self._global_generation_timestamps.popleft()

            if len(self._global_generation_timestamps) >= max_requests:
                raise SessionRateLimitError(
                    "The global generation rate limit has been reached."
                )

            self._global_generation_timestamps.append(now)

    def create_conversation(
        self,
        session_id: str,
        service_factory: Callable[[], Any],
    ) -> ConversationEntry:
        """Create a conversation owned by a specific live session."""

        session = self.get_session(session_id)

        # Re-check ownership while holding the registry lock. Otherwise a
        # concurrent session close could remove and clean the session between
        # get_session() and this insertion, leaving an orphan conversation.
        with self._lock:
            if self._sessions.get(session_id) is not session:
                raise SessionNotFoundError("The session was not found.")

            with session.lock:
                if len(session.conversations) >= self.max_conversations_per_session:
                    raise ConversationLimitError(
                        "The conversation limit has been reached."
                    )

                while True:
                    conversation_id = secrets.token_urlsafe(18)
                    if conversation_id not in session.conversations:
                        break

                service = service_factory()
                entry = ConversationEntry(
                    conversation_id=conversation_id,
                    service=service,
                )
                session.close_requested_at = None
                session.last_activity_at = self._clock()
                session.conversations[conversation_id] = entry
                return entry

    def get_conversation(
        self,
        session_id: str,
        conversation_id: str,
    ) -> ConversationEntry:
        """Retrieve a conversation without waiting for its generation lock."""

        session = self.get_session(session_id)

        with session.lock:
            entry = session.conversations.get(conversation_id)

            if entry is None or entry.closing:
                raise SessionNotFoundError("The conversation was not found.")

            return entry

    @contextmanager
    def conversation_operation(
        self,
        session_id: str,
        conversation_id: str,
        *,
        blocking: bool = True,
    ) -> Iterator[ConversationEntry]:
        """Serialize operations per conversation.

        API handlers should use ``blocking=False`` so an overlapping request is
        rejected immediately instead of blocking the asyncio event loop. The
        blocking default remains available for maintenance/tests running in a
        worker thread.
        """
        session = self.get_session(session_id)

        with session.lock:
            entry = session.conversations.get(conversation_id)

            if entry is None or entry.closing:
                raise SessionNotFoundError("The conversation was not found.")

        acquired = entry.lock.acquire(blocking=blocking)
        if not acquired:
            raise ConversationBusyError("The conversation is busy.")

        if entry.closing:
            entry.lock.release()
            raise SessionNotFoundError("The conversation was not found.")

        try:
            yield entry
        finally:
            entry.lock.release()

    def delete_conversation(
        self,
        session_id: str,
        conversation_id: str,
        *,
        blocking: bool = True,
    ) -> bool:
        """Delete a conversation and close its service without lock inversion."""
        session = self.get_session(session_id)

        with session.lock:
            entry = session.conversations.get(conversation_id)
            if entry is None:
                return False

        acquired = entry.lock.acquire(blocking=blocking)
        if not acquired:
            raise ConversationBusyError("The conversation is busy.")

        try:
            with session.lock:
                # It may have been removed by expiry/close while we acquired the lock.
                if session.conversations.get(conversation_id) is not entry:
                    return False
                entry.closing = True
                session.conversations.pop(conversation_id, None)

            self._close_service(entry.service)
            return True
        finally:
            entry.lock.release()

    def close_all(self) -> int:
        """Remove all sessions and close their conversation services."""

        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()

        self._close_sessions(sessions)
        return len(sessions)

    def cleanup_expired(self) -> int:
        """Remove expired sessions and close their conversation services."""

        now = self._clock()

        with self._lock:
            expired = self._pop_expired_locked(now)

        self._close_sessions(expired)
        return len(expired)

    def _pop_expired_locked(self, now: float) -> list[AnonymousSession]:
        expired_ids = [
            session_id
            for session_id, session in self._sessions.items()
            if self._is_expired(session, now)
        ]

        return [
            self._sessions.pop(session_id)
            for session_id in expired_ids
        ]

    def _close_sessions(
        self,
        sessions: list[AnonymousSession],
        *,
        wait: bool = True,
    ) -> None:
        """Close conversation services without lock inversion.

        Expiry can be discovered inside an async request handler. In that path,
        ``wait=False`` detaches the expired state immediately and defers waiting
        on any active conversation lock to a daemon cleanup worker. Explicit
        shutdown/close and the periodic sweeper use the default waiting mode.
        """
        entries_to_close: list[ConversationEntry] = []
        for session in sessions:
            # Never acquire a conversation lock while holding session.lock:
            # normal delete operations acquire the conversation lock first and
            # then briefly re-enter session.lock to remove that entry.
            with session.lock:
                entries_to_close.extend(session.conversations.values())
                session.conversations.clear()

        if not entries_to_close:
            return

        if not wait:
            worker = threading.Thread(
                target=self._close_entries,
                args=(entries_to_close,),
                name="hadota-session-cleanup",
                daemon=True,
            )
            worker.start()
            return

        self._close_entries(entries_to_close)

    def _close_entries(self, entries: list[ConversationEntry]) -> None:
        """Wait for each active operation, mark it closing, and close its service."""
        for entry in entries:
            with entry.lock:
                entry.closing = True
                self._close_service(entry.service)

    @staticmethod
    def _close_service(service: Any) -> None:
        """Best-effort cleanup of a service without owning its shared runtime."""

        close = getattr(service, "close", None)

        if callable(close):
            try:
                close()
            except Exception:
                # Cleanup must continue even if one service fails to close.
                pass

    def _is_expired(self, session: AnonymousSession, now: float) -> bool:
        close_requested_expired = (
            session.close_requested_at is not None
            and now - session.close_requested_at >= self.close_grace_seconds
        )
        return (
            now - session.last_activity_at >= self.idle_ttl_seconds
            or now - session.created_at >= self.absolute_ttl_seconds
            or close_requested_expired
        )

    @property
    def active_session_count(self) -> int:
        """Return the current number of stored sessions."""

        with self._lock:
            return len(self._sessions)
