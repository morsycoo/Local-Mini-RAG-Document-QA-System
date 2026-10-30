"""Tests for anonymous session lifecycle management."""

import pytest

from app.web.sessions import (
    ConversationBusyError,
    ConversationLimitError,
    SessionLimitError,
    SessionRateLimitError,
    SessionManager,
    SessionNotFoundError,
)


class FakeClock:
    """A controllable clock for deterministic expiry tests."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def manager(clock):
    return SessionManager(
        max_sessions=2,
        max_conversations_per_session=2,
        idle_ttl_seconds=60,
        absolute_ttl_seconds=180,
        clock=clock,
    )


def test_create_session_generates_unique_opaque_ids(manager):
    first = manager.create_session()
    second = manager.create_session()

    assert first.session_id != second.session_id
    assert len(first.session_id) >= 32
    assert manager.active_session_count == 2


def test_session_limit_is_enforced(manager):
    manager.create_session()
    manager.create_session()

    with pytest.raises(SessionLimitError):
        manager.create_session()


def test_get_session_updates_last_activity(manager, clock):
    session = manager.create_session()

    clock.advance(20)
    retrieved = manager.get_session(session.session_id)

    assert retrieved is session
    assert retrieved.last_activity_at == clock.now


def test_idle_expiry_rejects_session(manager, clock):
    session = manager.create_session()

    clock.advance(60)

    with pytest.raises(SessionNotFoundError):
        manager.get_session(session.session_id)

    assert manager.active_session_count == 0


def test_absolute_lifetime_expires_even_with_recent_activity(manager, clock):
    session = manager.create_session()

    clock.advance(50)
    manager.get_session(session.session_id)

    clock.advance(50)
    manager.get_session(session.session_id)

    clock.advance(80)

    with pytest.raises(SessionNotFoundError):
        manager.get_session(session.session_id)


def test_cleanup_removes_expired_sessions(manager, clock):
    manager.create_session()

    clock.advance(60)

    removed = manager.cleanup_expired()

    assert removed == 1
    assert manager.active_session_count == 0


def test_delete_session_is_idempotent(manager):
    session = manager.create_session()

    assert manager.delete_session(session.session_id) is True
    assert manager.delete_session(session.session_id) is False
    assert manager.active_session_count == 0


def test_conversation_is_accessible_only_through_owning_session(manager):
    first_session = manager.create_session()
    second_session = manager.create_session()

    entry = manager.create_conversation(
        first_session.session_id,
        service_factory=object,
    )

    retrieved = manager.get_conversation(
        first_session.session_id,
        entry.conversation_id,
    )

    assert retrieved is entry

    with pytest.raises(SessionNotFoundError):
        manager.get_conversation(
            second_session.session_id,
            entry.conversation_id,
        )


def test_conversation_limit_is_enforced(manager):
    session = manager.create_session()

    manager.create_conversation(session.session_id, service_factory=object)
    manager.create_conversation(session.session_id, service_factory=object)

    with pytest.raises(ConversationLimitError):
        manager.create_conversation(
            session.session_id,
            service_factory=object,
        )


def test_delete_conversation_only_affects_owning_session(manager):
    first_session = manager.create_session()
    second_session = manager.create_session()

    entry = manager.create_conversation(
        first_session.session_id,
        service_factory=object,
    )

    assert manager.delete_conversation(
        first_session.session_id,
        entry.conversation_id,
    ) is True

    assert manager.delete_conversation(
        first_session.session_id,
        entry.conversation_id,
    ) is False

    with pytest.raises(SessionNotFoundError):
        manager.get_conversation(
            first_session.session_id,
            entry.conversation_id,
        )

    assert manager.active_session_count == 2


def test_invalid_capacity_configuration_is_rejected():
    with pytest.raises(ValueError):
        SessionManager(max_sessions=0)

    with pytest.raises(ValueError):
        SessionManager(max_conversations_per_session=0)

    with pytest.raises(ValueError):
        SessionManager(idle_ttl_seconds=0)

    with pytest.raises(ValueError):
        SessionManager(absolute_ttl_seconds=0)

def test_delete_session_closes_conversation_services(manager):
    session = manager.create_session()
    service = __import__("unittest.mock", fromlist=["Mock"]).Mock()

    manager.create_conversation(
        session.session_id,
        service_factory=lambda: service,
    )

    assert manager.delete_session(session.session_id) is True
    service.close.assert_called_once_with()


def test_cleanup_expired_closes_conversation_services(manager, clock):
    session = manager.create_session()
    service = __import__("unittest.mock", fromlist=["Mock"]).Mock()

    manager.create_conversation(
        session.session_id,
        service_factory=lambda: service,
    )

    clock.advance(60)

    assert manager.cleanup_expired() == 1
    service.close.assert_called_once_with()

def test_cleanup_continues_when_a_service_close_fails(manager):
    from unittest.mock import Mock

    session = manager.create_session()
    failing_service = Mock()
    healthy_service = Mock()

    failing_service.close.side_effect = RuntimeError(
        "Simulated cleanup failure"
    )

    manager.create_conversation(
        session.session_id,
        service_factory=lambda: failing_service,
    )
    manager.create_conversation(
        session.session_id,
        service_factory=lambda: healthy_service,
    )

    assert manager.delete_session(session.session_id) is True

    failing_service.close.assert_called_once_with()
    healthy_service.close.assert_called_once_with()
    assert manager.active_session_count == 0

def test_conversation_entry_has_independent_operation_lock(manager):
    from unittest.mock import Mock

    session = manager.create_session()

    first_service = Mock()
    second_service = Mock()

    first = manager.create_conversation(
        session.session_id,
        service_factory=lambda: first_service,
    )
    second = manager.create_conversation(
        session.session_id,
        service_factory=lambda: second_service,
    )

    assert first is not second
    assert hasattr(first, "lock")
    assert hasattr(second, "lock")
    assert first.lock is not second.lock

def test_delete_conversation_waits_for_active_operation(manager):
    import threading
    from unittest.mock import Mock

    session = manager.create_session()
    service = Mock()

    entry = manager.create_conversation(
        session.session_id,
        service_factory=lambda: service,
    )

    operation_started = threading.Event()
    allow_operation_to_finish = threading.Event()
    deletion_finished = threading.Event()

    def active_operation():
        with entry.lock:
            operation_started.set()
            assert allow_operation_to_finish.wait(timeout=5)

    def delete_conversation():
        manager.delete_conversation(
            session.session_id,
            entry.conversation_id,
        )
        deletion_finished.set()

    operation_thread = threading.Thread(target=active_operation)
    operation_thread.start()

    assert operation_started.wait(timeout=5)

    deletion_thread = threading.Thread(target=delete_conversation)
    deletion_thread.start()

    try:
        assert not deletion_finished.wait(timeout=0.2)
        service.close.assert_not_called()
    finally:
        allow_operation_to_finish.set()
        operation_thread.join(timeout=5)
        deletion_thread.join(timeout=5)

    assert not operation_thread.is_alive()
    assert not deletion_thread.is_alive()
    assert deletion_finished.is_set()
    service.close.assert_called_once_with()


def test_cleanup_expired_waits_for_active_operation(manager, clock):
    import threading
    from unittest.mock import Mock

    session = manager.create_session()
    service = Mock()

    entry = manager.create_conversation(
        session.session_id,
        service_factory=lambda: service,
    )

    operation_started = threading.Event()
    allow_operation_to_finish = threading.Event()
    cleanup_finished = threading.Event()
    cleanup_result = []

    def active_operation():
        with entry.lock:
            operation_started.set()
            assert allow_operation_to_finish.wait(timeout=5)

    def cleanup():
        cleanup_result.append(manager.cleanup_expired())
        cleanup_finished.set()

    operation_thread = threading.Thread(target=active_operation)
    operation_thread.start()

    assert operation_started.wait(timeout=5)
    clock.advance(60)

    cleanup_thread = threading.Thread(target=cleanup)
    cleanup_thread.start()

    try:
        assert not cleanup_finished.wait(timeout=0.2)
        service.close.assert_not_called()
    finally:
        allow_operation_to_finish.set()
        operation_thread.join(timeout=5)
        cleanup_thread.join(timeout=5)

    assert not operation_thread.is_alive()
    assert not cleanup_thread.is_alive()
    assert cleanup_finished.is_set()
    assert cleanup_result == [1]
    service.close.assert_called_once_with()

def test_get_conversation_rejects_entry_marked_as_closing(manager):
    from unittest.mock import Mock

    session = manager.create_session()
    service = Mock()

    entry = manager.create_conversation(
        session.session_id,
        service_factory=lambda: service,
    )

    with entry.lock:
        entry.closing = True

    with pytest.raises(SessionNotFoundError):
        manager.get_conversation(
            session.session_id,
            entry.conversation_id,
        )


def test_conversation_operation_holds_lock_until_operation_finishes(manager):
    import threading
    from unittest.mock import Mock

    session = manager.create_session()
    service = Mock()

    entry = manager.create_conversation(
        session.session_id,
        service_factory=lambda: service,
    )

    operation_started = threading.Event()
    allow_operation_to_finish = threading.Event()
    deletion_started = threading.Event()
    deletion_finished = threading.Event()
    deletion_result = []

    def active_operation():
        with manager.conversation_operation(
            session.session_id,
            entry.conversation_id,
        ) as active_entry:
            assert active_entry is entry
            operation_started.set()
            assert allow_operation_to_finish.wait(timeout=5)

    def delete_operation():
        assert operation_started.wait(timeout=5)
        deletion_started.set()
        deletion_result.append(
            manager.delete_conversation(
                session.session_id,
                entry.conversation_id,
            )
        )
        deletion_finished.set()

    operation_thread = threading.Thread(target=active_operation)
    deletion_thread = threading.Thread(target=delete_operation)

    operation_thread.start()
    deletion_thread.start()

    try:
        assert operation_started.wait(timeout=5)
        assert deletion_started.wait(timeout=5)

        # Give the deletion thread an opportunity to reach the lock.
        assert not deletion_finished.wait(timeout=0.1)
        service.close.assert_not_called()
    finally:
        allow_operation_to_finish.set()
        operation_thread.join(timeout=5)
        deletion_thread.join(timeout=5)

    assert not operation_thread.is_alive()
    assert not deletion_thread.is_alive()
    assert deletion_finished.is_set()
    assert deletion_result == [True]
    service.close.assert_called_once_with()

    with pytest.raises(SessionNotFoundError):
        with manager.conversation_operation(
            session.session_id,
            entry.conversation_id,
        ):
            pass

def test_conversation_operation_blocks_expired_session_cleanup(manager, clock):
    import threading
    from unittest.mock import Mock

    session = manager.create_session()
    service = Mock()

    entry = manager.create_conversation(
        session.session_id,
        service_factory=lambda: service,
    )

    operation_started = threading.Event()
    allow_operation_to_finish = threading.Event()
    cleanup_started = threading.Event()
    cleanup_finished = threading.Event()
    cleanup_result = []

    def active_operation():
        with manager.conversation_operation(
            session.session_id,
            entry.conversation_id,
        ) as active_entry:
            assert active_entry is entry
            operation_started.set()
            assert allow_operation_to_finish.wait(timeout=5)

    def cleanup():
        assert operation_started.wait(timeout=5)
        clock.advance(60)
        cleanup_started.set()
        cleanup_result.append(manager.cleanup_expired())
        cleanup_finished.set()

    operation_thread = threading.Thread(target=active_operation)
    cleanup_thread = threading.Thread(target=cleanup)

    operation_thread.start()
    cleanup_thread.start()

    try:
        assert operation_started.wait(timeout=5)
        assert cleanup_started.wait(timeout=5)
        assert not cleanup_finished.wait(timeout=0.1)
        service.close.assert_not_called()
    finally:
        allow_operation_to_finish.set()
        operation_thread.join(timeout=5)
        cleanup_thread.join(timeout=5)

    assert not operation_thread.is_alive()
    assert not cleanup_thread.is_alive()
    assert cleanup_finished.is_set()
    assert cleanup_result == [1]
    service.close.assert_called_once_with()

    with pytest.raises(SessionNotFoundError):
        manager.get_session(session.session_id)

def test_close_all_closes_all_sessions_and_conversations():
    from unittest.mock import Mock

    manager = SessionManager()
    first_service = Mock()
    second_service = Mock()

    first_session = manager.create_session()
    second_session = manager.create_session()

    manager.create_conversation(
        first_session.session_id,
        service_factory=lambda: first_service,
    )
    manager.create_conversation(
        second_session.session_id,
        service_factory=lambda: second_service,
    )

    closed_sessions = manager.close_all()

    assert closed_sessions == 2
    assert manager.active_session_count == 0
    first_service.close.assert_called_once_with()
    second_service.close.assert_called_once_with()

    with pytest.raises(SessionNotFoundError):
        manager.get_session(first_session.session_id)

    with pytest.raises(SessionNotFoundError):
        manager.get_session(second_session.session_id)


def test_close_all_is_idempotent():
    from unittest.mock import Mock

    manager = SessionManager()
    service = Mock()
    session = manager.create_session()

    manager.create_conversation(
        session.session_id,
        service_factory=lambda: service,
    )

    assert manager.close_all() == 1
    assert manager.close_all() == 0

    assert manager.active_session_count == 0
    service.close.assert_called_once_with()


def test_nonblocking_conversation_operation_rejects_busy_lock(manager):
    session = manager.create_session()
    entry = manager.create_conversation(session.session_id, service_factory=object)
    assert entry.lock.acquire(blocking=False)
    try:
        with pytest.raises(ConversationBusyError):
            with manager.conversation_operation(
                session.session_id,
                entry.conversation_id,
                blocking=False,
            ):
                pass
    finally:
        entry.lock.release()


def test_nonblocking_delete_rejects_busy_conversation(manager):
    session = manager.create_session()
    entry = manager.create_conversation(session.session_id, service_factory=object)
    assert entry.lock.acquire(blocking=False)
    try:
        with pytest.raises(ConversationBusyError):
            manager.delete_conversation(
                session.session_id,
                entry.conversation_id,
                blocking=False,
            )
        assert manager.get_conversation(session.session_id, entry.conversation_id) is entry
    finally:
        entry.lock.release()


def test_session_rate_limit_uses_sliding_window(manager, clock):
    session = manager.create_session()
    manager.enforce_request_rate_limit(
        session.session_id,
        max_requests=2,
        window_seconds=30,
    )
    manager.enforce_request_rate_limit(
        session.session_id,
        max_requests=2,
        window_seconds=30,
    )
    with pytest.raises(SessionRateLimitError):
        manager.enforce_request_rate_limit(
            session.session_id,
            max_requests=2,
            window_seconds=30,
        )

    clock.advance(31)
    manager.enforce_request_rate_limit(
        session.session_id,
        max_requests=2,
        window_seconds=30,
    )


def test_global_generation_rate_limit_is_shared_across_sessions(manager, clock):
    first = manager.create_session()
    second = manager.create_session()
    manager.enforce_global_generation_rate_limit(
        max_requests=1,
        window_seconds=30,
    )
    with pytest.raises(SessionRateLimitError):
        manager.enforce_global_generation_rate_limit(
            max_requests=1,
            window_seconds=30,
        )
    clock.advance(31)
    manager.enforce_global_generation_rate_limit(
        max_requests=1,
        window_seconds=30,
    )
    # Both sessions exist; the limiter itself is process-wide, not per session.
    assert manager.get_session(first.session_id) is first
    assert manager.get_session(second.session_id) is second


def test_page_close_request_expires_after_grace_without_activity(clock):
    manager = SessionManager(
        max_sessions=2,
        max_conversations_per_session=2,
        idle_ttl_seconds=60,
        absolute_ttl_seconds=180,
        close_grace_seconds=10,
        clock=clock,
    )
    session = manager.create_session()

    assert manager.request_session_close(session.session_id) is True
    clock.advance(9)
    assert manager.get_session(session.session_id) is session
    assert session.close_requested_at is None

    assert manager.request_session_close(session.session_id) is True
    clock.advance(10)
    assert manager.cleanup_expired() == 1
    assert manager.active_session_count == 0


def test_activity_during_close_grace_preserves_session(manager, clock):
    session = manager.create_session()
    assert manager.request_session_close(session.session_id) is True
    clock.advance(10)

    assert manager.get_session(session.session_id) is session
    assert session.close_requested_at is None


def test_session_cleanup_does_not_hold_session_lock_while_waiting_for_entry_lock(manager):
    import threading
    import time

    class ObservedLock:
        def __init__(self):
            self._inner = threading.Lock()
            self.waiting = threading.Event()

        def acquire(self, blocking=True, timeout=-1):
            if blocking:
                self.waiting.set()
            if timeout == -1:
                return self._inner.acquire(blocking)
            return self._inner.acquire(blocking, timeout)

        def release(self):
            self._inner.release()

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, exc_type, exc, tb):
            self.release()
            return False

    session = manager.create_session()
    entry = manager.create_conversation(session.session_id, service_factory=object)
    entry.lock = ObservedLock()
    assert entry.lock.acquire(blocking=False)
    entry.lock.waiting.clear()

    cleanup_done = threading.Event()

    def remove_session():
        try:
            manager.delete_session(session.session_id)
        finally:
            cleanup_done.set()

    worker = threading.Thread(target=remove_session, daemon=True)
    worker.start()

    try:
        assert entry.lock.waiting.wait(timeout=1.0)
        # Cleanup should have released session.lock before waiting on entry.lock.
        acquired = session.lock.acquire(timeout=0.5)
        assert acquired, "cleanup held session.lock while waiting on the entry lock"
        session.lock.release()
    finally:
        entry.lock.release()

    worker.join(timeout=2.0)
    assert cleanup_done.is_set(), "session cleanup did not finish"
    assert manager.active_session_count == 0


def test_expired_session_request_does_not_wait_for_active_conversation_lock(manager, clock):
    import threading

    class ClosingService:
        def __init__(self):
            self.closed = threading.Event()

        def close(self):
            self.closed.set()

    session = manager.create_session()
    service = ClosingService()
    entry = manager.create_conversation(
        session.session_id,
        service_factory=lambda: service,
    )
    assert entry.lock.acquire(blocking=False)
    clock.advance(60)

    done = threading.Event()
    errors = []

    def resolve_expired_session():
        try:
            manager.get_session(session.session_id)
        except BaseException as error:
            errors.append(error)
        finally:
            done.set()

    worker = threading.Thread(target=resolve_expired_session, daemon=True)
    worker.start()
    try:
        assert done.wait(timeout=0.5), (
            "expired-session lookup blocked while waiting on active conversation cleanup"
        )
    finally:
        entry.lock.release()

    worker.join(timeout=1.0)
    assert errors and isinstance(errors[0], SessionNotFoundError)
    assert service.closed.wait(timeout=1.0)


def test_create_conversation_rechecks_session_ownership_after_lookup(manager, monkeypatch):
    session = manager.create_session()
    original_get_session = manager.get_session

    def expire_between_lookup_and_insert(session_id):
        retrieved = original_get_session(session_id)
        manager.delete_session(session_id)
        return retrieved

    monkeypatch.setattr(manager, "get_session", expire_between_lookup_and_insert)
    with pytest.raises(SessionNotFoundError):
        manager.create_conversation(session.session_id, service_factory=object)
