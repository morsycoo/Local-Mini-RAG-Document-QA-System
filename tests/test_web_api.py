"""Tests for the initial HADOTA web API."""

from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.web.main import create_app


def make_runtime(healthy: bool = True) -> Mock:
    """Create a fake runtime without contacting Ollama."""
    runtime = Mock()
    runtime.health.return_value = healthy
    runtime.close.return_value = None
    return runtime


def make_service_factory(runtime: Mock, service: Mock):
    """Return a factory that provides fake dependencies."""

    def factory():
        return runtime, service

    return factory


def test_health_reports_web_liveness_even_when_runtime_is_healthy():
    runtime = make_runtime(healthy=True)
    service = Mock()
    app = create_app(make_service_factory(runtime, service))

    with TestClient(app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    runtime.health.assert_not_called()
    runtime.close.assert_called_once_with()


def test_health_remains_ok_when_runtime_is_unhealthy():
    runtime = make_runtime(healthy=False)
    service = Mock()
    app = create_app(make_service_factory(runtime, service))

    with TestClient(app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    runtime.health.assert_not_called()
    runtime.close.assert_called_once_with()


def test_readiness_returns_unavailable_when_runtime_is_unhealthy():
    runtime = make_runtime(healthy=False)
    service = Mock()
    app = create_app(make_service_factory(runtime, service))

    with TestClient(app) as client:
        response = client.get("/api/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "reason": "runtime_unavailable",
    }
    runtime.health.assert_called_once_with()


def test_readiness_returns_ready_when_runtime_is_healthy():
    runtime = make_runtime(healthy=True)
    service = Mock()
    app = create_app(make_service_factory(runtime, service))

    with TestClient(app) as client:
        response = client.get("/api/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
    runtime.health.assert_called_once_with()


def test_health_does_not_leak_runtime_errors():
    from app.runtime.exceptions import RuntimeConnectionError

    runtime = make_runtime()
    runtime.health.side_effect = RuntimeConnectionError(
        "Internal runtime connection details"
    )
    service = Mock()
    app = create_app(make_service_factory(runtime, service))

    with TestClient(app) as client:
        response = client.get("/api/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "reason": "runtime_unavailable",
    }
    assert "Internal runtime connection details" not in response.text


def test_runtime_is_closed_when_application_stops():
    runtime = make_runtime()
    service = Mock()
    app = create_app(make_service_factory(runtime, service))

    with TestClient(app):
        runtime.close.assert_not_called()

    runtime.close.assert_called_once_with()


def test_api_documentation_endpoints_are_disabled():
    runtime = make_runtime()
    service = Mock()
    app = create_app(make_service_factory(runtime, service))

    with TestClient(app) as client:
        assert client.get("/openapi.json").status_code == 404
        assert client.get("/docs").status_code == 404
        assert client.get("/redoc").status_code == 404

def test_session_manager_closes_all_sessions_when_application_stops(monkeypatch):
    from app.web import main

    runtime = make_runtime()
    service = Mock()
    session_manager = Mock()
    session_manager.close_all.return_value = 0

    monkeypatch.setattr(
        main,
        "SessionManager",
        lambda **kwargs: session_manager,
    )

    app = create_app(make_service_factory(runtime, service))

    with TestClient(app):
        assert app.state.session_manager is session_manager

    session_manager.close_all.assert_called_once_with()
    runtime.close.assert_called_once_with()



def test_readiness_rejects_missing_configured_model():
    from app.config import RuntimeSettings

    runtime = make_runtime(healthy=True)
    runtime.list_models.return_value = ["other-model:latest"]
    service = Mock()
    app = create_app(make_service_factory(runtime, service))
    app.state.settings = RuntimeSettings(model_name="qwen3.8:27b")

    with TestClient(app) as client:
        response = client.get("/api/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "reason": "model_unavailable",
    }
    runtime.list_models.assert_called_once_with()
