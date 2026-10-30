"""Tests for the interactive command-line interface."""

from collections.abc import Iterator
from typing import Any

import pytest

from app.cli import run_cli
from app.config import RuntimeSettings
from app.conversation import ConversationManager
from app.runtime.exceptions import (
    IncompleteGenerationError,
    RuntimeConnectionError,
)
from app.schemas import GenerationResult, StreamEvent
from app.services.conversation_service import ConversationService
from app.structured_output import StructuredOutputValidationError


class FakeRuntime:
    """Provide a deterministic runtime for CLI tests."""

    def __init__(self, settings: RuntimeSettings) -> None:
        self.settings = settings
        self.closed = False

    def health(self) -> bool:
        """Simulate a healthy Ollama service."""
        return True

    def list_models(self) -> list[str]:
        """Return the configured model as an available model."""
        return [self.settings.model_name]

    def close(self) -> None:
        """Record that the CLI closed the runtime."""
        self.closed = True


class FakeService:
    """Simulate conversation operations without HTTP requests."""

    def __init__(self) -> None:
        self.conversation = ConversationManager()
        self.normal_calls: list[str] = []
        self.structured_calls: list[str] = []
        self.streaming_calls: list[str] = []

    def send_message(self, content: str) -> GenerationResult:
        """Return a predictable non-streaming response."""
        self.normal_calls.append(content)
        return GenerationResult(
            text=f"Reply to: {content}",
            model="qwen3.8:27b",
            done=True,
        )

    def send_structured_message(
        self,
        content: str,
        schema: dict[str, Any],
        **generation_options: Any,
    ) -> dict[str, Any]:
        """Return a schema-shaped response for JSON-mode tests."""

        self.structured_calls.append(content)

        # Keep the fake compatible with generation options passed by the CLI.
        assert generation_options.get("max_new_tokens") == 256

        return {
            "answer": f"Reply to: {content}",
            "confidence": 0.9,
        }

    def stream_message(self, content: str) -> Iterator[StreamEvent]:
        """Yield a deterministic streaming response."""

        self.streaming_calls.append(content)

        # Use a monotonic timestamp to represent when each event was observed.
        import time

        yield StreamEvent(
            text_delta="Streamed reply",
            event_timestamp=time.perf_counter(),
            is_final=False,
        )

        yield StreamEvent(
            text_delta="",
            event_timestamp=time.perf_counter(),
            is_final=True,
            final_metadata=GenerationResult(
                text="Streamed reply",
                model="qwen3.8:27b",
                done=True,
            ),
        )


@pytest.fixture
def cli_components(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[FakeRuntime, FakeService, list[str]]:
    """Inject fake components and deterministic user input."""

    import app.cli as cli_module

    settings = RuntimeSettings()
    runtime = FakeRuntime(settings)
    service = FakeService()

    # Avoid reading machine-specific environment settings in this test.
    monkeypatch.setattr(cli_module, "load_settings", lambda: settings)

    # Prevent the CLI from constructing a real HTTP client.
    monkeypatch.setattr(
        cli_module,
        "create_conversation_service",
        lambda _settings: (runtime, service),
    )

    return runtime, service, []


def run_with_inputs(
    monkeypatch: pytest.MonkeyPatch,
    commands: list[str],
    capsys: pytest.CaptureFixture[str],
) -> str:
    """Run the CLI with scripted input and return captured terminal output."""

    import app.cli as cli_module

    iterator = iter(commands)

    # Replace interactive input with a deterministic command sequence.
    monkeypatch.setattr(
        cli_module,
        "input",
        lambda _prompt="": next(iterator),
        raising=False,
    )

    run_cli()
    return capsys.readouterr().out


def test_help_command_and_exit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_components: tuple[FakeRuntime, FakeService, list[str]],
) -> None:
    """The help command displays available commands and exits cleanly."""

    runtime, _, _ = cli_components

    output = run_with_inputs(monkeypatch, ["/help", "/exit"], capsys)

    assert "Available commands:" in output
    assert "/status" in output
    assert "Goodbye!" in output
    assert runtime.closed


def test_status_command_reports_configured_model(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_components: tuple[FakeRuntime, FakeService, list[str]],
) -> None:
    """The status command reports connectivity and model availability."""

    _, _, _ = cli_components

    output = run_with_inputs(monkeypatch, ["/status", "/exit"], capsys)

    assert "Ollama status: CONNECTED" in output
    assert "Configured model status: AVAILABLE" in output


def test_stats_and_clear_commands(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_components: tuple[FakeRuntime, FakeService, list[str]],
) -> None:
    """Statistics reflect stored messages and clear removes conversation history."""

    _, service, _ = cli_components
    service.conversation.add_user_message("Earlier question")
    service.conversation.add_assistant_message("Earlier answer")

    output = run_with_inputs(
        monkeypatch,
        ["/stats", "/clear", "/stats", "/exit"],
        capsys,
    )

    assert "Total stored messages: 2" in output
    assert "Conversation history cleared." in output
    assert "Total stored messages: 0" in output


def test_normal_chat_uses_non_streaming_service(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_components: tuple[FakeRuntime, FakeService, list[str]],
) -> None:
    """Normal chat sends user input through the non-streaming service."""

    _, service, _ = cli_components

    output = run_with_inputs(
        monkeypatch,
        ["Hello", "/exit"],
        capsys,
    )

    assert service.normal_calls == ["Hello"]
    assert "Assistant: Reply to: Hello" in output


def test_stream_toggle_uses_streaming_service(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_components: tuple[FakeRuntime, FakeService, list[str]],
) -> None:
    """Enabling streaming routes subsequent messages through stream_message."""

    _, service, _ = cli_components

    output = run_with_inputs(
        monkeypatch,
        ["/stream", "Hello", "/exit"],
        capsys,
    )

    assert service.streaming_calls == ["Hello"]
    assert service.normal_calls == []
    assert "Streaming: ON" in output
    assert "Streamed reply" in output


def test_json_toggle_uses_structured_service(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_components: tuple[FakeRuntime, FakeService, list[str]],
) -> None:
    """Enabling JSON mode routes messages through structured generation."""

    _, service, _ = cli_components

    output = run_with_inputs(
        monkeypatch,
        ["/json", "Hello", "/exit"],
        capsys,
    )

    assert service.structured_calls == ["Hello"]
    assert '"answer": "Reply to: Hello"' in output
    assert '"confidence": 0.9' in output


def test_runtime_error_does_not_terminate_cli(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_components: tuple[FakeRuntime, FakeService, list[str]],
) -> None:
    """A generation error is reported while the CLI remains interactive."""

    _, service, _ = cli_components

    def fail_generation(content: str) -> GenerationResult:
        """Simulate a runtime connection failure."""
        raise RuntimeConnectionError("Ollama is unavailable.")

    service.send_message = fail_generation  # type: ignore[method-assign]

    output = run_with_inputs(
        monkeypatch,
        ["Hello", "/exit"],
        capsys,
    )

    assert "Request failed: Ollama is unavailable." in output
    assert "Goodbye!" in output


def test_eof_exits_and_closes_runtime(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_components: tuple[FakeRuntime, FakeService, list[str]],
) -> None:
    """End-of-input exits the application and closes the runtime."""

    runtime, _, _ = cli_components

    import app.cli as cli_module

    # Simulate a terminal whose input stream has been closed.
    def raise_eof(_prompt: str = "") -> str:
        raise EOFError

    monkeypatch.setattr(cli_module, "input", raise_eof, raising=False)

    run_cli()
    output = capsys.readouterr().out

    assert "Input closed. Exiting." in output
    assert runtime.closed


def test_empty_input_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_components: tuple[FakeRuntime, FakeService, list[str]],
) -> None:
    """Ignore empty input without calling any generation method."""
    _, service, _ = cli_components

    output = run_with_inputs(
        monkeypatch,
        ["", "   ", "Hello", "/exit"],
        capsys,
    )

    assert service.normal_calls == ["Hello"]
    assert service.structured_calls == []
    assert service.streaming_calls == []
    assert "Assistant: Reply to: Hello" in output


def test_unknown_command_does_not_call_generation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_components: tuple[FakeRuntime, FakeService, list[str]],
) -> None:
    """Report an unknown command without sending it to the model."""
    _, service, _ = cli_components

    output = run_with_inputs(
        monkeypatch,
        ["/unknown", "/exit"],
        capsys,
    )

    assert "Unknown command." in output
    assert service.normal_calls == []
    assert service.structured_calls == []
    assert service.streaming_calls == []


def test_json_mode_takes_precedence_when_both_modes_are_enabled(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_components: tuple[FakeRuntime, FakeService, list[str]],
) -> None:
    """Use structured generation when JSON and streaming modes are both enabled."""
    _, service, _ = cli_components

    output = run_with_inputs(
        monkeypatch,
        ["/stream", "/json", "Hello", "/exit"],
        capsys,
    )

    assert service.structured_calls == ["Hello"]
    assert service.normal_calls == []
    assert service.streaming_calls == []
    assert "Structured JSON mode: ON" in output
    assert '"answer": "Reply to: Hello"' in output


def test_incomplete_generation_does_not_terminate_cli(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_components: tuple[FakeRuntime, FakeService, list[str]],
) -> None:
    """Report incomplete generation and keep the CLI interactive."""
    _, service, _ = cli_components

    def incomplete_generation(content: str) -> GenerationResult:
        service.normal_calls.append(content)
        raise IncompleteGenerationError(
            "Generation is incomplete; assistant response was not stored."
        )

    service.send_message = incomplete_generation  # type: ignore[method-assign]

    output = run_with_inputs(
        monkeypatch,
        ["Hello", "/exit"],
        capsys,
    )

    assert (
        "Request failed: Generation is incomplete; assistant response was not stored."
    ) in output
    assert "Goodbye!" in output
    assert service.normal_calls == ["Hello"]

def test_shared_runtime_factory_creates_independent_services():
    from unittest.mock import Mock

    from app.cli import create_conversation_service_with_runtime

    runtime = Mock()
    runtime.settings = Mock()

    first_service = create_conversation_service_with_runtime(runtime)
    second_service = create_conversation_service_with_runtime(runtime)

    assert first_service is not second_service
    assert first_service.conversation is not second_service.conversation
    assert first_service._runtime is runtime
    assert second_service._runtime is runtime

    runtime.close.assert_not_called()

