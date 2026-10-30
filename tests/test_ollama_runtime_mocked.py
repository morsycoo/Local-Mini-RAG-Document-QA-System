"""Mocked tests for the Ollama runtime adapter."""

import json

import httpx
import pytest

from app.config import RuntimeSettings
from app.runtime.exceptions import (
    InvalidGenerationRequestError,
    ModelNotFoundError,
    RuntimeConnectionError,
    RuntimeHTTPError,
    RuntimeResponseError,
    RuntimeStreamingError,
    RuntimeTimeoutError,
)
from app.runtime.ollama_runtime import OllamaRuntime
from app.schemas import GenerationRequest, StreamEvent


def make_runtime(handler):
    """Create an adapter backed by a mocked HTTP transport."""

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
    )

    settings = RuntimeSettings(
        ollama_host="http://test-server",
        model_name="qwen3.8:27b",
    )

    runtime = OllamaRuntime(
        settings=settings,
        client=client,
    )

    return runtime, client


def test_health_success() -> None:
    """Accept a valid version response."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/version"

        return httpx.Response(
            200,
            json={"version": "0.35.1"},
        )

    runtime, client = make_runtime(handler)

    try:
        assert runtime.health() is True
    finally:
        client.close()


def test_health_rejects_missing_version() -> None:
    """Reject a successful response with an invalid version payload."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    runtime, client = make_runtime(handler)

    try:
        with pytest.raises(RuntimeResponseError):
            runtime.health()
    finally:
        client.close()


def test_list_models_success() -> None:
    """Return model identifiers from the inventory response."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"

        return httpx.Response(
            200,
            json={
                "models": [
                    {"name": "qwen3.8:27b"},
                    {"name": "example-model:latest"},
                ]
            },
        )

    runtime, client = make_runtime(handler)

    try:
        assert runtime.list_models() == [
            "qwen3.8:27b",
            "example-model:latest",
        ]
    finally:
        client.close()


def test_list_models_empty_inventory() -> None:
    """Accept a valid inventory containing no models."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"models": []},
        )

    runtime, client = make_runtime(handler)

    try:
        assert runtime.list_models() == []
    finally:
        client.close()


def test_list_models_rejects_invalid_payload() -> None:
    """Reject an inventory response without a models list."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"unexpected": []},
        )

    runtime, client = make_runtime(handler)

    try:
        with pytest.raises(RuntimeResponseError):
            runtime.list_models()
    finally:
        client.close()


def test_http_error_is_normalized() -> None:
    """Preserve the HTTP status in the runtime exception."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            json={"error": "Service unavailable"},
        )

    runtime, client = make_runtime(handler)

    try:
        with pytest.raises(RuntimeHTTPError) as captured:
            runtime.health()

        assert captured.value.status_code == 503
    finally:
        client.close()


def test_timeout_is_normalized() -> None:
    """Translate a timeout into the runtime timeout exception."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(
            "Simulated timeout",
            request=request,
        )

    runtime, client = make_runtime(handler)

    try:
        with pytest.raises(RuntimeTimeoutError):
            runtime.health()
    finally:
        client.close()


def test_connection_error_is_normalized() -> None:
    """Translate a connection failure into a runtime connection exception."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(
            "Simulated connection failure",
            request=request,
        )

    runtime, client = make_runtime(handler)

    try:
        with pytest.raises(RuntimeConnectionError):
            runtime.health()
    finally:
        client.close()


def test_invalid_json_is_normalized() -> None:
    """Translate an invalid JSON response into a runtime response error."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"{invalid-json",
            headers={"Content-Type": "application/json"},
        )

    runtime, client = make_runtime(handler)

    try:
        with pytest.raises(RuntimeResponseError):
            runtime.health()
    finally:
        client.close()


def test_generate_success_normalizes_response_and_payload() -> None:
    """Normalize a successful response and verify the outgoing payload."""

    captured_payload = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/generate"

        body = json.loads(request.content)
        captured_payload.update(body)

        return httpx.Response(
            200,
            json={
                "model": "qwen3.8:27b",
                "response": "A runtime executes model inference.",
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 8,
                "eval_count": 7,
                "total_duration": 1_200_000_000,
                "load_duration": 100_000_000,
                "prompt_eval_duration": 200_000_000,
                "eval_duration": 800_000_000,
            },
        )

    runtime, client = make_runtime(handler)

    try:
        result = runtime.generate(
            GenerationRequest(
                prompt="Explain an LLM runtime.",
            )
        )

        assert result.text == "A runtime executes model inference."
        assert result.model == "qwen3.8:27b"
        assert result.done is True
        assert result.done_reason == "stop"
        assert result.prompt_eval_count == 8
        assert result.eval_count == 7
        assert result.total_duration_ns == 1_200_000_000
        assert result.load_duration_ns == 100_000_000
        assert result.prompt_eval_duration_ns == 200_000_000
        assert result.eval_duration_ns == 800_000_000
        assert result.client_latency_ms is not None
        assert result.client_latency_ms >= 0

        assert captured_payload["model"] == "qwen3.8:27b"
        assert captured_payload["prompt"] == "Explain an LLM runtime."
        assert captured_payload["stream"] is False
        assert captured_payload["keep_alive"] == "5m"

        assert captured_payload["options"] == {
            "num_ctx": 4096,
            "num_predict": 512,
            "temperature": 0.0,
        }

    finally:
        client.close()


def test_generate_passes_json_format_to_ollama() -> None:
    """Pass the requested structured output format to Ollama."""

    captured_payload = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payload.update(json.loads(request.content))

        return httpx.Response(
            200,
            json={
                "model": "qwen3.8:27b",
                "response": '{"status":"ok"}',
                "done": True,
            },
        )

    runtime, client = make_runtime(handler)

    try:
        result = runtime.generate(
            GenerationRequest(
                prompt="Return a JSON object with status set to ok.",
                format="json",
            )
        )

        assert result.text == '{"status":"ok"}'
        assert captured_payload["format"] == "json"
    finally:
        client.close()


def test_generate_omits_format_when_not_requested() -> None:
    """Preserve the default payload when no output format is requested."""

    captured_payload = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payload.update(json.loads(request.content))

        return httpx.Response(
            200,
            json={
                "model": "qwen3.8:27b",
                "response": "Ordinary text response.",
                "done": True,
            },
        )

    runtime, client = make_runtime(handler)

    try:
        runtime.generate(GenerationRequest(prompt="Explain an LLM runtime."))

        assert "format" not in captured_payload
    finally:
        client.close()


def test_generate_passes_json_schema_to_ollama() -> None:
    """Pass the complete JSON Schema to Ollama."""

    captured_payload = {}

    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["answer", "confidence"],
        "additionalProperties": False,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payload.update(json.loads(request.content))

        return httpx.Response(
            200,
            json={
                "model": "qwen3.8:27b",
                "response": '{"answer":"Test response","confidence":0.95}',
                "done": True,
            },
        )

    runtime, client = make_runtime(handler)

    try:
        result = runtime.generate(
            GenerationRequest(
                prompt="Answer the question using the provided JSON Schema.",
                format=schema,
            )
        )

        assert result.text == ('{"answer":"Test response","confidence":0.95}')
        assert captured_payload["format"] == schema
    finally:
        client.close()


def test_generate_resolves_explicit_request_overrides() -> None:
    """Use request options when they override the central defaults."""

    captured_payload = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payload.update(json.loads(request.content))

        return httpx.Response(
            200,
            json={
                "model": "qwen3.8:27b",
                "response": "Override accepted.",
                "done": True,
            },
        )

    runtime, client = make_runtime(handler)

    try:
        runtime.generate(
            GenerationRequest(
                prompt="Test explicit overrides.",
                max_new_tokens=24,
                num_ctx=2048,
                temperature=0.2,
                top_k=20,
                top_p=0.9,
                seed=42,
                stop=["END_MARKER"],
                keep_alive="10m",
            )
        )

        assert captured_payload["stream"] is False
        assert captured_payload["keep_alive"] == "10m"

        assert captured_payload["options"] == {
            "num_ctx": 2048,
            "num_predict": 24,
            "temperature": 0.2,
            "top_k": 20,
            "top_p": 0.9,
            "seed": 42,
            "stop": ["END_MARKER"],
        }

    finally:
        client.close()


def test_generate_rejects_streaming_request_before_network_call() -> None:
    """Reject streaming requests in the non-streaming method."""

    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("No HTTP request should be sent.")

    runtime, client = make_runtime(handler)

    try:
        request = GenerationRequest(
            prompt="Test streaming rejection.",
            stream=True,
        )

        with pytest.raises(InvalidGenerationRequestError):
            runtime.generate(request)

    finally:
        client.close()


def test_generate_rejects_invalid_response_structure() -> None:
    """Reject a response that omits required generation fields."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "qwen3.8:27b",
                "done": True,
            },
        )

    runtime, client = make_runtime(handler)

    try:
        with pytest.raises(RuntimeResponseError):
            runtime.generate(GenerationRequest(prompt="Test invalid response."))

    finally:
        client.close()


def test_generate_rejects_negative_runtime_metadata() -> None:
    """Reject invalid runtime counters instead of normalizing them."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "qwen3.8:27b",
                "response": "Invalid metadata.",
                "done": True,
                "eval_count": -1,
            },
        )

    runtime, client = make_runtime(handler)

    try:
        with pytest.raises(RuntimeResponseError):
            runtime.generate(GenerationRequest(prompt="Test invalid metadata."))

    finally:
        client.close()


def test_generate_maps_model_not_found_error() -> None:
    """Translate the Ollama model-not-found response into a specific exception."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={"error": "model 'missing-model' not found"},
        )

    runtime, client = make_runtime(handler)

    try:
        with pytest.raises(ModelNotFoundError):
            runtime.generate(
                GenerationRequest(
                    prompt="Test missing model.",
                    model="missing-model",
                )
            )

    finally:
        client.close()


def test_generate_stream_success_and_final_metadata() -> None:
    """Yield ordered text chunks and normalized final metadata."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/generate"

        payload = json.loads(request.content)
        assert payload["stream"] is True
        assert payload["think"] is False

        chunks = [
            {
                "model": "qwen3.8:27b",
                "response": "Streaming ",
                "done": False,
            },
            {
                "model": "qwen3.8:27b",
                "response": "works.",
                "done": False,
            },
            {
                "model": "qwen3.8:27b",
                "response": "",
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 8,
                "eval_count": 4,
                "total_duration": 1_000_000_000,
            },
        ]

        content = "".join(json.dumps(chunk) + "\n" for chunk in chunks)

        return httpx.Response(
            200,
            content=content.encode(),
            headers={"Content-Type": "application/x-ndjson"},
        )

    runtime, client = make_runtime(handler)

    try:
        events = list(
            runtime.generate_stream(
                GenerationRequest(
                    prompt="Explain streaming.",
                    stream=True,
                )
            )
        )

        text_events = [event.text_delta for event in events if event.text_delta]

        assert text_events == ["Streaming ", "works."]

        final_events = [event for event in events if event.is_final]
        assert len(final_events) == 1

        final_event = final_events[0]
        assert final_event.final_metadata is not None
        assert final_event.final_metadata.text == "Streaming works."
        assert final_event.final_metadata.model == "qwen3.8:27b"
        assert final_event.final_metadata.done is True
        assert final_event.final_metadata.done_reason == "stop"
        assert final_event.final_metadata.prompt_eval_count == 8
        assert final_event.final_metadata.eval_count == 4
        assert final_event.final_metadata.client_latency_ms >= 0

        assert all(isinstance(event, StreamEvent) for event in events)

    finally:
        client.close()


def test_generate_stream_rejects_non_streaming_request() -> None:
    """Reject requests with streaming disabled before making an HTTP call."""

    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("No HTTP request should be sent.")

    runtime, client = make_runtime(handler)

    try:
        with pytest.raises(InvalidGenerationRequestError):
            list(
                runtime.generate_stream(
                    GenerationRequest(
                        prompt="Test invalid streaming request.",
                        stream=False,
                    )
                )
            )
    finally:
        client.close()


def test_generate_stream_rejects_invalid_json() -> None:
    """Reject malformed JSON in a streaming response."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b'{"model":"qwen3.8:27b","response":\n',
            headers={"Content-Type": "application/x-ndjson"},
        )

    runtime, client = make_runtime(handler)

    try:
        with pytest.raises(RuntimeStreamingError):
            list(
                runtime.generate_stream(
                    GenerationRequest(
                        prompt="Test malformed stream.",
                        stream=True,
                    )
                )
            )
    finally:
        client.close()


def test_generate_stream_requires_final_event() -> None:
    """Reject a stream that ends without a completion event."""

    def handler(request: httpx.Request) -> httpx.Response:
        chunk = {
            "model": "qwen3.8:27b",
            "response": "Partial output",
            "done": False,
        }

        return httpx.Response(
            200,
            content=(json.dumps(chunk) + "\n").encode(),
            headers={"Content-Type": "application/x-ndjson"},
        )

    runtime, client = make_runtime(handler)

    try:
        with pytest.raises(RuntimeStreamingError):
            list(
                runtime.generate_stream(
                    GenerationRequest(
                        prompt="Test missing final event.",
                        stream=True,
                    )
                )
            )
    finally:
        client.close()
