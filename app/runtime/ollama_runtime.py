"""Ollama runtime adapter for the Local AI Assistant."""

from __future__ import annotations

import json
from collections.abc import Iterator
from time import perf_counter
from typing import Any

import httpx

from app.config import RuntimeSettings, load_settings
from app.runtime.exceptions import (
    InvalidGenerationRequestError,
    ModelNotFoundError,
    RuntimeConnectionError,
    RuntimeHTTPError,
    RuntimeResponseError,
    RuntimeStreamingError,
    RuntimeTimeoutError,
)
from app.schemas import GenerationRequest, GenerationResult, StreamEvent


class OllamaRuntime:
    """HTTP adapter for the local Ollama runtime."""

    def __init__(
        self,
        settings: RuntimeSettings | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self._owns_client = client is None

        self._client = client or httpx.Client(
            timeout=self.settings.request_timeout_seconds,
        )

    def close(self) -> None:
        """Close the HTTP client when it is owned by this adapter."""

        if self._owns_client:
            self._client.close()

    def __enter__(self) -> OllamaRuntime:  # noqa: PYI034
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _request_json(
        self,
        method: str,
        endpoint: str,
        *,
        json_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send an HTTP request and validate its JSON response."""

        url = f"{self.settings.ollama_host}{endpoint}"

        try:
            response = self._client.request(
                method,
                url,
                json=json_payload,
                timeout=self.settings.request_timeout_seconds,
            )

        except httpx.TimeoutException as error:
            raise RuntimeTimeoutError(
                "The Ollama request exceeded its configured timeout."
            ) from error

        except httpx.RequestError as error:
            raise RuntimeConnectionError(
                "The Ollama service could not be reached."
            ) from error

        try:
            response.raise_for_status()

        except httpx.HTTPStatusError as error:
            if response.status_code == 404:
                try:
                    error_payload = response.json()
                except ValueError:
                    error_payload = {}

                if isinstance(error_payload, dict):
                    error_message = error_payload.get("error", "")

                    if (
                        isinstance(error_message, str)
                        and "model" in error_message.lower()
                        and "not found" in error_message.lower()
                    ):
                        raise ModelNotFoundError(
                            "The requested model is not available in Ollama."
                        ) from error

            raise RuntimeHTTPError(
                "The Ollama service returned an unsuccessful HTTP status.",
                status_code=response.status_code,
            ) from error

        try:
            payload = response.json()

        except ValueError as error:
            raise RuntimeResponseError(
                "The Ollama service returned invalid JSON."
            ) from error

        if not isinstance(payload, dict):
            raise RuntimeResponseError("The Ollama response must be a JSON object.")

        return payload

    def _get_json(self, endpoint: str) -> dict[str, Any]:
        """Send a GET request and return a validated JSON object."""

        return self._request_json("GET", endpoint)

    def _post_json(
        self,
        endpoint: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Send a POST request and return a validated JSON object."""

        return self._request_json(
            "POST",
            endpoint,
            json_payload=payload,
        )

    def health(self) -> bool:
        """Return True when the Ollama version endpoint responds correctly."""

        payload = self._get_json("/api/version")
        version = payload.get("version")

        if not isinstance(version, str) or not version.strip():
            raise RuntimeResponseError(
                "The Ollama version response is missing a valid version."
            )

        return True

    def list_models(self) -> list[str]:
        """Return model identifiers reported by the Ollama inventory endpoint."""

        payload = self._get_json("/api/tags")
        models = payload.get("models")

        if not isinstance(models, list):
            raise RuntimeResponseError(
                "The Ollama model inventory does not contain a valid model list."
            )

        model_names: list[str] = []

        for item in models:
            if not isinstance(item, dict):
                raise RuntimeResponseError(
                    "The Ollama model inventory contains an invalid entry."
                )

            name = item.get("name")

            if not isinstance(name, str) or not name.strip():
                raise RuntimeResponseError(
                    "The Ollama model inventory contains an invalid model name."
                )

            model_names.append(name)

        return model_names

    @staticmethod
    def _optional_non_negative_int(
        payload: dict[str, Any],
        field_name: str,
    ) -> int | None:
        """Read an optional non-negative integer from a runtime response."""

        if field_name not in payload:
            return None

        value = payload[field_name]

        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise RuntimeResponseError(
                f"The Ollama response field '{field_name}' must be "
                "a non-negative integer when present."
            )

        return value

    @staticmethod
    def _optional_string(
        payload: dict[str, Any],
        field_name: str,
    ) -> str | None:
        """Read an optional string from a runtime response."""

        if field_name not in payload:
            return None

        value = payload[field_name]

        if not isinstance(value, str):
            raise RuntimeResponseError(
                f"The Ollama response field '{field_name}' must be a string."
            )

        return value

    def _build_generation_payload(
        self,
        request: GenerationRequest,
    ) -> dict[str, Any]:
        """Resolve request overrides and build the Ollama generate payload."""

        model_name = (
            request.model if request.model is not None else self.settings.model_name
        )

        num_ctx = (
            request.num_ctx if request.num_ctx is not None else self.settings.num_ctx
        )

        max_new_tokens = (
            request.max_new_tokens
            if request.max_new_tokens is not None
            else self.settings.max_new_tokens
        )

        temperature = (
            request.temperature
            if request.temperature is not None
            else self.settings.temperature
        )

        options: dict[str, Any] = {
            "num_ctx": num_ctx,
            "num_predict": max_new_tokens,
            "temperature": temperature,
        }

        for option_name in ("top_k", "top_p", "seed"):
            request_value = getattr(request, option_name)
            settings_value = getattr(self.settings, option_name)

            option_value = (
                request_value if request_value is not None else settings_value
            )

            if option_value is not None:
                options[option_name] = option_value

        if request.stop is not None:
            options["stop"] = request.stop

        keep_alive = (
            request.keep_alive
            if request.keep_alive is not None
            else self.settings.keep_alive
        )

        payload = {
            "model": model_name,
            "prompt": request.prompt,
            "stream": False,
            "think": self.settings.think,
            "keep_alive": keep_alive,
            "options": options,
        }

        if request.format is not None:
            payload["format"] = request.format

        return payload

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResult:
        """Execute a non-streaming generation request."""

        if request.stream:
            raise InvalidGenerationRequestError(
                "generate() requires stream=False. "
                "Use generate_stream() for streaming requests."
            )

        payload = self._build_generation_payload(request)

        start_time = perf_counter()

        raw_response = self._post_json(
            "/api/generate",
            payload,
        )

        client_latency_ms = (perf_counter() - start_time) * 1000.0

        response_text = raw_response.get("response")
        model_name = raw_response.get("model")
        done = raw_response.get("done")

        if not isinstance(response_text, str):
            raise RuntimeResponseError(
                "The Ollama response is missing valid generated text."
            )

        if not isinstance(model_name, str) or not model_name.strip():
            raise RuntimeResponseError(
                "The Ollama response is missing a valid model identifier."
            )

        if not isinstance(done, bool):
            raise RuntimeResponseError(
                "The Ollama response is missing a valid completion status."
            )

        return GenerationResult(
            text=response_text,
            model=model_name,
            done=done,
            done_reason=self._optional_string(
                raw_response,
                "done_reason",
            ),
            prompt_eval_count=self._optional_non_negative_int(
                raw_response,
                "prompt_eval_count",
            ),
            eval_count=self._optional_non_negative_int(
                raw_response,
                "eval_count",
            ),
            total_duration_ns=self._optional_non_negative_int(
                raw_response,
                "total_duration",
            ),
            load_duration_ns=self._optional_non_negative_int(
                raw_response,
                "load_duration",
            ),
            prompt_eval_duration_ns=self._optional_non_negative_int(
                raw_response,
                "prompt_eval_duration",
            ),
            eval_duration_ns=self._optional_non_negative_int(
                raw_response,
                "eval_duration",
            ),
            client_latency_ms=client_latency_ms,
        )

    def generate_stream(
        self,
        request: GenerationRequest,
    ) -> Iterator[StreamEvent]:
        """Yield normalized events from a streaming Ollama generation."""

        if not request.stream:
            raise InvalidGenerationRequestError(
                "generate_stream() requires stream=True. "
                "Use generate() for non-streaming requests."
            )

        payload = self._build_generation_payload(request)
        payload["stream"] = True

        url = f"{self.settings.ollama_host}/api/generate"
        start_time = perf_counter()

        accumulated_text: list[str] = []
        model_name = request.model or self.settings.model_name
        final_event_received = False

        try:
            with self._client.stream(
                "POST",
                url,
                json=payload,
                timeout=self.settings.request_timeout_seconds,
            ) as response:
                if response.is_error:
                    response.read()

                    try:
                        error_payload = response.json()
                    except ValueError:
                        error_payload = {}

                    if not isinstance(error_payload, dict):
                        error_payload = {}

                    error_message = error_payload.get("error", "")

                    if (
                        response.status_code == 404
                        and isinstance(error_message, str)
                        and "model" in error_message.lower()
                        and "not found" in error_message.lower()
                    ):
                        raise ModelNotFoundError(
                            "The requested model is not available in Ollama."
                        )

                    raise RuntimeHTTPError(
                        "The Ollama service returned an unsuccessful HTTP status.",
                        status_code=response.status_code,
                    )

                for line in response.iter_lines():
                    if not line.strip():
                        continue

                    try:
                        chunk = json.loads(line)
                    except (json.JSONDecodeError, TypeError) as error:
                        raise RuntimeStreamingError(
                            "The Ollama stream contains invalid JSON."
                        ) from error

                    if not isinstance(chunk, dict):
                        raise RuntimeStreamingError(
                            "Each Ollama stream event must be a JSON object."
                        )

                    error_message = chunk.get("error")

                    if isinstance(error_message, str) and error_message:
                        raise RuntimeStreamingError(
                            f"Ollama streaming failed: {error_message}"
                        )

                    text_delta = chunk.get("response", "")

                    if not isinstance(text_delta, str):
                        raise RuntimeStreamingError(
                            "The Ollama stream response must be a string."
                        )

                    chunk_model = chunk.get("model")

                    if chunk_model is not None:
                        if (
                            not isinstance(chunk_model, str)
                            or not chunk_model.strip()
                        ):
                            raise RuntimeStreamingError(
                                "The Ollama stream contains an invalid model name."
                            )

                        model_name = chunk_model

                    done = chunk.get("done", False)

                    if not isinstance(done, bool):
                        raise RuntimeStreamingError(
                            "The Ollama stream completion flag must be a boolean."
                        )

                    if text_delta:
                        accumulated_text.append(text_delta)

                    event_timestamp = perf_counter()

                    if done:
                        final_metadata = GenerationResult(
                            text="".join(accumulated_text),
                            model=model_name,
                            done=True,
                            done_reason=self._optional_string(
                                chunk,
                                "done_reason",
                            ),
                            prompt_eval_count=self._optional_non_negative_int(
                                chunk,
                                "prompt_eval_count",
                            ),
                            eval_count=self._optional_non_negative_int(
                                chunk,
                                "eval_count",
                            ),
                            total_duration_ns=self._optional_non_negative_int(
                                chunk,
                                "total_duration",
                            ),
                            load_duration_ns=self._optional_non_negative_int(
                                chunk,
                                "load_duration",
                            ),
                            prompt_eval_duration_ns=self._optional_non_negative_int(
                                chunk,
                                "prompt_eval_duration",
                            ),
                            eval_duration_ns=self._optional_non_negative_int(
                                chunk,
                                "eval_duration",
                            ),
                            client_latency_ms=(
                                perf_counter() - start_time
                            ) * 1000.0,
                        )

                        yield StreamEvent(
                            text_delta=text_delta,
                            event_timestamp=event_timestamp,
                            is_final=True,
                            final_metadata=final_metadata,
                        )

                        final_event_received = True
                        break

                    if text_delta:
                        yield StreamEvent(
                            text_delta=text_delta,
                            event_timestamp=event_timestamp,
                        )

                if not final_event_received:
                    raise RuntimeStreamingError(
                        "The Ollama stream ended without a final completion event."
                    )

        except httpx.TimeoutException as error:
            raise RuntimeTimeoutError(
                "The Ollama streaming request exceeded its configured timeout."
            ) from error

        except httpx.RequestError as error:
            raise RuntimeConnectionError(
                "The Ollama streaming service could not be reached."
            ) from error
