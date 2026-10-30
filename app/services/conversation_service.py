"""Conversation orchestration service for the Local AI Assistant."""

from collections.abc import Iterator
from typing import Any

from app.config import RuntimeSettings, load_settings
from app.context.window_manager import ContextWindowManager
from app.conversation import ConversationManager, ConversationMessage
from app.prompt_builder import PromptBuilder
from app.runtime.base import RuntimeInterface
from app.runtime.exceptions import IncompleteGenerationError, RuntimeStreamingError
from app.schemas import (
    GenerationRequest,
    GenerationResult,
    StreamEvent,
)
from app.structured_output import parse_and_validate_json


def _merge_continuation_text(previous_text: str, continuation_text: str) -> str:
    """Merge continuation text while removing an echoed response or overlap."""
    if not continuation_text:
        return previous_text

    # Some models repeat the entire previous response before continuing.
    if continuation_text.startswith(previous_text):
        continuation_text = continuation_text[len(previous_text):]
        return previous_text + continuation_text

    # Otherwise remove the longest exact suffix/prefix overlap.
    max_overlap = min(len(previous_text), len(continuation_text))
    for overlap in range(max_overlap, 3, -1):
        if previous_text.endswith(continuation_text[:overlap]):
            return previous_text + continuation_text[overlap:]

    return previous_text + continuation_text


class ConversationService:
    """Coordinate conversation history, prompt construction, and generation."""

    def __init__(
        self,
        runtime: RuntimeInterface,
        conversation: ConversationManager,
        prompt_builder: PromptBuilder,
        context_window_manager: ContextWindowManager | None = None,
        settings: RuntimeSettings | None = None,
    ) -> None:
        self._runtime = runtime
        self._conversation = conversation
        self._prompt_builder = prompt_builder
        self._context_window_manager = context_window_manager
        self._settings = settings or load_settings()
        self._continuation_available = False

    @property
    def conversation(self) -> ConversationManager:
        """Expose the managed conversation."""
        return self._conversation

    def _build_request(
        self,
        generation_options: dict[str, Any],
        *,
        messages: list[ConversationMessage] | None = None,
    ) -> GenerationRequest:
        """Build a generation request with consistent context settings."""
        if messages is None:
            messages = self._conversation.get_messages()
        else:
            messages = list(messages)

        num_ctx = generation_options.get("num_ctx")
        if num_ctx is None:
            num_ctx = self._settings.num_ctx

        max_new_tokens = generation_options.get("max_new_tokens")
        if max_new_tokens is None:
            max_new_tokens = self._settings.max_new_tokens

        if self._context_window_manager is not None:
            messages = self._context_window_manager.fit_messages(
                messages=messages,
                num_ctx=num_ctx,
                max_new_tokens=max_new_tokens,
            )

        resolved_options = {
            **generation_options,
            "num_ctx": num_ctx,
            "max_new_tokens": max_new_tokens,
        }

        return self._prompt_builder.build_request(
            messages,
            **resolved_options,
        )

    def send_message(
        self,
        content: str,
        **generation_options: Any,
    ) -> GenerationResult:
        """Generate an assistant response to a new user message."""
        self._validate_message(content, generation_options)

        # A new user turn replaces any pending continuation state. Failed turns
        # are rolled back so a normal retry does not duplicate the user message.
        self._continuation_available = False
        self._conversation.add_user_message(content)

        try:
            request = self._build_request(generation_options)
            result = self._runtime.generate(request)

            if not result.done:
                raise IncompleteGenerationError(
                    "Generation is incomplete; assistant response was not stored."
                )

            self._conversation.add_assistant_message(result.text)
            self._continuation_available = result.done_reason == "length"
            return result
        except BaseException:
            self._conversation.remove_last_message_if(
                role="user",
                content=content,
            )
            self._continuation_available = False
            raise

    @property
    def continuation_available(self) -> bool:
        """Whether the latest assistant response can be continued."""
        return self._continuation_available

    def continue_message(
        self,
        **generation_options: Any,
    ) -> GenerationResult:
        """Continue a truncated assistant response without adding a user turn."""
        if not self._continuation_available:
            raise ValueError("There is no incomplete assistant response to continue.")

        messages = self._conversation.get_messages()
        if not messages or messages[-1].role != "assistant":
            self._continuation_available = False
            raise ValueError("The latest conversation message is not an assistant response.")

        previous_text = messages[-1].content
        continuation_instruction = (
            "Continue your previous response from exactly where it stopped. "
            "Do not repeat or summarize any text already written. "
            "Continue naturally, preserving the language, formatting, and structure. "
            "If the response is still incomplete, finish at a sensible boundary."
        )
        prompt_messages = [
            *messages,
            ConversationMessage(
                role="user",
                content=continuation_instruction,
            ),
        ]

        self._validate_message(continuation_instruction, generation_options)
        request = self._build_request(
            generation_options,
            messages=prompt_messages,
        )
        result = self._runtime.generate(request)

        if not result.done:
            raise IncompleteGenerationError(
                "Continuation is incomplete; conversation history was not updated."
            )

        combined_text = _merge_continuation_text(previous_text, result.text)
        self._conversation.replace_last_assistant_message(combined_text)
        self._continuation_available = result.done_reason == "length"

        return result.model_copy(update={"text": combined_text})

    def send_structured_message(
        self,
        content: str,
        schema: dict[str, Any],
        **generation_options: Any,
    ) -> Any:
        """Generate, parse, and validate a structured assistant response."""
        if "format" in generation_options:
            raise ValueError(
                "Use the schema argument instead of specifying format directly."
            )

        self._validate_message(content, generation_options)

        messages = [
            *self._conversation.get_messages(),
            ConversationMessage(role="user", content=content),
        ]

        num_ctx = generation_options.get("num_ctx")
        if num_ctx is None:
            num_ctx = self._settings.num_ctx

        max_new_tokens = generation_options.get("max_new_tokens")
        if max_new_tokens is None:
            max_new_tokens = self._settings.max_new_tokens

        if self._context_window_manager is not None:
            messages = self._context_window_manager.fit_messages(
                messages=messages,
                num_ctx=num_ctx,
                max_new_tokens=max_new_tokens,
            )

        request = self._prompt_builder.build_request(
            messages,
            **{
                **generation_options,
                "format": schema,
                "num_ctx": num_ctx,
                "max_new_tokens": max_new_tokens,
            },
        )

        self._conversation.add_user_message(content)
        try:
            result = self._runtime.generate(request)

            if not result.done:
                raise IncompleteGenerationError(
                    "Generation is incomplete; assistant response was not stored."
                )

            parsed_result = parse_and_validate_json(result.text, schema)
            self._conversation.add_assistant_message(result.text)
            return parsed_result
        except BaseException:
            self._conversation.remove_last_message_if(
                role="user",
                content=content,
            )
            raise

    def stream_message(
        self,
        content: str,
        **generation_options: Any,
    ) -> Iterator[StreamEvent]:
        """Stream a response and commit the user/assistant turn only on success.

        A failed or interrupted stream rolls back the pending user turn, allowing
        a retry without silently duplicating history.
        """
        self._validate_message(content, generation_options)

        if generation_options.get("stream", True) is not True:
            raise ValueError("stream_message requires stream=True.")

        options = {
            **generation_options,
            "stream": True,
        }

        self._continuation_available = False
        self._conversation.add_user_message(content)

        collected_text: list[str] = []
        final_result: GenerationResult | None = None
        runtime_stream: Iterator[StreamEvent] | None = None
        completed = False

        try:
            request: GenerationRequest = self._build_request(options)
            runtime_stream = self._runtime.generate_stream(request)

            for event in runtime_stream:
                if event.is_final:
                    final_result = event.final_metadata

                    if final_result is None or not final_result.done:
                        raise RuntimeStreamingError(
                            "Streaming ended without valid completion metadata."
                        )

                if event.text_delta:
                    collected_text.append(event.text_delta)

                yield event

            if final_result is None:
                raise RuntimeStreamingError(
                    "Streaming ended without a final completion event."
                )

            response_text = "".join(collected_text)

            if response_text != final_result.text:
                raise RuntimeStreamingError(
                    "Streamed text does not match final generation metadata."
                )

            self._conversation.add_assistant_message(response_text)
            self._continuation_available = final_result.done_reason == "length"
            completed = True

        finally:
            try:
                if runtime_stream is not None:
                    close = getattr(runtime_stream, "close", None)
                    if callable(close):
                        close()
            finally:
                if not completed:
                    self._conversation.remove_last_message_if(
                        role="user",
                        content=content,
                    )
                    self._continuation_available = False

    @staticmethod
    def _validate_message(
        content: str,
        generation_options: dict[str, Any],
    ) -> None:
        """Validate user input and prevent prompt overrides."""
        if not isinstance(content, str) or not content.strip():
            raise ValueError("User message must not be empty or whitespace-only.")

        if "prompt" in generation_options:
            raise ValueError("The prompt must be generated from conversation messages.")
