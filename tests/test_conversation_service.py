"""Tests for the conversation service."""

from collections.abc import Iterator
from unittest.mock import Mock

import pytest

from app.conversation import ConversationManager
from app.prompt_builder import PromptBuilder
from app.runtime.exceptions import (
    IncompleteGenerationError,
    RuntimeConnectionError,
    RuntimeStreamingError,
)
from app.schemas import GenerationRequest, GenerationResult, StreamEvent
from app.structured_output import (
    StructuredOutputParsingError,
    StructuredOutputValidationError,
)


@pytest.fixture
def runtime() -> Mock:
    """Create a mocked runtime adapter."""
    adapter = Mock()
    adapter.generate.return_value = GenerationResult(
        text="Hello! How can I help you?",
        model="qwen3.8:27b",
        done=True,
        done_reason="stop",
        prompt_eval_count=10,
        eval_count=8,
    )
    return adapter


@pytest.fixture
def conversation() -> ConversationManager:
    """Create a conversation manager with a system prompt."""
    return ConversationManager(system_prompt="You are a helpful AI assistant.")


@pytest.fixture
def prompt_builder() -> PromptBuilder:
    """Create a prompt builder."""
    return PromptBuilder()


@pytest.fixture
def service(
    runtime: Mock,
    conversation: ConversationManager,
    prompt_builder: PromptBuilder,
):
    """Create a conversation service instance."""
    from app.services.conversation_service import ConversationService

    return ConversationService(
        runtime=runtime,
        conversation=conversation,
        prompt_builder=prompt_builder,
    )


def test_send_message_returns_generation_result(service, runtime):
    """Return the generation result produced by the runtime."""
    result = service.send_message("Hello!")

    assert isinstance(result, GenerationResult)
    assert result.text == "Hello! How can I help you?"
    runtime.generate.assert_called_once()


def test_send_message_adds_user_and_assistant_messages(service, conversation):
    """Store the user message and successful assistant response."""
    service.send_message("Hello!")

    messages = conversation.get_messages()

    assert [message.role for message in messages] == [
        "system",
        "user",
        "assistant",
    ]
    assert messages[1].content == "Hello!"
    assert messages[2].content == "Hello! How can I help you?"


def test_send_message_builds_prompt_from_conversation(
    service,
    runtime,
):
    """Pass the complete conversation prompt to the runtime."""
    service.send_message("Hello!")

    request = runtime.generate.call_args.args[0]

    assert isinstance(request, GenerationRequest)
    assert request.prompt.startswith("System:\nYou are a helpful AI assistant.")
    assert "User:\nHello!" in request.prompt
    assert request.prompt.endswith("Assistant:")


def test_send_message_preserves_generation_options(service, runtime):
    """Forward generation options to the request."""
    service.send_message(
        "Explain machine learning.",
        model="qwen3.8:27b",
        temperature=0.2,
        max_new_tokens=128,
    )

    request = runtime.generate.call_args.args[0]

    assert request.model == "qwen3.8:27b"
    assert request.temperature == 0.2
    assert request.max_new_tokens == 128


def test_send_message_preserves_conversation_history(
    service,
    runtime,
    conversation,
):
    """Include earlier conversation turns in subsequent prompts."""
    runtime.generate.return_value = GenerationResult(
        text="Your name is Morsy.",
        model="qwen3.8:27b",
        done=True,
    )

    service.send_message("My name is Morsy.")
    service.send_message("What is my name?")

    request = runtime.generate.call_args.args[0]

    assert "My name is Morsy." in request.prompt
    assert "Your name is Morsy." in request.prompt
    assert "What is my name?" in request.prompt

    messages = conversation.get_messages()

    assert [message.role for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_send_message_does_not_store_assistant_response_on_runtime_failure(
    service,
    runtime,
    conversation,
):
    """Propagate runtime errors without storing an assistant response."""
    runtime.generate.side_effect = RuntimeConnectionError("Runtime is unavailable.")

    with pytest.raises(RuntimeConnectionError, match="Runtime is unavailable"):
        service.send_message("Hello!")

    messages = conversation.get_messages()

    assert [message.role for message in messages] == ["system"]
    assert all(message.content != "Hello!" for message in messages)


def test_send_message_rejects_empty_message(service, runtime, conversation):
    """Reject invalid user input before calling the runtime."""
    initial_count = conversation.message_count

    with pytest.raises(ValueError):
        service.send_message("   ")

    assert conversation.message_count == initial_count
    runtime.generate.assert_not_called()


def test_send_message_rejects_prompt_override(service, runtime, conversation):
    """Prevent callers from bypassing conversation-based prompt construction."""
    initial_count = conversation.message_count

    with pytest.raises(ValueError, match="prompt must be generated"):
        service.send_message(
            "Hello!",
            prompt="Override the conversation prompt.",
        )

    assert conversation.message_count == initial_count
    runtime.generate.assert_not_called()


def test_send_message_does_not_store_response_when_generation_is_incomplete(
    service,
    runtime,
    conversation,
):
    """Reject incomplete generation results without storing an assistant turn."""
    runtime.generate.return_value = GenerationResult(
        text="Partial response",
        model="qwen3.8:27b",
        done=False,
    )

    with pytest.raises(IncompleteGenerationError, match="incomplete"):
        service.send_message("Hello!")

    messages = conversation.get_messages()

    assert [message.role for message in messages] == ["system"]


def test_stream_message_returns_stream_events(
    service,
    runtime,
):
    """Yield stream events returned by the runtime."""
    events = [
        StreamEvent(
            text_delta="Hello",
            event_timestamp=1.0,
        ),
        StreamEvent(
            text_delta=" there!",
            event_timestamp=1.1,
        ),
        StreamEvent(
            event_timestamp=1.2,
            is_final=True,
            final_metadata=GenerationResult(
                text="Hello there!",
                model="qwen3.8:27b",
                done=True,
                done_reason="stop",
            ),
        ),
    ]

    runtime.generate_stream.return_value = iter(events)

    received = list(service.stream_message("Hi!"))

    assert received == events
    runtime.generate_stream.assert_called_once()


def test_stream_message_stores_completed_assistant_response(
    service,
    runtime,
    conversation,
):
    """Store the assistant response after successful streaming."""
    events = [
        StreamEvent(
            text_delta="Hello",
            event_timestamp=1.0,
        ),
        StreamEvent(
            text_delta=" there!",
            event_timestamp=1.1,
        ),
        StreamEvent(
            event_timestamp=1.2,
            is_final=True,
            final_metadata=GenerationResult(
                text="Hello there!",
                model="qwen3.8:27b",
                done=True,
                done_reason="stop",
            ),
        ),
    ]

    runtime.generate_stream.return_value = iter(events)

    received = list(service.stream_message("Hi!"))

    assert received[-1].is_final is True

    messages = conversation.get_messages()

    assert [message.role for message in messages] == [
        "system",
        "user",
        "assistant",
    ]
    assert messages[-1].content == "Hello there!"


def test_stream_message_preserves_previous_conversation_history(
    service,
    runtime,
):
    """Include previous turns in subsequent streaming requests."""
    first_events = [
        StreamEvent(
            text_delta="Your name is Morsy.",
            event_timestamp=1.0,
        ),
        StreamEvent(
            event_timestamp=1.1,
            is_final=True,
            final_metadata=GenerationResult(
                text="Your name is Morsy.",
                model="qwen3.8:27b",
                done=True,
            ),
        ),
    ]

    second_events = [
        StreamEvent(
            text_delta="Your name is Morsy.",
            event_timestamp=2.0,
        ),
        StreamEvent(
            event_timestamp=2.1,
            is_final=True,
            final_metadata=GenerationResult(
                text="Your name is Morsy.",
                model="qwen3.8:27b",
                done=True,
            ),
        ),
    ]

    runtime.generate_stream.side_effect = [
        iter(first_events),
        iter(second_events),
    ]

    list(service.stream_message("My name is Morsy."))
    list(service.stream_message("What is my name?"))

    assert runtime.generate_stream.call_count == 2

    request = runtime.generate_stream.call_args.args[0]

    assert "My name is Morsy." in request.prompt
    assert "Your name is Morsy." in request.prompt
    assert "What is my name?" in request.prompt


def test_stream_message_does_not_store_incomplete_response(
    service,
    runtime,
    conversation,
):
    """Do not store an assistant response if the stream is incomplete."""
    runtime.generate_stream.return_value = iter(
        [
            StreamEvent(
                text_delta="Partial response",
                event_timestamp=1.0,
            ),
        ]
    )

    with pytest.raises(RuntimeStreamingError):
        list(service.stream_message("Hello!"))

    messages = conversation.get_messages()

    assert [message.role for message in messages] == ["system"]


def test_stream_message_propagates_runtime_streaming_errors(
    service,
    runtime,
    conversation,
):
    """Propagate runtime streaming errors without storing an assistant response."""

    def failing_stream() -> Iterator[StreamEvent]:
        yield StreamEvent(
            text_delta="Partial response",
            event_timestamp=1.0,
        )
        raise RuntimeStreamingError("Stream interrupted.")

    runtime.generate_stream.return_value = failing_stream()

    with pytest.raises(RuntimeStreamingError, match="Stream interrupted"):
        list(service.stream_message("Hello!"))

    messages = conversation.get_messages()

    assert [message.role for message in messages] == ["system"]


def test_stream_message_rejects_empty_input(
    service,
    runtime,
    conversation,
):
    """Reject empty messages before starting streaming."""
    initial_count = conversation.message_count

    with pytest.raises(ValueError):
        list(service.stream_message("   "))

    assert conversation.message_count == initial_count
    runtime.generate_stream.assert_not_called()


def test_stream_message_rejects_prompt_override(
    service,
    runtime,
    conversation,
):
    """Prevent bypassing conversation-based prompt construction."""
    initial_count = conversation.message_count

    with pytest.raises(ValueError, match="prompt must be generated"):
        list(
            service.stream_message(
                "Hello!",
                prompt="Override the conversation prompt.",
            )
        )

    assert conversation.message_count == initial_count
    runtime.generate_stream.assert_not_called()


def test_stream_message_closes_runtime_iterator_when_cancelled(
    service,
    runtime,
    conversation,
):
    """Close the underlying runtime iterator when the consumer cancels."""

    class ClosableStream:
        def __init__(self):
            self._events = iter(
                [
                    StreamEvent(
                        text_delta="Partial response",
                        event_timestamp=1.0,
                    ),
                    StreamEvent(
                        text_delta=" more text",
                        event_timestamp=1.1,
                    ),
                ]
            )
            self.closed = False

        def __iter__(self):
            return self

        def __next__(self):
            return next(self._events)

        def close(self):
            self.closed = True

    stream = ClosableStream()
    runtime.generate_stream.return_value = stream

    response_stream = service.stream_message("Hello!")

    first_event = next(response_stream)

    assert first_event.text_delta == "Partial response"

    response_stream.close()

    assert stream.closed is True

    messages = conversation.get_messages()

    assert [message.role for message in messages] == ["system"]


def test_stream_message_does_not_store_response_after_unexpected_error(
    service,
    runtime,
    conversation,
):
    """Propagate unexpected stream errors without storing partial output."""

    def failing_stream():
        yield StreamEvent(
            text_delta="Partial response",
            event_timestamp=1.0,
        )
        raise ValueError("Unexpected streaming failure.")

    runtime.generate_stream.return_value = failing_stream()

    with pytest.raises(
        ValueError,
        match="Unexpected streaming failure",
    ):
        list(service.stream_message("Hello!"))

    messages = conversation.get_messages()

    assert [message.role for message in messages] == ["system"]


def test_stream_message_rejects_stream_false(
    service,
    runtime,
    conversation,
):
    """Reject non-streaming options before modifying conversation history."""
    initial_count = conversation.message_count

    with pytest.raises(ValueError, match="requires stream=True"):
        list(
            service.stream_message(
                "Hello!",
                stream=False,
            )
        )

    assert conversation.message_count == initial_count
    runtime.generate_stream.assert_not_called()


def test_stream_message_accepts_explicit_stream_true(
    service,
    runtime,
):
    """Accept an explicit stream=True option without duplicate keyword errors."""
    events = [
        StreamEvent(
            text_delta="Hello!",
            event_timestamp=1.0,
        ),
        StreamEvent(
            event_timestamp=1.1,
            is_final=True,
            final_metadata=GenerationResult(
                text="Hello!",
                model="qwen3.8:27b",
                done=True,
                done_reason="stop",
            ),
        ),
    ]

    runtime.generate_stream.return_value = iter(events)

    received = list(
        service.stream_message(
            "Hi!",
            stream=True,
        )
    )

    assert received == events

    request = runtime.generate_stream.call_args.args[0]

    assert request.stream is True


def test_send_message_uses_context_window_manager(
    runtime,
    conversation,
    prompt_builder,
):
    """Apply context window management before normal generation."""
    from app.context.window_manager import ContextWindowManager
    from app.services.conversation_service import ConversationService

    def word_count(text: str) -> int:
        return len(text.split())

    context_manager = ContextWindowManager(
        prompt_builder=prompt_builder,
        token_counter=word_count,
        safety_margin=0,
    )

    service = ConversationService(
        runtime=runtime,
        conversation=conversation,
        prompt_builder=prompt_builder,
        context_window_manager=context_manager,
    )

    conversation.add_user_message("An old question.")
    conversation.add_assistant_message("An old answer.")

    service.send_message(
        "Explain context windows.",
        num_ctx=40,
        max_new_tokens=10,
    )

    request = runtime.generate.call_args.args[0]

    assert "Explain context windows." in request.prompt
    assert request.num_ctx == 40
    assert request.max_new_tokens == 10


def test_stream_message_uses_context_window_manager(
    runtime,
    conversation,
    prompt_builder,
):
    """Apply context window management before streaming generation."""
    from app.context.window_manager import ContextWindowManager
    from app.services.conversation_service import ConversationService

    def word_count(text: str) -> int:
        return len(text.split())

    context_manager = ContextWindowManager(
        prompt_builder=prompt_builder,
        token_counter=word_count,
        safety_margin=0,
    )

    service = ConversationService(
        runtime=runtime,
        conversation=conversation,
        prompt_builder=prompt_builder,
        context_window_manager=context_manager,
    )

    events = [
        StreamEvent(
            text_delta="Context windows manage prompt size.",
            event_timestamp=1.0,
        ),
        StreamEvent(
            event_timestamp=1.1,
            is_final=True,
            final_metadata=GenerationResult(
                text="Context windows manage prompt size.",
                model="qwen3.8:27b",
                done=True,
                done_reason="stop",
            ),
        ),
    ]

    runtime.generate_stream.return_value = iter(events)

    received = list(
        service.stream_message(
            "Explain context windows.",
            num_ctx=40,
            max_new_tokens=10,
        )
    )

    assert received == events
    runtime.generate_stream.assert_called_once()

    request = runtime.generate_stream.call_args.args[0]

    assert "Explain context windows." in request.prompt
    assert request.num_ctx == 40
    assert request.max_new_tokens == 10
    assert request.stream is True


def test_send_message_uses_runtime_settings_for_context_defaults(
    runtime,
    conversation,
    prompt_builder,
):
    """Use runtime settings as the default context generation options."""
    from unittest.mock import Mock

    from app.config import RuntimeSettings
    from app.services.conversation_service import ConversationService

    context_manager = Mock()
    context_manager.fit_messages.side_effect = (
        lambda *, messages, num_ctx, max_new_tokens: messages
    )

    settings = RuntimeSettings(
        num_ctx=512,
        max_new_tokens=64,
    )

    service = ConversationService(
        runtime=runtime,
        conversation=conversation,
        prompt_builder=prompt_builder,
        context_window_manager=context_manager,
        settings=settings,
    )

    service.send_message("Explain context window management.")

    request = runtime.generate.call_args.args[0]
    context_call = context_manager.fit_messages.call_args

    assert request.num_ctx == 512
    assert request.max_new_tokens == 64

    assert context_call.kwargs["num_ctx"] == 512
    assert context_call.kwargs["max_new_tokens"] == 64


def test_send_message_uses_runtime_defaults_when_options_are_none(
    runtime,
    conversation,
    prompt_builder,
):
    """Use runtime defaults when context options are explicitly None."""
    from unittest.mock import Mock

    from app.config import RuntimeSettings
    from app.services.conversation_service import ConversationService

    context_manager = Mock()
    context_manager.fit_messages.side_effect = (
        lambda *, messages, num_ctx, max_new_tokens: messages
    )

    settings = RuntimeSettings(
        num_ctx=512,
        max_new_tokens=64,
    )

    service = ConversationService(
        runtime=runtime,
        conversation=conversation,
        prompt_builder=prompt_builder,
        context_window_manager=context_manager,
        settings=settings,
    )

    service.send_message(
        "Explain context management.",
        num_ctx=None,
        max_new_tokens=None,
    )

    request = runtime.generate.call_args.args[0]
    context_call = context_manager.fit_messages.call_args

    assert request.num_ctx == 512
    assert request.max_new_tokens == 64

    assert context_call.kwargs["num_ctx"] == 512
    assert context_call.kwargs["max_new_tokens"] == 64


def test_continue_message_merges_text_without_adding_a_user_turn(
    service,
    runtime,
    conversation,
):
    runtime.generate.return_value = GenerationResult(
        text=" continued.",
        model="qwen3.8:27b",
        done=True,
        done_reason="stop",
    )
    runtime.generate.side_effect = [
        GenerationResult(
            text="Initial partial",
            model="qwen3.8:27b",
            done=True,
            done_reason="length",
        ),
        GenerationResult(
            text=" continued.",
            model="qwen3.8:27b",
            done=True,
            done_reason="stop",
        ),
    ]

    service.send_message("Explain AI.")
    assert service.continuation_available is True

    result = service.continue_message()

    messages = conversation.get_messages(include_system=False)
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[0].content == "Explain AI."
    assert messages[1].content == "Initial partial continued."
    assert result.text == "Initial partial continued."
    assert service.continuation_available is False

    request = runtime.generate.call_args.args[0]
    assert "Initial partial" in request.prompt
    assert "Do not repeat or summarize any text already written." in request.prompt



def test_continue_message_removes_echoed_previous_response(
    service,
    runtime,
    conversation,
):
    runtime.generate.side_effect = [
        GenerationResult(
            text="Opening answer. Ending answer.",
            model="qwen3.8:27b",
            done=True,
            done_reason="length",
        ),
        GenerationResult(
            text="Opening answer. Ending answer. Additional details.",
            model="qwen3.8:27b",
            done=True,
            done_reason="stop",
        ),
    ]

    service.send_message("Explain AI.")
    result = service.continue_message()

    assert result.text == "Opening answer. Ending answer. Additional details."
    messages = conversation.get_messages(include_system=False)
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[-1].content == result.text
    assert service.continuation_available is False


def test_continue_message_removes_suffix_prefix_overlap(
    service,
    runtime,
    conversation,
):
    runtime.generate.side_effect = [
        GenerationResult(
            text="The answer ends here",
            model="qwen3.8:27b",
            done=True,
            done_reason="length",
        ),
        GenerationResult(
            text="here and continues.",
            model="qwen3.8:27b",
            done=True,
            done_reason="stop",
        ),
    ]

    service.send_message("Explain AI.")
    result = service.continue_message()

    assert result.text == "The answer ends here and continues."
    assert conversation.get_messages(include_system=False)[-1].content == result.text


def test_continue_message_is_unavailable_after_normal_completion(
    service,
    runtime,
):
    service.send_message("Hello.")

    assert service.continuation_available is False

    with pytest.raises(ValueError, match="no incomplete assistant response"):
        service.continue_message()


def test_continue_message_remains_available_if_continuation_hits_token_limit(
    service,
    runtime,
):
    runtime.generate.side_effect = [
        GenerationResult(
            text="Part one.",
            model="qwen3.8:27b",
            done=True,
            done_reason="length",
        ),
        GenerationResult(
            text=" Part two.",
            model="qwen3.8:27b",
            done=True,
            done_reason="length",
        ),
    ]

    service.send_message("Explain AI.")
    result = service.continue_message()

    assert result.text == "Part one. Part two."
    assert service.continuation_available is True


def test_send_structured_message_returns_validated_data(service, runtime):
    """Return parsed data when the generated JSON matches the schema."""
    runtime.generate.return_value = GenerationResult(
        text='{"answer": "Hello", "confidence": 0.95}',
        model="qwen3.8:27b",
        done=True,
    )

    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["answer", "confidence"],
        "additionalProperties": False,
    }

    result = service.send_structured_message("Say hello.", schema)

    assert result == {"answer": "Hello", "confidence": 0.95}
    request = runtime.generate.call_args.args[0]
    assert request.format == schema


def test_send_structured_message_rejects_invalid_json(service, runtime):
    """Raise a parsing error when the runtime returns malformed JSON."""
    runtime.generate.return_value = GenerationResult(
        text="{invalid json}",
        model="qwen3.8:27b",
        done=True,
    )

    schema = {"type": "object"}

    with pytest.raises(StructuredOutputParsingError):
        service.send_structured_message("Return JSON.", schema)


def test_send_structured_message_rejects_schema_violation(service, runtime):
    """Raise a validation error when generated JSON violates the schema."""
    runtime.generate.return_value = GenerationResult(
        text='{"confidence": 1.5}',
        model="qwen3.8:27b",
        done=True,
    )

    schema = {
        "type": "object",
        "properties": {
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["confidence"],
    }

    with pytest.raises(StructuredOutputValidationError):
        service.send_structured_message("Return confidence.", schema)


def test_send_structured_message_does_not_store_invalid_response(
    service,
    runtime,
    conversation,
):
    """Do not store an assistant response when structured validation fails."""
    runtime.generate.return_value = GenerationResult(
        text='{"confidence": 1.5}',
        model="qwen3.8:27b",
        done=True,
    )

    schema = {
        "type": "object",
        "properties": {
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
        },
        "required": ["confidence"],
    }

    with pytest.raises(StructuredOutputValidationError):
        service.send_structured_message("Return confidence.", schema)

    messages = conversation.get_messages()

    assert [message.role for message in messages] == ["system"]


def test_send_structured_message_does_not_store_incomplete_response(
    service,
    runtime,
    conversation,
):
    """Reject incomplete structured generation without storing an assistant turn."""
    runtime.generate.return_value = GenerationResult(
        text='{"answer": "Partial"}',
        model="qwen3.8:27b",
        done=False,
    )

    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }

    with pytest.raises(IncompleteGenerationError, match="incomplete"):
        service.send_structured_message("Return JSON.", schema)

    messages = conversation.get_messages()

    assert [message.role for message in messages] == ["system"]
