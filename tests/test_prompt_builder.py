"""Tests for the prompt builder."""

import pytest
from pydantic import ValidationError

from app.conversation import ConversationManager, ConversationMessage
from app.prompt_builder import PromptBuilder


@pytest.fixture
def builder() -> PromptBuilder:
    return PromptBuilder()


@pytest.fixture
def conversation() -> ConversationManager:
    manager = ConversationManager(system_prompt="You are a helpful AI assistant.")
    manager.add_user_message("My name is Morsy.")
    manager.add_assistant_message("Nice to meet you, Morsy!")
    manager.add_user_message("What is my name?")
    return manager


def test_build_prompt_preserves_message_order(
    builder: PromptBuilder,
    conversation: ConversationManager,
) -> None:
    prompt = builder.build_prompt(conversation.get_messages())

    assert prompt.index("System:") < prompt.index("User:")
    assert prompt.index("My name is Morsy.") < prompt.index("Nice to meet you, Morsy!")
    assert prompt.index("Nice to meet you, Morsy!") < prompt.index("What is my name?")


def test_build_prompt_includes_system_message(
    builder: PromptBuilder,
    conversation: ConversationManager,
) -> None:
    prompt = builder.build_prompt(conversation.get_messages())

    assert prompt.startswith("System:\nYou are a helpful AI assistant.")


def test_build_prompt_ends_with_assistant_prefix(
    builder: PromptBuilder,
    conversation: ConversationManager,
) -> None:
    prompt = builder.build_prompt(conversation.get_messages())

    assert prompt.endswith("Assistant:")


def test_build_prompt_separates_messages_with_blank_lines(
    builder: PromptBuilder,
    conversation: ConversationManager,
) -> None:
    prompt = builder.build_prompt(conversation.get_messages())

    assert "\n\nUser:\nMy name is Morsy." in prompt


def test_build_prompt_without_system_message(
    builder: PromptBuilder,
) -> None:
    manager = ConversationManager()
    manager.add_user_message("Hello.")

    prompt = builder.build_prompt(manager.get_messages(include_system=False))

    assert prompt.startswith("User:\nHello.")
    assert prompt.endswith("Assistant:")
    assert "System:" not in prompt


def test_build_prompt_rejects_empty_conversation(
    builder: PromptBuilder,
) -> None:
    with pytest.raises(ValueError, match="empty conversation"):
        builder.build_prompt([])


def test_build_prompt_rejects_multiple_system_messages(
    builder: PromptBuilder,
) -> None:
    messages = [
        ConversationMessage(
            role="system",
            content="First system message.",
        ),
        ConversationMessage(
            role="system",
            content="Second system message.",
        ),
        ConversationMessage(
            role="user",
            content="Hello.",
        ),
    ]

    with pytest.raises(ValueError, match="at most one system message"):
        builder.build_prompt(messages)


def test_build_prompt_requires_system_message_first(
    builder: PromptBuilder,
) -> None:
    messages = [
        ConversationMessage(
            role="user",
            content="Hello.",
        ),
        ConversationMessage(
            role="system",
            content="System instructions.",
        ),
    ]

    with pytest.raises(ValueError, match="must be the first message"):
        builder.build_prompt(messages)


def test_build_prompt_requires_at_least_one_user_message(
    builder: PromptBuilder,
) -> None:
    messages = [
        ConversationMessage(
            role="system",
            content="System instructions.",
        ),
    ]

    with pytest.raises(ValueError, match="at least one user message"):
        builder.build_prompt(messages)


def test_build_prompt_requires_latest_message_from_user(
    builder: PromptBuilder,
) -> None:
    messages = [
        ConversationMessage(
            role="user",
            content="Hello.",
        ),
        ConversationMessage(
            role="assistant",
            content="Hello!",
        ),
    ]

    with pytest.raises(ValueError, match="latest conversation message"):
        builder.build_prompt(messages)


def test_build_request_creates_valid_generation_request(
    builder: PromptBuilder,
    conversation: ConversationManager,
) -> None:
    request = builder.build_request(
        conversation.get_messages(),
        stream=True,
        temperature=0.2,
        max_new_tokens=128,
    )

    assert request.prompt.endswith("Assistant:")
    assert request.stream is True
    assert request.temperature == 0.2
    assert request.max_new_tokens == 128


def test_build_request_preserves_generation_options(
    builder: PromptBuilder,
    conversation: ConversationManager,
) -> None:
    request = builder.build_request(
        conversation.get_messages(),
        model="qwen3.8:27b",
        stream=False,
        temperature=0.5,
        top_k=20,
        top_p=0.9,
        seed=42,
    )

    assert request.model == "qwen3.8:27b"
    assert request.stream is False
    assert request.temperature == 0.5
    assert request.top_k == 20
    assert request.top_p == 0.9
    assert request.seed == 42


def test_build_request_rejects_direct_prompt_override(
    builder: PromptBuilder,
    conversation: ConversationManager,
) -> None:
    with pytest.raises(ValueError, match="prompt must be generated"):
        builder.build_request(
            conversation.get_messages(),
            prompt="Unrelated prompt",
        )


def test_build_request_rejects_invalid_generation_options(
    builder: PromptBuilder,
    conversation: ConversationManager,
) -> None:
    with pytest.raises(ValidationError):
        builder.build_request(
            conversation.get_messages(),
            temperature=5.0,
        )


def test_build_request_rejects_invalid_stream_type(
    builder: PromptBuilder,
    conversation: ConversationManager,
) -> None:
    with pytest.raises(ValidationError):
        builder.build_request(
            conversation.get_messages(),
            stream="true",
        )
