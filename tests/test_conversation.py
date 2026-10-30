"""Tests for the in-memory conversation manager."""

import pytest
from pydantic import ValidationError

from app.conversation import ConversationManager, ConversationMessage


def test_manager_can_be_created_without_system_prompt() -> None:
    manager = ConversationManager()

    assert manager.system_prompt is None
    assert manager.message_count == 0
    assert manager.get_messages() == []


def test_manager_can_be_created_with_system_prompt() -> None:
    manager = ConversationManager(system_prompt="You are a helpful AI assistant.")

    messages = manager.get_messages()

    assert manager.system_prompt == "You are a helpful AI assistant."
    assert len(messages) == 1
    assert messages[0].role == "system"


def test_user_and_assistant_messages_preserve_order() -> None:
    manager = ConversationManager()

    manager.add_user_message("Hello.")
    manager.add_assistant_message("Hello! How can I help?")
    manager.add_user_message("Explain machine learning.")

    messages = manager.get_messages(include_system=False)

    assert [message.role for message in messages] == [
        "user",
        "assistant",
        "user",
    ]

    assert [message.content for message in messages] == [
        "Hello.",
        "Hello! How can I help?",
        "Explain machine learning.",
    ]


def test_replace_last_assistant_message_preserves_message_order() -> None:
    manager = ConversationManager()
    manager.add_user_message("Explain AI.")
    manager.add_assistant_message("The answer is incomplete.")

    replaced = manager.replace_last_assistant_message(
        "The answer is now complete."
    )

    messages = manager.get_messages(include_system=False)
    assert replaced.content == "The answer is now complete."
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[-1].content == "The answer is now complete."
    assert manager.message_count == 2


def test_replace_last_assistant_message_requires_assistant_at_end() -> None:
    manager = ConversationManager()
    manager.add_user_message("A question.")

    with pytest.raises(ValueError):
        manager.replace_last_assistant_message("An answer.")


def test_replace_last_assistant_message_rejects_blank_content() -> None:
    manager = ConversationManager()
    manager.add_assistant_message("Original answer.")

    with pytest.raises(ValidationError):
        manager.replace_last_assistant_message("   ")

    assert manager.get_messages(include_system=False)[-1].content == "Original answer."


def test_message_count_excludes_system_message() -> None:
    manager = ConversationManager(system_prompt="You are a helpful AI assistant.")

    manager.add_user_message("Hello.")
    manager.add_assistant_message("Hi!")

    assert manager.message_count == 2
    assert len(manager.get_messages()) == 3


def test_get_messages_can_exclude_system_message() -> None:
    manager = ConversationManager(system_prompt="You are a helpful AI assistant.")
    manager.add_user_message("Hello.")

    messages = manager.get_messages(include_system=False)

    assert len(messages) == 1
    assert messages[0].role == "user"


def test_system_message_is_returned_first() -> None:
    manager = ConversationManager(system_prompt="Follow the instructions.")

    manager.add_user_message("Question.")
    manager.add_assistant_message("Answer.")

    messages = manager.get_messages()

    assert [message.role for message in messages] == [
        "system",
        "user",
        "assistant",
    ]


def test_get_messages_does_not_expose_internal_list() -> None:
    manager = ConversationManager()
    manager.add_user_message("Original message.")

    returned_messages = manager.get_messages()
    returned_messages.clear()

    assert manager.message_count == 1
    assert len(manager.get_messages(include_system=False)) == 1


def test_conversation_message_is_immutable() -> None:
    message = ConversationMessage(
        role="user",
        content="Original message.",
    )

    with pytest.raises(ValidationError):
        message.content = "Modified message."


@pytest.mark.parametrize(
    "content",
    [
        "",
        " ",
        "\t",
        "\n",
    ],
)
def test_blank_message_content_is_rejected(content: str) -> None:
    with pytest.raises(ValidationError):
        ConversationMessage(
            role="user",
            content=content,
        )


@pytest.mark.parametrize(
    "role",
    [
        "developer",
        "invalid",
        "",
    ],
)
def test_invalid_message_role_is_rejected(role: str) -> None:
    with pytest.raises(ValidationError):
        ConversationMessage(
            role=role,
            content="Test message.",
        )


@pytest.mark.parametrize(
    "content",
    [
        None,
        123,
        ["not", "a", "string"],
    ],
)
def test_non_string_message_content_is_rejected(
    content: object,
) -> None:
    with pytest.raises(ValidationError):
        ConversationMessage(
            role="user",
            content=content,
        )


def test_clear_removes_messages_but_preserves_system_prompt() -> None:
    manager = ConversationManager(system_prompt="You are a helpful AI assistant.")

    manager.add_user_message("Hello.")
    manager.add_assistant_message("Hi!")

    manager.clear()

    assert manager.message_count == 0
    assert manager.system_prompt == "You are a helpful AI assistant."
    assert len(manager.get_messages()) == 1
    assert manager.get_messages()[0].role == "system"


def test_system_prompt_can_be_removed() -> None:
    manager = ConversationManager(system_prompt="Initial instructions.")

    manager.set_system_prompt(None)

    assert manager.system_prompt is None
    assert manager.get_messages() == []


def test_system_prompt_can_be_updated() -> None:
    manager = ConversationManager(system_prompt="Initial instructions.")

    manager.set_system_prompt("Updated instructions.")

    messages = manager.get_messages()

    assert manager.system_prompt == "Updated instructions."
    assert len(messages) == 1
    assert messages[0].content == "Updated instructions."


def test_invalid_user_message_does_not_change_conversation() -> None:
    manager = ConversationManager()
    manager.add_user_message("Valid message.")

    with pytest.raises(ValidationError):
        manager.add_user_message("   ")

    assert manager.message_count == 1
    assert manager.get_messages(include_system=False)[0].content == ("Valid message.")


def test_remove_last_message_if_only_removes_matching_pending_turn():
    conversation = ConversationManager(system_prompt="system")
    conversation.add_user_message("earlier question")
    conversation.add_assistant_message("earlier answer")
    conversation.add_user_message("failed question")

    assert conversation.remove_last_message_if(
        role="user",
        content="different question",
    ) is False
    assert conversation.message_count == 3

    assert conversation.remove_last_message_if(
        role="user",
        content="failed question",
    ) is True
    assert [m.content for m in conversation.get_messages(include_system=False)] == [
        "earlier question",
        "earlier answer",
    ]


def test_conversation_snapshots_remain_independent_from_later_mutations():
    conversation = ConversationManager()
    conversation.add_user_message("first")
    snapshot = conversation.get_messages()
    conversation.add_assistant_message("answer")

    assert [message.content for message in snapshot] == ["first"]
    assert [message.content for message in conversation.get_messages()] == [
        "first",
        "answer",
    ]
