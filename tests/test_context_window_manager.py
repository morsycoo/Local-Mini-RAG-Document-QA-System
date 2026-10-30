import pytest

from app.context.window_manager import ContextWindowManager
from app.conversation import ConversationManager, ConversationMessage
from app.prompt_builder import PromptBuilder


def word_count(text: str) -> int:
    """Provide a deterministic token-count approximation for testing."""
    return len(text.split())


def create_manager() -> ContextWindowManager:
    """Create a context window manager with a deterministic test counter."""
    return ContextWindowManager(
        prompt_builder=PromptBuilder(),
        token_counter=word_count,
        safety_margin=0,
    )


def create_conversation() -> ConversationManager:
    """Create a conversation with a system prompt and multiple turns."""
    conversation = ConversationManager(system_prompt="You are a helpful assistant.")

    conversation.add_user_message("First question about Python.")
    conversation.add_assistant_message("First answer about Python.")

    conversation.add_user_message("Second question about machine learning.")
    conversation.add_assistant_message("Second answer about machine learning.")

    conversation.add_user_message("What is a context window?")

    return conversation


def test_preserves_system_prompt():
    manager = create_manager()
    conversation = create_conversation()

    selected = manager.fit_messages(
        messages=conversation.get_messages(),
        num_ctx=100,
        max_new_tokens=20,
    )

    assert selected[0].role == "system"
    assert selected[0].content == "You are a helpful assistant."


def test_preserves_latest_user_message():
    manager = create_manager()
    conversation = create_conversation()

    selected = manager.fit_messages(
        messages=conversation.get_messages(),
        num_ctx=100,
        max_new_tokens=20,
    )

    assert selected[-1].role == "user"
    assert selected[-1].content == "What is a context window?"


def test_removes_oldest_complete_turns_first():
    manager = create_manager()
    conversation = create_conversation()

    selected = manager.fit_messages(
        messages=conversation.get_messages(),
        num_ctx=30,
        max_new_tokens=10,
    )

    contents = [message.content for message in selected]

    assert "You are a helpful assistant." in contents
    assert "What is a context window?" in contents

    assert "First question about Python." not in contents
    assert "First answer about Python." not in contents


def test_keeps_recent_turns_in_original_order():
    manager = create_manager()
    conversation = create_conversation()

    selected = manager.fit_messages(
        messages=conversation.get_messages(),
        num_ctx=40,
        max_new_tokens=10,
    )

    roles = [message.role for message in selected]

    assert roles == [
        "system",
        "user",
        "assistant",
        "user",
    ]

    assert selected[-3].content == ("Second question about machine learning.")
    assert selected[-2].content == ("Second answer about machine learning.")
    assert selected[-1].content == "What is a context window?"


def test_does_not_modify_original_conversation():
    manager = create_manager()
    conversation = create_conversation()

    original_messages = conversation.get_messages()

    manager.fit_messages(
        messages=original_messages,
        num_ctx=30,
        max_new_tokens=10,
    )

    assert conversation.get_messages() == original_messages


def test_raises_when_latest_message_cannot_fit():
    manager = create_manager()

    messages = [
        ConversationMessage(
            role="system",
            content="You are a helpful assistant.",
        ),
        ConversationMessage(
            role="user",
            content="Explain an extremely complicated topic in detail.",
        ),
    ]

    with pytest.raises(ValueError):
        manager.fit_messages(
            messages=messages,
            num_ctx=8,
            max_new_tokens=4,
        )


def test_reserves_tokens_for_generation():
    manager = create_manager()

    messages = [
        ConversationMessage(
            role="user",
            content="Explain context windows.",
        ),
    ]

    selected = manager.fit_messages(
        messages=messages,
        num_ctx=20,
        max_new_tokens=5,
    )

    prompt = PromptBuilder().build_prompt(selected)

    assert word_count(prompt) <= 15


def test_rejects_invalid_context_configuration():
    manager = create_manager()

    messages = [
        ConversationMessage(
            role="user",
            content="Explain context windows.",
        ),
    ]

    with pytest.raises(ValueError):
        manager.fit_messages(
            messages=messages,
            num_ctx=10,
            max_new_tokens=10,
        )


def test_rejects_empty_message_history():
    manager = create_manager()

    try:
        manager.fit_messages(
            messages=[],
            num_ctx=100,
            max_new_tokens=20,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for an empty message history.")


def test_rejects_history_ending_with_assistant_message():
    manager = create_manager()

    messages = [
        ConversationMessage(
            role="user",
            content="Explain context windows.",
        ),
        ConversationMessage(
            role="assistant",
            content="A context window limits the tokens processed by a model.",
        ),
    ]

    with pytest.raises(ValueError):
        manager.fit_messages(
            messages=messages,
            num_ctx=100,
            max_new_tokens=20,
        )


def test_rejects_system_message_in_middle_of_history():
    manager = create_manager()

    messages = [
        ConversationMessage(
            role="user",
            content="First question.",
        ),
        ConversationMessage(
            role="assistant",
            content="First answer.",
        ),
        ConversationMessage(
            role="system",
            content="Unexpected system message.",
        ),
        ConversationMessage(
            role="user",
            content="Second question.",
        ),
    ]

    with pytest.raises(ValueError):
        manager.fit_messages(
            messages=messages,
            num_ctx=100,
            max_new_tokens=20,
        )


def test_rejects_multiple_system_messages():
    manager = create_manager()

    messages = [
        ConversationMessage(
            role="system",
            content="First system prompt.",
        ),
        ConversationMessage(
            role="system",
            content="Second system prompt.",
        ),
        ConversationMessage(
            role="user",
            content="Explain context windows.",
        ),
    ]

    with pytest.raises(ValueError):
        manager.fit_messages(
            messages=messages,
            num_ctx=100,
            max_new_tokens=20,
        )


def test_ignores_unanswered_older_user_messages():
    """Ignore unanswered older user messages when selecting context."""
    manager = create_manager()

    messages = [
        ConversationMessage(
            role="system",
            content="You are a helpful assistant.",
        ),
        ConversationMessage(
            role="user",
            content="An earlier request that failed.",
        ),
        ConversationMessage(
            role="user",
            content="What is a context window?",
        ),
    ]

    selected = manager.fit_messages(
        messages=messages,
        num_ctx=100,
        max_new_tokens=20,
    )

    assert [message.role for message in selected] == [
        "system",
        "user",
    ]

    assert [message.content for message in selected] == [
        "You are a helpful assistant.",
        "What is a context window?",
    ]
