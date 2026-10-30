import pytest
from pydantic import ValidationError

from app.web.schemas import (
    ConversationResponse,
    CreateConversationRequest,
    MessageResponse,
    SendMessageRequest,
    SessionResponse,
)


def test_create_conversation_request_accepts_optional_title():
    assert CreateConversationRequest().title is None
    assert CreateConversationRequest(title="My chat").title == "My chat"


@pytest.mark.parametrize("title", ["", "   ", "\t"])
def test_create_conversation_request_rejects_blank_title(title):
    with pytest.raises(ValidationError):
        CreateConversationRequest(title=title)


def test_create_conversation_request_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        CreateConversationRequest(unexpected="value")


def test_send_message_request_accepts_valid_content():
    request = SendMessageRequest(content="Hello")
    assert request.content == "Hello"


def test_send_message_request_accepts_maximum_length():
    request = SendMessageRequest(content="x" * 4000)
    assert len(request.content) == 4000


def test_send_message_request_rejects_content_over_limit():
    with pytest.raises(ValidationError):
        SendMessageRequest(content="x" * 4001)


@pytest.mark.parametrize("content", ["", " ", "\t", "\n"])
def test_send_message_request_rejects_blank_content(content):
    with pytest.raises(ValidationError):
        SendMessageRequest(content=content)


@pytest.mark.parametrize("content", [None, 123, True, ["hello"], {"text": "hello"}])
def test_send_message_request_rejects_non_string_content(content):
    with pytest.raises(ValidationError):
        SendMessageRequest(content=content)


def test_send_message_request_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        SendMessageRequest(content="Hello", prompt="override")


def test_session_response_defaults_to_ok():
    assert SessionResponse().model_dump() == {"status": "ok"}


def test_conversation_response_contains_public_fields():
    response = ConversationResponse(
        conversation_id="conversation-123",
        title="My chat",
    )
    assert response.model_dump() == {
        "conversation_id": "conversation-123",
        "title": "My chat",
    }


def test_message_response_defaults_to_completed_assistant_message():
    response = MessageResponse(
        conversation_id="conversation-123",
        content="Hello",
    )
    assert response.model_dump() == {
        "conversation_id": "conversation-123",
        "role": "assistant",
        "content": "Hello",
        "title": None,
        "done": True,
        "continuation_available": False,
    }
