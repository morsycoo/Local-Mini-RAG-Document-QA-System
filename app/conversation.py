"""Conversation message models and in-memory conversation management."""

from threading import RLock
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MessageRole = Literal["system", "user", "assistant"]

ConversationTurnRole = Literal["user", "assistant"]


class ConversationMessage(BaseModel):
    """Validated and immutable message stored by the conversation manager."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    role: MessageRole = Field(
        description="The role associated with the conversation message.",
    )

    content: str = Field(
        min_length=1,
        description="The textual content of the conversation message.",
    )

    @field_validator("content", mode="before")
    @classmethod
    def validate_content(cls, value: object) -> str:
        """Require non-empty string content while preserving its original text."""

        if not isinstance(value, str):
            raise ValueError("Message content must be a string.")

        if not value.strip():
            raise ValueError("Message content must not be empty or whitespace-only.")

        return value


class ConversationManager:
    """Manage an ordered conversation history in memory.

    The system prompt is stored separately from conversational turns.
    Clearing the conversation removes user and assistant messages but
    preserves the configured system prompt.
    """

    def __init__(self, system_prompt: str | None = None) -> None:
        self._lock = RLock()
        self._system_message: ConversationMessage | None = None
        self._messages: list[ConversationMessage] = []

        if system_prompt is not None:
            self.set_system_prompt(system_prompt)

    @property
    def system_prompt(self) -> str | None:
        """Return the configured system prompt, if one exists."""

        with self._lock:
            if self._system_message is None:
                return None
            return self._system_message.content

    @property
    def message_count(self) -> int:
        """Return the number of stored user and assistant messages."""

        with self._lock:
            return len(self._messages)

    def set_system_prompt(self, content: str | None) -> None:
        """Set or remove the system prompt."""

        with self._lock:
            if content is None:
                self._system_message = None
                return

            self._system_message = ConversationMessage(
                role="system",
                content=content,
            )

    def add_user_message(self, content: str) -> ConversationMessage:
        """Append a validated user message to the conversation."""

        with self._lock:
            return self._add_message("user", content)

    def add_assistant_message(self, content: str) -> ConversationMessage:
        """Append a validated assistant message to the conversation."""

        with self._lock:
            return self._add_message("assistant", content)

    def replace_last_assistant_message(
        self,
        content: str,
    ) -> ConversationMessage:
        """Replace the most recent assistant message."""

        with self._lock:
            if not self._messages or self._messages[-1].role != "assistant":
                raise ValueError("The last conversation message is not an assistant message.")

            message = ConversationMessage(role="assistant", content=content)
            self._messages[-1] = message
            return message

    def remove_last_message_if(
        self,
        *,
        role: ConversationTurnRole,
        content: str,
    ) -> bool:
        """Remove the last message only when it matches the expected turn.

        This is used to roll back a failed, not-yet-committed generation without
        disturbing earlier conversation history.
        """
        with self._lock:
            if not self._messages:
                return False
            last = self._messages[-1]
            if last.role != role or last.content != content:
                return False
            self._messages.pop()
            return True

    def get_messages(
        self,
        *,
        include_system: bool = True,
    ) -> list[ConversationMessage]:
        """Return messages in order without exposing the internal list.

        If requested and configured, the system message appears first.
        Returned message objects are immutable.
        """

        with self._lock:
            messages = list(self._messages)
            system_message = self._system_message

        if include_system and system_message is not None:
            return [system_message, *messages]

        return messages

    def clear(self) -> None:
        """Clear conversational turns while preserving the system prompt."""

        with self._lock:
            self._messages.clear()

    def _add_message(
        self,
        role: ConversationTurnRole,
        content: str,
    ) -> ConversationMessage:
        """Validate and append a user or assistant message."""

        message = ConversationMessage(
            role=role,
            content=content,
        )

        self._messages.append(message)

        return message
