"""Prompt construction for the Local AI Assistant."""

from collections.abc import Sequence

from app.conversation import ConversationMessage
from app.schemas import GenerationRequest


class PromptBuilder:
    """Convert conversation messages into a text prompt and generation request."""

    ROLE_LABELS = {
        "system": "System",
        "user": "User",
        "assistant": "Assistant",
    }

    def build_prompt(
        self,
        messages: Sequence[ConversationMessage],
    ) -> str:
        """Build an ordered text prompt from validated conversation messages."""

        if not messages:
            raise ValueError("Cannot build a prompt from an empty conversation.")

        system_messages = [message for message in messages if message.role == "system"]

        if len(system_messages) > 1:
            raise ValueError("A conversation can contain at most one system message.")

        if system_messages and messages[0].role != "system":
            raise ValueError("The system message must be the first message.")

        conversation_messages = [
            message for message in messages if message.role != "system"
        ]

        if not conversation_messages:
            raise ValueError("A conversation must contain at least one user message.")

        if conversation_messages[-1].role != "user":
            raise ValueError("The latest conversation message must be from the user.")

        sections = [
            (f"{self.ROLE_LABELS[message.role]}:\n{message.content}")
            for message in messages
        ]

        sections.append("Assistant:")

        return "\n\n".join(sections)

    def build_request(
        self,
        messages: Sequence[ConversationMessage],
        **generation_options: object,
    ) -> GenerationRequest:
        """Build a validated generation request from a conversation."""

        if "prompt" in generation_options:
            raise ValueError("The prompt must be generated from conversation messages.")

        payload: dict[str, object] = {
            "prompt": self.build_prompt(messages),
        }

        payload.update(generation_options)

        return GenerationRequest.model_validate(payload)
