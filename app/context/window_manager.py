from collections.abc import Callable, Sequence

from app.conversation import ConversationMessage
from app.prompt_builder import PromptBuilder


class ContextWindowManager:
    """Select conversation messages that fit within a context window."""

    def __init__(
        self,
        prompt_builder: PromptBuilder,
        token_counter: Callable[[str], int],
        safety_margin: int = 32,
    ) -> None:
        if not callable(token_counter):
            raise TypeError("token_counter must be callable.")

        if isinstance(safety_margin, bool) or not isinstance(safety_margin, int):
            raise TypeError("safety_margin must be an integer.")

        if safety_margin < 0:
            raise ValueError("safety_margin cannot be negative.")

        self.prompt_builder = prompt_builder
        self.token_counter = token_counter
        self.safety_margin = safety_margin

    def fit_messages(
        self,
        messages: Sequence[ConversationMessage],
        num_ctx: int,
        max_new_tokens: int,
    ) -> list[ConversationMessage]:
        """Keep the most recent complete conversation turns that fit."""

        if not messages:
            raise ValueError("Message history cannot be empty.")

        if isinstance(num_ctx, bool) or not isinstance(num_ctx, int):
            raise TypeError("num_ctx must be an integer.")

        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise TypeError("max_new_tokens must be an integer.")

        if num_ctx <= 0:
            raise ValueError("num_ctx must be positive.")

        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")

        prompt_budget = num_ctx - max_new_tokens - self.safety_margin

        if prompt_budget <= 0:
            raise ValueError(
                "No prompt budget remains after reserving output tokens "
                "and the safety margin."
            )

        # Validate system message placement before processing history.
        system_messages = [message for message in messages if message.role == "system"]

        if len(system_messages) > 1:
            raise ValueError("Only one system message is allowed.")

        if system_messages and messages[0].role != "system":
            raise ValueError("The system message must be the first message.")

        conversation_messages = [
            message for message in messages if message.role != "system"
        ]

        if not conversation_messages:
            raise ValueError("At least one conversation message is required.")

        # The prompt builder requires the latest conversational
        # message to be a user message.
        if conversation_messages[-1].role != "user":
            raise ValueError("The latest conversation message must be from the user.")

        # Preserve complete historical turns and the latest user message.
        # Ignore incomplete older turns left by failed generation attempts.
        latest_user_message = conversation_messages[-1]
        historical_messages = conversation_messages[:-1]

        turns: list[list[ConversationMessage]] = []
        pending_user_message: ConversationMessage | None = None

        for message in historical_messages:
            if message.role == "user":
                # Keep the latest unanswered user message.
                # Older unanswered requests may result from failed generations.
                pending_user_message = message

            elif message.role == "assistant":
                if pending_user_message is None:
                    raise ValueError("An assistant message must follow a user message.")

                turns.append([pending_user_message, message])
                pending_user_message = None

        while True:
            selected = (
                system_messages
                + [message for turn in turns for message in turn]
                + [latest_user_message]
            )

            prompt = self.prompt_builder.build_prompt(selected)
            token_count = self.token_counter(prompt)

            if isinstance(token_count, bool) or not isinstance(token_count, int):
                raise TypeError("token_counter must return an integer.")

            if token_count < 0:
                raise ValueError("token_counter cannot return a negative value.")

            if token_count <= prompt_budget:
                return selected

            # Remove one complete, oldest turn on every iteration.
            if turns:
                turns.pop(0)
                continue

            # The system prompt and latest user message alone do not fit.
            raise ValueError(
                "The system prompt and latest user message exceed "
                "the available prompt budget."
            )
