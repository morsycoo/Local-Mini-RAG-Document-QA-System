"""Interactive command-line interface for the Local AI Assistant."""

from __future__ import annotations

import json
from collections.abc import Callable

from app.config import RuntimeSettings, load_settings
from app.conversation import ConversationManager
from app.prompt_builder import PromptBuilder
from app.runtime.exceptions import RuntimeErrorBase
from app.runtime.ollama_runtime import OllamaRuntime
from app.services.conversation_service import ConversationService
from app.structured_output import (
    AssistantStructuredResponse,
    StructuredOutputError,
)


HADOTA_SYSTEM_PROMPT = (
"You are HADOTA, a local AI assistant. The Arabic brand name is حَدُّوتَة. Only introduce yourself when the user explicitly asks who you are or asks about your identity. Otherwise, answer the user's request directly without greetings, self-introductions, or unnecessary preambles. When the user writes in Arabic, reply in natural Arabic, preferably Egyptian Arabic. Answer every part of the user's question. If the user asks multiple questions, address each one clearly instead of answering only the first. For identity questions in Arabic, introduce yourself as: أنا حَدُّوتَة، مساعدك الذكي المحلي. أقدر أساعدك إزاي النهارده؟ Be helpful, honest, concise but complete, and conversational. Never claim to be human or invent capabilities you do not have.")


# Keep the first CLI version independent of external tokenizer files.
def create_conversation_service(
    settings: RuntimeSettings | None = None,
) -> tuple[OllamaRuntime, ConversationService]:
    """Create the runtime and conversation service using project settings."""

    resolved_settings = settings or load_settings()

    runtime = OllamaRuntime(settings=resolved_settings)
    conversation = ConversationManager(system_prompt=HADOTA_SYSTEM_PROMPT)
    prompt_builder = PromptBuilder()

    service = ConversationService(
        runtime=runtime,
        conversation=conversation,
        prompt_builder=prompt_builder,
        settings=resolved_settings,
    )

    return runtime, service


def create_conversation_service_with_runtime(
    runtime: OllamaRuntime,
) -> ConversationService:
    """Create an independent conversation service using a shared runtime."""

    conversation = ConversationManager(system_prompt=HADOTA_SYSTEM_PROMPT)
    prompt_builder = PromptBuilder()

    return ConversationService(
        runtime=runtime,
        conversation=conversation,
        prompt_builder=prompt_builder,
        settings=runtime.settings,
    )


def _print_help() -> None:
    """Display the supported interactive commands."""

    print(
        """
Available commands:
  /help       Show available commands
  /status     Check Ollama connectivity and configured model
  /stats      Show conversation statistics
  /clear      Clear conversation history
  /stream     Toggle streaming responses
  /json       Toggle structured JSON responses
  /exit       Exit the application

Enter a normal message to chat with the assistant.
"""
    )


def _print_status(runtime: OllamaRuntime) -> None:
    """Report runtime connectivity and configured model availability."""

    settings = runtime.settings

    print(f"Ollama host: {settings.ollama_host}")
    print(f"Configured model: {settings.model_name}")

    try:
        if not runtime.health():
            print("Ollama status: UNAVAILABLE")
            return

        print("Ollama status: CONNECTED")

        available_models = runtime.list_models()

        if settings.model_name in available_models:
            print("Configured model status: AVAILABLE")
        else:
            print("Configured model status: NOT FOUND")
            print("Available models:", ", ".join(available_models) or "(none)")

    except RuntimeErrorBase as error:
        print(f"Status check failed: {error}")


def _print_stats(service: ConversationService) -> None:
    """Display basic conversation statistics without claiming token counts."""

    conversation = service.conversation
    messages = conversation.get_messages()

    user_messages = sum(message.role == "user" for message in messages)
    assistant_messages = sum(message.role == "assistant" for message in messages)

    print(f"User messages: {user_messages}")
    print(f"Assistant messages: {assistant_messages}")
    print(f"Total stored messages: {conversation.message_count}")
    print(f"System prompt configured: {conversation.system_prompt is not None}")


def _handle_message(
    service: ConversationService,
    content: str,
    *,
    streaming: bool,
    structured: bool,
) -> None:
    """Generate and display a response using the selected interaction mode."""

    if structured:
        # Use the existing Pydantic model to define the JSON response schema.
        schema = AssistantStructuredResponse.model_json_schema()

        try:
            parsed_response = service.send_structured_message(
                content,
                schema=schema,
                max_new_tokens=256,
            )

        except StructuredOutputError as error:
            # Report the validation failure without exposing unrelated data.
            print(f"Structured output validation failed: {error}")
            print("The generated response was rejected.")
            return

        # Print structured data in a readable, valid JSON representation.
        print(
            json.dumps(
                parsed_response,
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if streaming:
        # Print generated text as it arrives instead of waiting for completion.
        print("Assistant: ", end="", flush=True)

        for event in service.stream_message(content):
            if event.text_delta:
                print(event.text_delta, end="", flush=True)

        print()
        return

    result = service.send_message(content)
    print(f"Assistant: {result.text}")


def run_cli(
    *,
    service_factory: Callable[
        [RuntimeSettings | None],
        tuple[OllamaRuntime, ConversationService],
    ] | None = None,
    input_fn: Callable[[str], str] | None = None,
) -> None:
    """Run the interactive CLI until the user requests an exit."""

    # Resolve dependencies at runtime so tests can replace them reliably.
    if service_factory is None:
        service_factory = create_conversation_service

    if input_fn is None:
        input_fn = input

    runtime: OllamaRuntime | None = None

    try:
        settings = load_settings()
        runtime, service = service_factory(settings)

        streaming = False
        structured = False

        print("Local AI Assistant")
        print(f"Model: {settings.model_name}")
        print("Type /help to see available commands.")

        while True:
            try:
                user_input = input_fn("You: ").strip()

            except EOFError:
                print("\nInput closed. Exiting.")
                break

            if not user_input:
                continue

            command = user_input.lower()

            if command in {"/exit", "/quit"}:
                print("Goodbye!")
                break

            if command == "/help":
                _print_help()
                continue

            if command == "/status":
                _print_status(runtime)
                continue

            if command == "/stats":
                _print_stats(service)
                continue

            if command == "/clear":
                service.conversation.clear()
                print("Conversation history cleared.")
                continue

            if command == "/stream":
                streaming = not streaming
                print(f"Streaming: {'ON' if streaming else 'OFF'}")
                continue

            if command == "/json":
                structured = not structured
                print(f"Structured JSON mode: {'ON' if structured else 'OFF'}")
                continue

            if user_input.startswith("/"):
                print("Unknown command. Type /help to see available commands.")
                continue

            try:
                _handle_message(
                    service,
                    user_input,
                    streaming=streaming,
                    structured=structured,
                )

            except (RuntimeErrorBase, StructuredOutputError, ValueError) as error:
                # Report expected generation and validation failures without
                # terminating the interactive session.
                print(f"Request failed: {error}")

    except RuntimeErrorBase as error:
        print(f"Unable to initialize the assistant: {error}")

    finally:
        # Close the HTTP client only when a runtime was successfully created.
        if runtime is not None:
            runtime.close()


def main() -> None:
    """Provide the executable entry point."""

    run_cli()


if __name__ == "__main__":
    main()
