"""Real integration smoke test for Ollama streaming generation."""

from time import perf_counter

from app.config import load_settings
from app.runtime.ollama_runtime import OllamaRuntime
from app.schemas import GenerationRequest


def test_ollama_streaming_smoke() -> None:
    """Verify streaming generation against the local Ollama service."""

    settings = load_settings()

    with OllamaRuntime(settings) as runtime:
        assert runtime.health() is True

        models = runtime.list_models()
        assert settings.model_name in models

        request = GenerationRequest(
            prompt=(
                "Explain what an LLM runtime does in two short sentences. "
                "Answer directly."
            ),
            stream=True,
            max_new_tokens=96,
        )

        start_time = perf_counter()
        first_chunk_time = None
        text_parts = []
        final_metadata = None

        for event in runtime.generate_stream(request):
            if event.text_delta:
                if first_chunk_time is None:
                    first_chunk_time = perf_counter() - start_time

                text_parts.append(event.text_delta)
                print(event.text_delta, end="", flush=True)

            if event.is_final:
                final_metadata = event.final_metadata

        print()

        generated_text = "".join(text_parts)

        print(f"Model: {settings.model_name}")
        print(f"Generated text: {generated_text}")
        print(f"Time to first text (seconds): {first_chunk_time}")
        print(f"Final metadata: {final_metadata}")

        assert generated_text.strip()
        assert first_chunk_time is not None
        assert final_metadata is not None
        assert final_metadata.done is True
        assert final_metadata.text == generated_text
        assert final_metadata.model == settings.model_name
        assert final_metadata.client_latency_ms is not None
