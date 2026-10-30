# Local LLM Runtime Engineering

## Overview

Local LLM Runtime Engineering focuses on building a structured runtime integration layer for the Local AI Assistant.

The implementation separates application-level generation requests from the underlying LLM runtime. It provides validated request and response schemas, centralized configuration, an Ollama HTTP adapter, normalized streaming events, runtime error handling, and automated tests.

The goal is to establish a maintainable runtime foundation that can support future assistant orchestration without coupling the application directly to Ollama's HTTP response format.

## Learning Objectives

- Centralized and validated runtime configuration.
- Explicit generation request and response contracts.
- A runtime interface independent of the concrete backend.
- Non-streaming and streaming text generation through Ollama.
- Normalized generation metadata and streaming events.
- Dedicated runtime exceptions.
- Mocked tests and real-runtime smoke tests.
- Separation between conversation orchestration and runtime communication.

## Runtime Architecture

The runtime integration follows this conceptual flow:

```text
Application
    |
    v
GenerationRequest
    |
    v
RuntimeInterface
    |
    v
OllamaRuntime
    |
    v
Ollama HTTP API
    |
    v
Raw Runtime Response
    |
    v
GenerationResult / StreamEvent
```

The conversation orchestration layer connects conversation history, prompt construction, context-window management, and runtime generation.

```text
ConversationManager
    |
    v
ContextWindowManager
    |
    v
PromptBuilder
    |
    v
GenerationRequest
    |
    v
OllamaRuntime
    |
    v
GenerationResult / StreamEvent
    |
    v
ConversationService
```

The context-window manager is optional when constructing `ConversationService`. When configured, it selects messages that fit the available prompt-token budget.

## Project Components

### Centralized Configuration

File: `app/config.py`

`RuntimeSettings` defines validated settings for the runtime and generation process.

| Setting | Default | Description |
|---|---|---|
| `ollama_host` | `http://127.0.0.1:11434` | Ollama base URL |
| `model_name` | `qwen3.8:27b` | Configured model identifier |
| `request_timeout_seconds` | `120.0` | HTTP request timeout |
| `keep_alive` | `5m` | Model retention setting |
| `think` | `False` | Thinking option passed to Ollama |
| `num_ctx` | `4096` | Context-window setting |
| `max_new_tokens` | `16` | Default maximum generated tokens |
| `temperature` | `0.0` | Generation temperature |
| `top_k` | `None` | Optional top-k sampling setting |
| `top_p` | `None` | Optional nucleus sampling setting |
| `seed` | `None` | Optional generation seed |

Configuration values can be loaded from defaults, a `.env` file, and operating-system environment variables.

The precedence order is:

1. Model defaults.
2. Values loaded from `.env`.
3. Operating-system environment variables.

`load_settings()` also supports explicitly supplied environment mappings for deterministic tests. The settings model validates values and rejects unexpected fields.

### Request and Response Schemas

File: `app/schemas.py`

The application defines three main data contracts.

#### GenerationRequest

Represents a validated generation request. Important fields include `prompt`, `model`, `stream`, `max_new_tokens`, `num_ctx`, `temperature`, `top_k`, `top_p`, `seed`, `stop`, and `keep_alive`.

Generation options can override the corresponding configured defaults where supported.

#### GenerationResult

Represents a normalized, non-streaming generation result or final streaming metadata. It includes generated text, model identifier, completion status, optional completion reason, prompt and generated token counts, runtime timing metadata, and client-observed HTTP latency.

Runtime timing values are represented in nanoseconds when reported by Ollama. Client-observed latency is represented in milliseconds.

#### StreamEvent

Represents a normalized streaming event. It includes:

- `text_delta`: the text fragment received.
- `event_timestamp`: a client-observed monotonic timestamp.
- `is_final`: whether the event marks completion.
- `final_metadata`: normalized final generation metadata when available.

### Runtime Interface

File: `app/runtime/base.py`

`RuntimeInterface` defines the expected operations for a runtime adapter:

- `health()`
- `list_models()`
- `generate(request)`
- `generate_stream(request)`

The interface uses the application's request, result, and streaming-event schemas. This contract separates the orchestration layer from the concrete runtime implementation.

### Ollama Runtime Adapter

File: `app/runtime/ollama_runtime.py`

`OllamaRuntime` implements the runtime interface using `httpx`.

| Operation | Ollama endpoint | Purpose |
|---|---|---|
| Health check | `/api/version` | Verify service availability |
| Model inventory | `/api/tags` | Retrieve available model identifiers |
| Non-streaming generation | `/api/generate` | Generate a complete response |
| Streaming generation | `/api/generate` | Receive incremental response events |

The adapter validates HTTP responses and normalizes runtime data into application-level schemas. It supports explicit HTTP-client injection for testing and manages the lifecycle of clients it creates itself.

#### Generation Option Mapping

| Application setting | Ollama option |
|---|---|
| `num_ctx` | `num_ctx` |
| `max_new_tokens` | `num_predict` |
| `temperature` | `temperature` |
| `top_k` | `top_k` |
| `top_p` | `top_p` |
| `seed` | `seed` |
| `stop` | `stop` |

The `max_new_tokens` field is mapped to Ollama's `num_predict` option. The adapter also passes the configured thinking option and model-retention setting.

#### Response Normalization

The non-streaming adapter validates the generated text, model identifier, and completion status. When available, it also extracts:

- `done_reason`
- `prompt_eval_count`
- `eval_count`
- `total_duration`
- `load_duration`
- `prompt_eval_duration`
- `eval_duration`

The adapter measures client-observed request latency separately. Missing optional metadata remains unavailable rather than being replaced with invented values.

### Streaming Generation

Streaming generation uses the same Ollama generation endpoint with streaming enabled.

The adapter reads newline-delimited JSON events, validates their structure, and emits normalized `StreamEvent` objects. Text fragments are accumulated to construct the final generated response.

The final event includes normalized completion metadata. A stream that ends without a final completion event raises a runtime streaming exception.

`ConversationService.stream_message()` consumes these events and stores the assistant response only after successful completion and validation.

### Conversation Management

Files:

- `app/conversation.py`
- `app/prompt_builder.py`
- `app/services/conversation_service.py`

`ConversationManager` maintains an ordered in-memory history of user and assistant messages. It can also maintain a system prompt separately.

Conversation messages are validated and immutable. Clearing the conversation removes conversational turns while preserving the configured system prompt.

`PromptBuilder` converts ordered messages into a text prompt using the labels `System:`, `User:`, and `Assistant:`. It validates the conversation structure and constructs a `GenerationRequest`.

`ConversationService` coordinates conversation history, context management, prompt construction, and generation.

For non-streaming generation, it stores the assistant response only after the runtime reports successful completion.

For streaming generation, it verifies the final metadata and checks that the collected text matches the final result before storing the assistant response.

A failed generation can leave the user message in the conversation history. The context-window manager accounts for incomplete historical turns when selecting messages for subsequent requests.

### Context-Window Management

File: `app/context/window_manager.py`

`ContextWindowManager` selects the most recent complete conversation turns that fit within the available prompt budget.

The prompt budget is calculated as:

`prompt_budget = num_ctx - max_new_tokens - safety_margin`

The manager preserves the system message when configured, the latest user message, and the most recent complete historical turns that fit within the budget. Older complete turns are removed when necessary.

If the system prompt and latest user message cannot fit within the available prompt budget, the manager raises an error rather than silently discarding the latest request.

### Token Counting

File: `app/context/token_counter.py`

`TokenCounter` loads a tokenizer from a local tokenizer JSON file using the `tokenizers` library. It counts tokens by encoding the supplied text and returning the number of token IDs. The counter can be passed to `ContextWindowManager` as a callable.

**Known limitation:** Exact parity between the local tokenizer and the tokenizer behavior used by the configured Ollama model has not been independently established.

## Error Handling

File: `app/runtime/exceptions.py`

| Exception | Purpose |
|---|---|
| `RuntimeErrorBase` | Base exception for runtime integration errors |
| `RuntimeConnectionError` | Runtime service cannot be reached |
| `RuntimeTimeoutError` | Request exceeds the configured timeout |
| `RuntimeHTTPError` | Runtime returns an unsuccessful HTTP status |
| `RuntimeResponseError` | Runtime response is invalid |
| `ModelNotFoundError` | Requested model is unavailable |
| `RuntimeStreamingError` | Streaming cannot be completed correctly |
| `InvalidGenerationRequestError` | Request is invalid for the selected runtime operation |

These exceptions make runtime failures more explicit than relying exclusively on raw HTTP-client exceptions.

## Testing

The test suite includes unit tests, mocked runtime tests, and real-runtime smoke tests.

Relevant test files include:

- `tests/test_config.py`
- `tests/test_generation_result.py`
- `tests/test_runtime_schemas.py`
- `tests/test_runtime_interface.py`
- `tests/test_ollama_runtime_mocked.py`
- `tests/test_ollama_runtime_smoke.py`
- `tests/test_ollama_streaming_smoke.py`
- `tests/test_conversation.py`
- `tests/test_prompt_builder.py`
- `tests/test_conversation_service.py`
- `tests/test_context_window_manager.py`
- `tests/test_token_counter.py`

### Run the Full Test Suite

From the project root:

```powershell
python.exe -m pytest -v
```

### Run Mocked Runtime Tests

```powershell
python.exe -m pytest tests/test_ollama_runtime_mocked.py -v
```

Mocked tests validate adapter behavior without depending on a live Ollama generation request.

### Run the Non-Streaming Integration Smoke Test

```powershell
python.exe -m pytest tests/test_ollama_runtime_smoke.py -v
```

This test checks runtime health, verifies that the configured model is available, performs a bounded real generation request, and validates the normalized result.

### Run the Streaming Integration Smoke Test

```powershell
python.exe -m pytest tests/test_ollama_streaming_smoke.py -v
```

This test checks runtime health and model availability, performs a bounded streaming generation request, and verifies that the generated text matches the final metadata.

The real-runtime smoke tests require a running local Ollama service and the configured model to be available.

### Recorded Test Status

The latest recorded full-suite result for the project was:

- 171 tests passed.
- 0 tests failed.

The non-streaming smoke test also passed in a later standalone run.

Test counts and execution times may change as the project evolves. Re-run the tests to verify the current state.

## Supported Runtime

The implemented and tested runtime adapter is Ollama.

The configured model identifier is `qwen3.8:27b`, subject to local configuration and model availability. The adapter communicates with Ollama through its HTTP API.

The runtime interface is designed to separate application-level generation contracts from backend-specific implementation details. However, an alternative runtime adapter has not been established as a tested implementation in this section.

## Known Limitations

- The project does not yet establish a complete user-facing application entry point.
- No HTTP API or graphical user interface is documented as implemented by this section.
- The local tokenizer's exact parity with Ollama's tokenizer has not been independently verified.
- Raw logits, hidden states, attention tensors, and internal KV-cache tensors are not exposed through the response fields consumed by the current Ollama adapter.
- Direct Transformers inference has not been validated as an alternative runtime.
- Real-runtime smoke tests depend on the local Ollama service and the configured model.
- A dependency manifest was not present in the inspected project root at the time this document was prepared.

## Runtime Capability Boundary

| Capability | Status |
|---|---|
| Ollama health check | Implemented and tested |
| Model inventory | Implemented and tested |
| Non-streaming generation | Implemented and smoke-tested |
| Streaming generation | Implemented and smoke-tested |
| Normalized generation metadata | Implemented |
| Client-observed request latency | Implemented |
| Raw logits through the current adapter | Not exposed |
| Hidden states and attention tensors through the current adapter | Not exposed |
| Internal KV-cache tensors through the current adapter | Not exposed |
| Direct Transformers inference | Not validated |
| Exact local/Ollama tokenizer parity | Not independently verified |

## Outcome

We establishes a structured local runtime integration layer with validated request and response contracts, an Ollama adapter, streaming support, centralized configuration, runtime error handling, conversation orchestration, context-window management, and automated tests.

The implementation creates a foundation for future work on prompt construction, model-specific chat templates, instruction following, structured outputs, and prompt regression testing.

## Next Steps

The next development stage should focus on prompt construction and model-facing input design.

Planned topics include:

- System, user, and assistant roles.
- Model-specific chat templates.
- Instruction-following behavior.
- Structured output generation.
- Prompt regression tests.
