# End-to-End Local AI Assistant

**Project:** Local AI Assistant  
**Primary notebook:** `notebooks/05_end_to_end_local_ai_assistant.ipynb`  
**Application:** Python CLI backed by a local Ollama runtime  
**Tested model:** `qwen3.8:27b`  
**Runtime endpoint:** `http://127.0.0.1:11434`  
**Recorded environment:** Python 3.13.0, Ollama 0.40.4, Pydantic 2.13.4, `jsonschema` 4.26.0, pytest 9.0.3

---

## 1. Overview

We focused on integrating the Local AI Assistant components developed in earlier work into one usable, end-to-end application. The work reused the existing configuration, conversation manager, prompt builder, context-window manager, runtime adapter, and structured-output validation code rather than rebuilding those components.

The resulting application supports:

- Interactive terminal chat with multi-turn conversation history.
- Non-streaming and streaming generation.
- Structured responses using JSON or JSON Schema, followed by application-level validation.
- Runtime health and model-availability checks.
- Conversation statistics and conversation clearing.
- Request and generation performance observations.
- Normalized handling of runtime failures and invalid responses.
- Automated unit, mocked-runtime, and real-Ollama smoke tests.

The work also included a focused audit of the current prompt-construction path, end-to-end experiments, a full test-suite run, performance documentation, and a final notebook integrity review.

> **Evidence note:** This README records observed implementation details and test results from the work. Single-request measurements are experimental observations, not guaranteed performance values or a comprehensive benchmark.

## 2. Goals and Scope

The primary goal was integration, not another isolated implementation of an LLM component.

The work aimed to:

1. Reuse the existing application modules and maintain separation of responsibilities.
2. Expose normal chat, streaming, and structured output through the CLI.
3. preserve conversation history only when the relevant generation workflow succeeds.
4. inspect the exact application-level prompt sent to Ollama and decide whether the current format should be retained.
5. verify runtime health, configured model availability, request metadata, and performance observations.
6. exercise end-to-end behavior with real local-model requests.
7. verify failure handling through deterministic mocked tests and service-level tests.
8. document the actual results, limitations, and potential next steps.

The scope does not include building a GUI, adding persistent database-backed memory, or replacing Ollama with a different inference runtime.

## 3. Environment and Configuration

### 3.1 Recorded Environment

| Item | Recorded value |
|---|---|
| Python | `3.13.0` |
| Ollama | `0.40.4` |
| Local model | `qwen3.8:27b` |
| Ollama host | `http://127.0.0.1:11434` |
| Pydantic | `2.13.4` |
| `jsonschema` | `4.26.0` |
| pytest | `9.0.3` |
| Configured context window | `4096` tokens |
| Runtime request timeout | `120` seconds |
| Default `max_new_tokens` | `16` |
| Default temperature | `0.0` |
| `keep_alive` | `5m` |
| `top_k`, `top_p`, and seed | Not set in the recorded defaults |
| Thinking option | Disabled in the recorded configuration |

These are the values recorded during the notebook work; they are not intended to imply that every machine or deployment uses the same configuration.

### 3.2 Configuration Checks

The notebook inspected the runtime settings and checked that the project root, application directory, and test directory were available. It then used the existing runtime adapter to:

- query Ollama health;
- retrieve the model inventory;
- check whether the configured model identifier was present;
- print the relevant connection and availability status.

The recorded connectivity and model-availability checks passed. The configured `qwen3.8:27b` model appeared in the returned inventory.

## 4. Application Architecture

The application keeps interactive I/O, conversation state, prompt construction, runtime execution, and response validation in distinct components.

```text
User
  │
  ▼
Interactive CLI (app/cli.py)
  │
  ├── Commands: help, status, stats, clear, stream, json, exit
  │
  ▼
ConversationService
  │
  ├── ConversationManager ── stores ordered conversation turns
  ├── ContextWindowManager ── keeps history within the context budget
  ├── PromptBuilder ───────── formats the message history
  │
  ▼
OllamaRuntime
  │
  ├── Non-streaming generation
  ├── Streaming generation
  ├── Runtime status and model inventory
  └── Runtime metadata and normalized errors
  │
  ▼
Local Ollama runtime / qwen3.8:27b
  │
  ▼
Response handling
  ├── Verify generation completion
  ├── Parse and validate structured output when requested
  └── Store only successful assistant responses
```

### 4.1 Main Application Components

| Component | File | Responsibility |
|---|---|---|
| CLI | `app/cli.py` | Interactive input/output, command handling, and lifecycle management |
| Configuration | `app/config.py` | Runtime settings and configuration validation |
| Conversation manager | `app/conversation.py` | Ordered message history, system prompt, counts, and clearing behavior |
| Prompt builder | `app/prompt_builder.py` | Role-labelled prompt construction and generation-request creation |
| Runtime schemas | `app/schemas.py` | Typed generation-request and result contracts |
| Structured output | `app/structured_output.py` | JSON parsing, JSON Schema validation, and Pydantic response validation |
| Context token counter | `app/context/token_counter.py` | Local token counting used for context budgeting |
| Context window manager | `app/context/window_manager.py` | Context-budget management and conversation truncation policy |
| Runtime interface | `app/runtime/base.py` | Runtime interface contract |
| Runtime exceptions | `app/runtime/exceptions.py` | Normalized runtime exception types |
| Ollama adapter | `app/runtime/ollama_runtime.py` | HTTP communication, payload construction, streaming, and metadata handling |
| Conversation service | `app/services/conversation_service.py` | Coordinates conversation state, prompt building, context management, generation, and response persistence |

### 4.2 Separation of Responsibilities

The CLI does not build prompts or make its own raw generation requests. It sends user input through the conversation service, which uses the existing prompt builder and runtime adapter. The structured-output path also reuses the project’s parsing and validation utilities.

This design keeps terminal interaction separate from model execution and makes most behavior testable with fake or mocked runtime dependencies.

## 5. Interactive CLI

The CLI provides an interactive terminal loop and routes commands separately from ordinary user messages.

### 5.1 Supported Commands

| Command | Behavior |
|---|---|
| `/help` | Displays available commands and interaction guidance |
| `/status` | Reports the configured runtime/model status using the existing runtime checks |
| `/stats` | Displays conversation message counts and related available statistics |
| `/clear` | Clears the stored conversation turns according to the conversation manager’s contract |
| `/stream` | Toggles streaming mode on or off |
| `/json` | Toggles structured JSON response mode on or off |
| `/exit` | Exits the interactive session and closes runtime resources |

The recorded command set includes `/clear`; the notebook does not establish a separate `/new` command. `/stream` and `/json` are mode toggles: after enabling a mode, the next ordinary prompt is processed through that mode.

### 5.2 Conversation Statistics

The CLI statistics view reports user and assistant message counts and the total stored conversation messages. The system prompt is not counted as a user or assistant turn. In the recorded multi-turn test, two user messages and two assistant messages produced four stored conversation messages.

### 5.3 Lifecycle and Error Handling

The CLI includes tests for help and exit behavior, status, statistics and clearing, normal chat dispatch, streaming-mode dispatch, structured-mode dispatch, runtime-error recovery, and EOF shutdown behavior. The CLI test suite also checks that a handled runtime error does not unexpectedly terminate the interactive session.

## 6. Normal Chat Integration

Normal chat is routed through `ConversationService.send_message()` rather than directly calling the Ollama API from the CLI.

The normal request lifecycle is:

1. Validate the user message.
2. Add the user message to the conversation.
3. Apply context-window management.
4. Build the prompt using the existing `PromptBuilder`.
5. Build and send a `GenerationRequest` through the runtime adapter.
6. Inspect the runtime completion status.
7. Store the assistant response only after successful, complete generation.
8. Return the result for display and metric inspection.

### 6.1 Conversation-History Experiment

The end-to-end experiment sent an initial message containing a fictional project code and configuration label, then asked for the label in a follow-up message. The model returned `BLUE-LANTERN` for the project code `ORION-731`.

The experiment observed:

- the initial request completed;
- the follow-up request completed;
- the expected label was recovered;
- both user and assistant messages were retained;
- the conversation contained four stored non-system messages after the two-turn interaction.

This is a successful context-retention example, not a guarantee that all facts will be retained under every context length or prompt.

### 6.2 Output-Budget Behavior

The configured default `max_new_tokens` value of `16` produced a visibly truncated answer in an early normal-chat experiment. The runtime reported `done_reason: length`, which indicated that the generation limit had been reached.

A later controlled request with an explicit budget of `128` produced a more complete response and ended with `done_reason: stop`. This demonstrated the practical importance of choosing an output budget appropriate to the requested answer length.

## 7. Streaming Generation

Streaming uses the runtime’s streaming API and the service’s `stream_message()` workflow. The CLI displays text incrementally rather than waiting for the entire response.

### 7.1 Streaming Workflow

1. Enable streaming mode with `/stream`.
2. Submit a normal user prompt.
3. Receive stream events from `OllamaRuntime.generate_stream()` through `ConversationService.stream_message()`.
4. display text fragments in order while retaining the final result metadata.
5. Verify that the final event was received and that the concatenated text matches the final response.
6. Store the assistant response only when the stream completes successfully.
7. Close the iterator during cleanup or cancellation.

The service tests cover successful streams, completed assistant-response storage, preserved previous history, incomplete streams, runtime streaming errors, unexpected errors, cancellation cleanup, prompt override rejection, and invalid stream-option behavior.

### 7.2 Short Streaming Measurements

Two recorded short-response experiments each generated a 291-character answer and reported 52 stream events. The following values came from separate runs:

| Metric | Recorded run A | Recorded run B |
|---|---:|---:|
| Generated characters | 291 | 291 |
| Stream events | 52 | 52 |
| Time to first text | 9,403.10 ms | 8,069.76 ms |
| Total client-side latency | 20,245.98 ms | 18,170.03 ms |
| Completion reason | `stop` | `stop` |

Stream events are not guaranteed to map one-to-one to model tokens; therefore, event count is reported as event count, not token count.

### 7.3 Long-Response Completion Experiment

A separate five-step neural-network explanation was run with two output budgets. The prompt and the rest of the configuration were held constant while `max_new_tokens` changed.

| Metric | Run A | Run B |
|---|---:|---:|
| Maximum output tokens | 256 | 512 |
| Total stream events | 252 | 300 |
| Text events | 251 | 299 |
| Generated characters | 1,292 | 1,532 |
| Time to first text | 9,320.94 ms | 10,167.53 ms |
| Total stream time | 52,112.96 ms | 59,689.53 ms |
| Final event received | Yes | Yes |
| Runtime `done` status | `True` | `True` |
| Finish reason | `length` | `stop` |
| Five-step answer | Incomplete fifth step | All five steps present |

The first run reached the configured output limit and ended with an incomplete fifth step. The second run completed all five steps and ended with `stop`.

This experiment shows why runtime completion metadata and semantic completeness should be examined separately. It does **not** show that the larger output budget is always better; it shows that this particular longer request needed more output capacity to finish.

### 7.4 Streaming Versus Non-Streaming

Streaming improves response visibility because partial text becomes available before generation finishes. The recorded experiments do not establish that streaming is universally faster in total latency. Total time depends on output length, model work, prompt size, local system load, and runtime conditions.

## 8. Structured Output Integration

Structured output reuses the existing generation-request format support, Ollama payload builder, JSON parsing and validation utilities, and conversation service.

### 8.1 Structured Response Flow

1. Enable JSON mode with `/json`.
2. Submit the next prompt as an ordinary message.
3. Send a request using the existing structured-generation workflow.
4. Forward JSON or JSON Schema formatting through the Ollama payload.
5. Parse the generated response.
6. Validate the result against the requested schema or response model.
7. Return validated data and display it as formatted JSON.
8. Store the assistant response only after successful validation.

The current `ConversationService.send_structured_message()` returns a Python dictionary in the documented implementation path. The notebook was updated to serialize the returned dictionary with `json.dumps()` rather than treating it as a Pydantic model.

### 8.2 Pydantic Response Contract

The CLI experiment used the `AssistantStructuredResponse` model, with the following constraints:

| Field | Type | Constraint |
|---|---|---|
| `answer` | String | Must not be empty |
| `confidence` | Float | Must be between `0` and `1`, inclusive |

The model is configured for strict validation and rejects unexpected fields. The validation experiments rejected:

- a response missing `confidence`;
- a confidence value above `1`;
- an unexpected additional field.

These are application-level contract checks. A structurally valid response can still contain incorrect or incomplete information.

### 8.3 Real Structured-Response Experiment

The CLI and notebook produced a validated answer distinguishing supervised learning from unsupervised learning. The recorded output contained:

- a non-empty `answer` string;
- a floating-point `confidence` value of `0.98`;
- successful schema/field checks;
- one user message and one assistant message stored in the conversation.

An earlier real-model integration test in the existing test suite also exercises JSON Schema generation. That test’s schema is separate from the CLI’s `AssistantStructuredResponse` contract; the fields in those two schemas should not be conflated.

## 9. Runtime Status and Performance Metrics

The runtime status and performance sections reuse the existing `OllamaRuntime` health, model-inventory, generation-result, and streaming metadata behavior.

### 9.1 Recorded Non-Streaming Metrics

One normal generation run recorded the following values:

| Metric | Observed value |
|---|---:|
| Model | `qwen3.8:27b` |
| Prompt tokens | 20 |
| Generated tokens | 57 |
| Client-side latency | 21,952.10 ms |
| Model load duration | 6.27 ms |
| Prompt evaluation duration | 1.67 s |
| Generation evaluation duration | 11.26 s |
| Output generation rate | 5.06 tokens/s |
| Finish reason | `stop` |

These fields come from the recorded request and runtime metadata. The client-side wall-clock latency and the runtime’s internal duration fields have different measurement boundaries and should not be treated as interchangeable.

### 9.2 Metrics Interpretation

- **Client-side latency** records the elapsed time measured by the application around a request.
- **Prompt evaluation duration** and **generation evaluation duration** are runtime-provided measurements.
- **Time to first text** is measured from the streaming experiment’s start boundary to the first non-empty text fragment.
- **Stream event count** is the number of reported events, not an exact generated-token count.
- **Output tokens per second** depends on the documented numerator and timing denominator. The recorded 5.06 tokens/s value is a single-run observation, not a general performance guarantee.

The experiments were not repeated under controlled hardware and workload conditions. They provide a baseline for future investigation, not a final benchmark.

## 10. Prompt-Formatting Audit and Decision

The prompt-format investigation aimed to verify how the application builds and sends the prompt before making any format change.

### 10.1 Observed Prompt Format

The current `PromptBuilder` emits text using explicit role labels:

```text
System:
You are a helpful AI assistant.

User:
Explain tokenization in one sentence.

Assistant:
```

The builder preserves message order, separates messages with blank lines, validates the placement and count of system messages, requires at least one non-system message, and requires the latest conversation message to be from the user. The generation request carries the built prompt as a string.

The runtime adapter builds the Ollama generation payload and passes `request.prompt` to the Ollama prompt field. The investigated request path did not itself prove whether Ollama or the model applies any internal processing beyond what the application exposes.

### 10.2 Prompt-Formatting Test Cases

Six focused prompt-construction cases passed:

1. System message followed by a user message — accepted.
2. User-only conversation — accepted.
3. Empty conversation — rejected.
4. Multiple system messages — rejected.
5. System message outside the first position — rejected.
6. Conversation ending with an assistant message — rejected.

### 10.3 Instruction-Following Experiment

A real-model request asked for exactly three colors and a numbered list with three items. The observed response was:

```text
1. Red
2. Blue
3. Green
```

The response followed the requested count and format for that prompt.

### 10.4 Context-Retention Experiment

The model was given a fictional mapping between project code `ORION-731` and configuration label `BLUE-LANTERN`. A later turn asked for the label associated with that code. The response was `BLUE-LANTERN`, and the expected value was confirmed in the recorded experiment.

### 10.5 Formatting Edge Cases

The tests confirmed that the prompt builder preserved:

- multiline message content;
- special characters;
- content containing strings that resemble role labels, such as `Assistant:` and `System:`.

A whitespace-only user message was rejected. Preserving role-like text as message content does not by itself establish protection against all forms of prompt injection; security behavior requires separate evaluation.

### 10.6 Decision

The current role-labelled plain-text format was retained. The inspected implementation and recorded experiments did not demonstrate a concrete formatting defect that justified changing the production prompt representation.

This decision is deliberately limited: the tests do not prove that the format is optimal for every model, nor do they establish universal instruction-following or prompt-injection resistance.

## 11. End-to-End Validation

The notebook validated the assistant across normal conversation, streaming, structured response generation, runtime failure handling, and the full automated test suite.

### 11.1 Normal Conversation

A two-turn interaction verified that the model could use information from the previous turn. Both requests completed, and the conversation statistics showed two user messages, two assistant messages, and four stored conversation messages.

### 11.2 Streaming Conversation

The streaming scenario requested a five-step explanation of how a neural network learns. With an output budget of `256`, the answer was cut off in the fifth step. Repeating the scenario with an output budget of `512` yielded all five numbered steps and a `stop` finish reason.

### 11.3 Structured Response

The structured-output scenario requested a concise explanation of supervised and unsupervised learning with a confidence score. The validated output had a non-empty answer, a confidence value of `0.98`, and correct Python field types. The conversation stored one user message and one assistant message after validation.

### 11.4 Runtime Failure Scenarios

The recorded targeted test run covered mocked Ollama runtime failures, conversation-service failure paths, and CLI error handling. Test coverage includes connection errors, timeouts, HTTP errors, invalid JSON responses, missing models, malformed/incomplete streams, invalid structured responses, and the requirement not to store failed assistant responses as successful completed turns.

## 12. Automated Testing

### 12.1 Full Test Suite

The recorded final full test-suite run was executed with:

```powershell
python -m pytest -v
```

Observed result:

```text
220 passed in 52.32s
```

The run included real-Ollama smoke tests for non-streaming generation, JSON Schema generation, and streaming generation. The result is the outcome of that recorded run; the suite was not rerun as part of writing this README.

### 12.2 Targeted Integration and Failure Tests

The following targeted command was also run:

```powershell
python -m pytest .\tests\test_ollama_runtime_mocked.py .\tests\test_conversation_service.py .\tests\test_cli.py -v
```

Observed result:

```text
58 passed in 1.43s
```

### 12.3 Test Modules and Coverage Areas

| Test module | Main coverage |
|---|---|
| `tests/test_cli.py` | CLI commands, mode toggles, normal/stream/structured routing, runtime-error recovery, clean exit |
| `tests/test_config.py` | Defaults, environment overrides, validation, invalid configuration and model identifiers |
| `tests/test_context_window_manager.py` | System/latest-user preservation, turn truncation, token reservation, invalid histories, budget constraints |
| `tests/test_conversation.py` | Message order, immutability, validation, message counts, system prompt handling, clearing |
| `tests/test_conversation_service.py` | Normal, stream, and structured workflows; persistence policy; history; failure and cancellation behavior; context manager integration |
| `tests/test_generation_result.py` | Generation result fields and runtime metadata validation |
| `tests/test_ollama_runtime_mocked.py` | Health/model inventory, request payloads, response validation, normalized runtime failures, stream completion and metadata |
| `tests/test_ollama_runtime_smoke.py` | Real local Ollama non-streaming and JSON Schema smoke tests |
| `tests/test_ollama_streaming_smoke.py` | Real local Ollama streaming smoke test |
| `tests/test_prompt_builder.py` | Prompt formatting, message-order rules, request construction, generation options, invalid histories |
| `tests/test_prompt_regression_dataset.py` | Dataset structure, unique case IDs, expected categories, required fields, roles, automated-check definitions |
| `tests/test_runtime_interface.py` | Runtime interface methods and annotations |
| `tests/test_runtime_schemas.py` | Generation request validation, options, serialization, JSON and JSON Schema formats |
| `tests/test_structured_output.py` | JSON parsing, JSON Schema validation, Pydantic contract checks, invalid outputs |
| `tests/test_token_counter.py` | Local tokenizer loading, token counting, invalid inputs, missing tokenizer files |

The prompt-regression dataset tests validate the dataset’s structure and required checks. They do not, by themselves, establish model quality across all cases. The real-model prompt examples in this notebook cover selected scenarios only.

### 12.4 Earlier Baseline

The notebook recorded an earlier baseline of `212 passed in 78.89s` before completion of the work. The later recorded full-suite run reported `220 passed in 52.32s`. Run times can vary between executions; these two measurements are not a controlled performance comparison.

## 13. Notebook Structure and Integrity Review

The notebook was reviewed without re-executing all model calls.

### 13.1 Section Structure

The notebook contains the following main sections:

1. Project Overview
2. Environment and Configuration
3. End-to-End Application Design
4. Interactive CLI Foundation
5. Normal Chat Integration
6. Streaming Generation
7. Structured Output Integration
8. Runtime Status and Performance Metrics
9. Prompt Formatting Decision
10. End-to-End Validation
11. Final System Evaluation
12. Final Summary

Section 11 includes implemented features, test results, performance observations, known limitations, and potential improvements.

### 13.2 Recorded Integrity Checks

| Check | Observed result |
|---|---|
| Notebook JSON parsing | Valid |
| Total cells | 86 |
| Markdown cells | 64 |
| Code cells | 22 |
| Python syntax check using `ast.parse` | All non-empty code cells passed |
| Code cells with saved outputs | 22 of 22 |
| Recorded code-cell errors | None |
| Code cells with `execution_count` | 10 of 22 |

Twelve code cells had saved stream outputs but no `execution_count`. The reason for this metadata state was not established by the file inspection. The counts were not edited manually, because execution counts should reflect actual notebook execution history. The notebook was not fully re-executed solely to regenerate metadata, avoiding repeat long-running local-model requests.

These checks confirm file integrity, Python syntax, and the state of the saved outputs; they do not prove that every saved cell was executed in the current kernel session or that all outputs can be reproduced identically today.

## 14. Known Limitations

### 14.1 Local Runtime and Model Dependency

Generation depends on the local Ollama service, the configured model, and available system resources. A missing model, unavailable service, timeout, or runtime error can prevent a request from completing.

### 14.2 Output-Length Constraints

The default generation budget can be too small for normal explanatory responses. The experiments showed that a request may reach `done_reason: length` while the answer is semantically incomplete.

### 14.3 Model Quality and Prompt Behavior

A passing prompt-format test or one successful real-model example does not prove consistent instruction following for all prompts. The format decision is based on the observed test scope, not a claim of universal optimality or prompt-injection resistance.

### 14.4 Structured Validity Versus Correctness

JSON parsing and schema validation verify structure and constraints. They do not prove that the content is factual, relevant, or complete.

### 14.5 Token Counting

The local tokenizer supports context budgeting, but equivalence with Ollama’s internal tokenization was not independently established in this work. Locally counted token values should therefore be interpreted according to their source and purpose rather than assumed to match all runtime counters exactly.

### 14.6 Performance Evidence

Measurements came from individual runs with varying prompts and output budgets. The experiments did not use repeated controlled trials, hardware-load control, percentile latency analysis, or statistical comparisons. They are useful observations, not general throughput or latency guarantees.

### 14.7 Notebook Execution Metadata

Twelve code cells had saved outputs but missing execution counts. This was documented rather than silently repaired. A future clean-kernel validation can address this state if reproducibility requirements justify re-running all cells.

## 15. Potential Improvements

The following are recommendations for future work; they are not claims that each item is necessarily absent from the current code.

### High Priority

- Improve user-facing handling of incomplete generation and make truncation reasons clear.
- Expand tests for long histories, context-budget edge cases, and structured-response truncation.
- Preserve clear distinctions between runtime completion, schema validity, and semantic completeness.
- Evaluate bounded retry or recovery strategies only for failure types where retrying is safe.

### Medium Priority

- Establish repeatable benchmarks with identical prompts and settings across multiple runs.
- Report median and percentile latency in addition to individual samples.
- Measure time to first text, total latency, and output throughput with clearly defined timing boundaries.
- Expand real-model prompt evaluation to cover correction handling, unknown information, and more varied context-retention examples.
- Improve structured logging and diagnostics for request completion, output limits, and normalized error categories.

### Lower Priority / Based on Need

- Consider history summarization if turn removal proves insufficient for longer conversations.
- Refine CLI feedback for long-running requests and expose relevant generation settings where helpful.
- Continue documenting configuration requirements and failure modes as additional runtimes or workflows are introduced.

## 16. Reproducibility Commands

Run commands from the project root:

### Full test suite

```powershell
python -m pytest -v
```

### Targeted runtime, conversation-service, and CLI tests

```powershell
python -m pytest .\tests\test_ollama_runtime_mocked.py .\tests\test_conversation_service.py .\tests\test_cli.py -v
```

### Inspect notebook JSON and cell counts

```powershell
@'
import json
from pathlib import Path

path = Path("notebooks/05_end_to_end_local_ai_assistant.ipynb")
notebook = json.loads(path.read_text(encoding="utf-8"))
print(f"Cells: {len(notebook['cells'])}")
print(f"Markdown: {sum(c.get('cell_type') == 'markdown' for c in notebook['cells'])}")
print(f"Code: {sum(c.get('cell_type') == 'code' for c in notebook['cells'])}")
'@ | python -
```

The recorded results in this README reflect past runs. Re-running the commands may produce different durations, and real-Ollama smoke tests require a reachable local runtime with the expected model available.

## 17. Final Assessment

We delivered an integrated local assistant workflow combining interactive CLI behavior, normal chat, streaming, structured responses, context management, runtime status, request metrics, and application-level validation.

The implementation reuses the earlier runtime and conversation components. The prompt-format investigation did not find a concrete defect requiring a format change, and selected real-model experiments demonstrated instruction following, context retention, streaming output, and schema-validated structured responses.

The recorded final automated test run reported **220 passed in 52.32 seconds**, while the focused runtime/conversation/CLI run reported **58 passed in 1.43 seconds**. The notebook structure and Python syntax were also inspected, with the saved execution-count inconsistency disclosed rather than altered artificially.

The main engineering lessons are:

- integration should preserve the boundaries between CLI, conversation service, prompt builder, context manager, and runtime adapter;
- successful runtime completion is not always equivalent to a complete answer;
- structured output requires application-level validation;
- streaming events are not guaranteed to correspond one-to-one with model tokens;
- performance numbers require controlled repeated measurements before making broad claims;
- testing, saved experiment evidence, and limitations should be documented separately and precisely.

**Outcome:** a working end-to-end local AI assistant with recorded integration evidence, automated test coverage, measured generation behavior, and an explicit list of limitations and next-step improvements.
