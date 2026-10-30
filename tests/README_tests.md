# HADOTA Test Suite Guide

This document explains the purpose of each Python file in `tests/` and how the files fit into HADOTA's verification workflow.

## Test Suite at a Glance

HADOTA uses several kinds of checks:

- **Unit tests** verify individual components and their expected behavior.
- **Service and API tests** verify how components work together, often without requiring a live model.
- **Mocked-runtime tests** check Ollama request construction and response handling without making real inference requests.
- **Real-runtime smoke tests** verify selected operations against a running local Ollama service and model.
- **Live regression evaluation** runs a selected dataset of prompts against the configured model and records results for review.

> **Important:** Tests that call the real Ollama runtime need Ollama to be running and the configured model to be available. They can be slower and their outputs or timings can vary. A passing automated assertion does not, by itself, prove that a model response is factually correct or semantically ideal.

## File-by-File Reference

| File | What it covers | Why it matters |
|---|---|---|
| `live_regression_runner.py` | Executes selected prompt-regression cases against the configured conversation service/runtime and gathers per-case results. | Provides a repeatable way to inspect real-model behavior. This is a runner utility, not simply a unit-test module. |
| `test_cli.py` | Tests the command-line interface and its user-facing command/interaction behavior. | Helps ensure CLI commands and conversation interactions behave as expected. |
| `test_config.py` | Tests configuration loading, defaults, validation, and handling of configuration inputs. | Prevents invalid or unexpected settings from silently entering runtime behavior. |
| `test_context_window_manager.py` | Tests how conversation history is selected or shortened to fit the configured prompt/context token budget. | Helps keep prompts within the allowed context while preserving the most relevant conversation turns. |
| `test_conversation.py` | Tests conversation-history operations, including message handling and conversation state behavior. | Protects the core data structure used to represent a conversation. |
| `test_conversation_service.py` | Tests the service that coordinates conversation history, prompt construction, runtime generation, and response handling. | Verifies the central application workflow, including successful turns and relevant failure/incomplete-generation behavior. |
| `test_generation_result.py` | Tests the generation-result data model and its validation/metadata contract. | Keeps model output and generation metadata in a predictable application-level format. |
| `test_ollama_runtime_mocked.py` | Tests the Ollama adapter using mocked HTTP/runtime responses rather than a live model. | Checks request payloads, response normalization, and error handling quickly and deterministically. |
| `test_ollama_runtime_smoke.py` | Runs a small set of real-runtime checks against local Ollama, including actual generation paths supported by the test module. | Confirms that the integration works with the locally installed runtime/model, beyond mocked tests. |
| `test_ollama_streaming_smoke.py` | Runs smoke checks for streaming generation against the real local Ollama runtime. | Exercises the streaming adapter path with real runtime events. |
| `test_prompt_builder.py` | Tests construction of prompts from ordered conversation messages and related request inputs. | Helps ensure the intended conversation context and roles are sent to the model. |
| `test_prompt_regression_dataset.py` | Tests the prompt-regression dataset's structure and consistency, such as required fields and case identifiers. | Catches malformed or inconsistent evaluation cases before running model evaluations. |
| `test_prompt_regression_runner.py` | Tests the live-regression runner's dataset selection, case execution, checks, and result/report behavior. | Keeps the evaluation tooling predictable and makes its reporting logic testable without relying entirely on live inference. |
| `test_runtime_interface.py` | Tests the runtime interface contract and expected runtime operations. | Helps keep application services decoupled from a particular inference backend. |
| `test_runtime_schemas.py` | Tests validation of generation request/result schemas and supported configuration values. | Rejects invalid runtime inputs and keeps component boundaries consistent. |
| `test_structured_output.py` | Tests JSON parsing and structured-response validation, including JSON Schema/Pydantic validation and failure cases. | Prevents invalid structured model output from being treated as a valid result. |
| `test_token_counter.py` | Tests token-counting behavior using the project's tokenizer assets. | Supports context-budget calculations. These tests may depend on the tokenizer package and local tokenizer asset being present. |
| `test_web_api.py` | Tests the web application's API behavior and request/response handling. | Verifies API-level behavior beyond isolated backend components. |
| `test_web_endpoints.py` | Tests specific web routes/endpoints and their expected status codes and responses. | Helps catch regressions in the HTTP interface exposed by HADOTA. |
| `test_web_schemas.py` | Tests validation of web request and response schemas. | Ensures invalid payloads are rejected and API data contracts remain consistent. |
| `test_web_sessions.py` | Tests anonymous session lifecycle and session/conversation isolation, including relevant expiry, limits, and cleanup behavior. | Protects separation between users' temporary sessions and helps verify resource cleanup and lifecycle behavior. |

## How to Run the Tests

Run the full test suite from the project root:

```powershell
python -m pytest .\tests -q
```

Run one test module when investigating a specific component:

```powershell
python -m pytest .\tests\test_conversation_service.py -q
```

Run the real Ollama smoke tests separately when Ollama is available and the configured model has been downloaded:

```powershell
python -m pytest .\tests\test_ollama_runtime_smoke.py .\tests\test_ollama_streaming_smoke.py -q
```

Run prompt-regression tooling tests:

```powershell
python -m pytest .\tests\test_prompt_regression_dataset.py .\tests\test_prompt_regression_runner.py -q
```

The `live_regression_runner.py` utility is used to execute the selected prompt cases against the configured runtime. Check its argument/help interface before running it, since the available options are defined by the current implementation.

## Recommended Verification Order

1. Run the deterministic test suite first.
2. Run real-Ollama smoke tests when the local runtime and model are available.
3. Run the live prompt-regression evaluation when you want to review actual model behavior across selected cases.
4. Review generated reports and logs instead of relying only on the final pass count.

## Interpreting Results

- **Unit/mock tests** are generally faster and more repeatable than live inference tests.
- **Smoke tests** verify that the configured local runtime can complete selected real requests; they are not a comprehensive performance benchmark.
- **Prompt-regression checks** only establish that the configured assertions passed. Human review is still needed for factual accuracy, usefulness, tone, and semantic quality.
- **Coverage** measures which code paths were exercised; it does not guarantee correctness.

For the current recorded verification totals and closeout status, consult the latest `reports/HADOTA_Closeout_Report.md` rather than copying historical counts into this file.
