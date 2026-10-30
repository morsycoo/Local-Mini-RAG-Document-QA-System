# Prompt Construction and Structured Output

## Project

**Project:** Local AI Assistant\
**Notebook:**
`notebooks/04_prompt_construction_and_structured_output.ipynb`\
**Local model used for integration testing:** `qwen3.8:27b`

## 1. Overview

We extended the Local AI Assistant with structured generation using
JSON mode and JSON Schema. The implementation connects structured
generation to the existing Ollama runtime and conversation service while
preserving the established runtime result contract.

The work includes request-schema support, runtime payload construction,
reusable JSON parsing and schema validation, conversation-service
integration, automated tests, and a real-model integration test.

## 2. Objectives

-   Support JSON mode and JSON Schema in generation requests.
-   Forward the requested output format to Ollama.
-   Parse generated JSON and validate it against a supplied JSON Schema.
-   Handle parsing and validation failures explicitly.
-   Integrate structured generation into the conversation service.
-   Avoid storing an invalid assistant response as a successful
    response.
-   Verify the changes with automated tests and a real local model.

## 3. Implementation

### 3.1 Generation Request Schema

**File:** `app/schemas.py`

`GenerationRequest` supports the following output-format options:

-   `"json"` for JSON mode.
-   A JSON Schema dictionary for schema-constrained generation.
-   `None` when structured output is not requested.

A runtime-schema test verifies that a JSON Schema dictionary is accepted
by the request model.

### 3.2 Ollama Runtime Integration

**File:** `app/runtime/ollama_runtime.py`

The runtime payload builder conditionally includes the requested
`format` value:

-   JSON mode forwards `"json"`.
-   Schema-constrained generation forwards the supplied JSON Schema
    object.
-   Requests without structured output omit the `format` field.

The runtime continues to use the existing generation-result contract.

### 3.3 Reusable JSON Parsing and Validation

**File:** `app/structured_output.py`

The module provides reusable parsing and validation functions for
structured LLM responses.

The implementation supports two validation paths:

- `parse_and_validate_json`: Parses generated JSON and validates it
  against a supplied JSON Schema.
- `parse_and_validate_model`: Parses generated JSON and validates it
  against a Pydantic model.

The application defines `AssistantStructuredResponse` as a structured
response contract with the following constraints:

- `answer`: A non-empty string.
- `confidence`: A numeric value between 0 and 1.
- Additional fields are forbidden.
- Strict validation is enabled.
- Non-finite numeric values are forbidden.

The module distinguishes JSON parsing errors, schema errors, and
structured-output validation errors through explicit exception types.

The validation process is performed at the application level even
when the generation request supplies a JSON Schema to Ollama.

### 3.4 Conversation Service Integration

**File:** `app/services/conversation_service.py`

The `send_structured_message` workflow integrates structured generation
with the runtime and validates the generated response before storing the
assistant's message.

The service handles invalid structured output as a failure rather than
treating it as a successful assistant response. The conversation-service
tests include coverage for an invalid response not being stored as a
successful assistant message.

## 4. Structured Output Contract Used in Integration Testing

The real-model integration test used a JSON Schema with three required
fields:

  Field          Type               Constraint
  -------------- ------------------ ---------------------
  `answer`       String             Non-empty text
  `confidence`   Number             Between 0 and 1
  `topics`       Array of strings   Exactly three items

The schema also prohibited additional properties.

The model returned a JSON object containing the required fields, and the
integration test reported that JSON Schema validation passed.

## 5. Files and Test Coverage

### Application Files

| File | Responsibility |
|---|---|
| `app/schemas.py` | Defines generation request formats, including JSON mode and JSON Schema dictionaries. |
| `app/runtime/ollama_runtime.py` | Builds and sends Ollama generation payloads with the requested output format. |
| `app/structured_output.py` | Parses generated JSON and validates it using JSON Schema or Pydantic models, with explicit structured-output error handling. |
| `app/services/conversation_service.py` | Integrates structured generation and validation into conversation handling, ensuring that an invalid assistant response is not stored as a successful response. |

### Test Files

| File | Coverage |
|---|---|
| `tests/test_structured_output.py` | Tests JSON parsing, JSON Schema validation, Pydantic model validation, and structured-output failure cases. |
| `tests/test_conversation_service.py` | Tests conversation behavior, structured-message handling, and invalid-response handling. |
| `tests/test_ollama_runtime_mocked.py` | Tests runtime payload construction, JSON mode, and JSON Schema forwarding to Ollama. |
| `tests/test_runtime_schemas.py` | Tests generation request validation, including JSON Schema input and invalid request values. |
| `tests/test_ollama_runtime_smoke.py` | Runs real Ollama smoke tests, including JSON Schema generation and response validation using the local model. |
| `tests/test_prompt_regression_dataset.py` | Tests prompt regression dataset structure, required fields, unique identifiers, categories, and dataset consistency. |

### Verification Summary

The latest full project test suite completed successfully:

- **Total tests passed:** 212
- **Failed tests:** 0
- **Execution time:** 58.56 seconds

Command:

```bash
python -m pytest .\tests -q
```

## 6. Verification Results

### 6.1 Targeted Automated Tests

The following test groups were executed successfully.

| Test Group | Result |
|---|---:|
| Structured Output | 19 passed |
| Prompt Regression Dataset | 9 passed |
| Conversation Service | 28 passed |
| Ollama Runtime Mocked | 22 passed |
| Runtime Schemas and Ollama Smoke Tests | 42 passed |

The combined test groups completed without failures.

The results above describe the individual test-group executions.
The full project test suite is the authoritative verification result.

### 6.2 Real-Model Integration Measurements

  Metric                        Observed value
  --------------------------- ----------------
  Model                          `qwen3.8:27b`
  Prompt tokens                             43
  Generated tokens                         109
  Client latency                  24,585.76 ms
  Integration test duration      25.37 seconds
  JSON Schema validation                  PASS

These measurements describe one test execution. They are not a general
performance benchmark.

### 6.3 Full Test Suite

The complete project test suite was executed after the latest
implementation and test updates.

Command:

```bash
python -m pytest .\tests -q
```
## 7. Structured Generation Workflow

The resulting workflow is:

1.  Create a generation request with either JSON mode or a JSON Schema.
2.  Build the Ollama payload with the requested `format`.
3.  Send the request to the local Ollama runtime.
4.  Receive the generated response.
5.  Parse the response as JSON.
6.  Validate the parsed value against the supplied JSON Schema.
7.  Handle parsing and validation failures explicitly.
8.  Store the assistant response only after successful validation.

## 8. Key Engineering Takeaways

-   JSON mode requests JSON-formatted output; JSON Schema specifies the
    expected structure and constraints.
-   Model-side structured generation and application-level validation
    are complementary. The application still validates the returned
    response.
-   Real-model integration testing verifies the actual local Ollama
    path, while mocked tests verify request construction and controlled
    runtime behavior.
-   Explicit errors make malformed JSON and schema-invalid output
    distinguishable from successful responses.
-   Regression tests help ensure new structured-output capabilities do
    not break previously tested behavior.

## 9. Limitations and Future Work

Potential follow-up work includes:

1.  Expanding tests for nested schemas and additional JSON Schema
    constraints.
2.  Improving observability for Ollama HTTP errors, including useful
    diagnostic details.
3.  Testing malformed JSON and schema-invalid responses across more
    scenarios.
4.  Measuring latency across repeated runs under controlled conditions.
5.  Evaluating structured generation across different local models.

## 10. Final Summary

The Local AI Assistant now supports JSON mode and JSON Schema
forwarding through the Ollama runtime, reusable JSON parsing and
validation, and Pydantic-based structured response validation.

The conversation service integrates structured generation and
validates generated responses before storing the assistant response
as successful.

Verification includes automated tests for structured output,
conversation behavior, runtime payload construction, request schemas,
and prompt regression dataset consistency. Real-model integration
tests also verified JSON Schema generation through the local Ollama
runtime using `qwen3.8:27b`.

The latest full project test suite completed with:

**212 passed, 0 failed, in 58.56 seconds.**

**Status: Implementation and automated verification completed.**
