# HADOTA Web Chat — Reliability and Hardening Report

**Report date:** 2026-10-23  
**Reviewed baseline:** `HADOTA_FULL_AUDIT.zip` and `HADOTA_Master_Execution_Plan.md`  
**Status:** A hardening patch has been prepared and tested in the audit environment. It has **not yet been applied to the Windows working copy**. Windows full-suite and local Ollama verification remain required.

## Summary

The existing HADOTA web app already provided a FastAPI UI/API, anonymous in-memory sessions, conversation CRUD, a local Ollama integration, and a non-streaming chat flow. The user manually verified the earlier navigation fix while a reply was generating. This patch preserves those behaviors and adds reliability and resource controls identified in the static audit.

## Changes in this patch

### Conversation and runtime reliability

- Added a re-entrant lock inside `ConversationManager` so snapshots and mutations of message history are synchronized when the web handler reads history while a worker thread generates a reply.
- Corrected cleanup lock ordering: session cleanup no longer holds the session lock while waiting to acquire a conversation lock, preventing a lock inversion with conversation deletion. Added a regression test for this contention case.
- Expiry discovered inside a request now defers waiting for active conversation locks to a cleanup thread, avoiding a long cleanup wait on the ASGI event loop; explicit shutdown and the periodic sweeper still wait for cleanup to finish. Conversation creation also rechecks session-registry ownership to avoid inserting into a session concurrently removed during close.
- Added a conditional `remove_last_message_if()` operation and rollback behavior for failed/incomplete `send_message`, `stream_message`, and structured-output turns. A normal retry no longer silently appends the same failed user message a second time.
- Added non-blocking per-conversation operations for API handlers. A turn already running in a conversation returns a clear `409` instead of waiting synchronously on a `threading.Lock` in the ASGI event loop.
- Added a non-blocking global generation semaphore with a one-generation default. A second generation receives a retryable `429` busy response rather than consuming unbounded model work.
- Shielded synchronous inference in a worker task and wait for the actual worker to exit before releasing the operation context if the request coroutine is cancelled. This does not claim to cancel the upstream Ollama generation; it prevents premature lock/slot release while the worker can still mutate history.
- Made deletion use non-blocking per-conversation acquisition and moved potentially waiting session deletion/close-all work off the event loop.
- Added an injectable per-conversation service factory for tests/integrations so independent conversations do not accidentally share one mocked service/history.

### Anonymous-session and request protections

- Added a bounded API request-body middleware (64 KiB by default, configurable) that checks both `Content-Length` and received body chunks before FastAPI parses the payload. A focused test verifies that a non-default configured byte cap is actually honored.
- Added per-session and process-wide sliding-window generation rate limits (defaults: 10/session/minute and 20/process/minute).
- Wired configurable limits into session creation: at most 25 active sessions, 5 conversations/session, 15-minute idle TTL and 4-hour absolute TTL by default.
- Added public-mode validation requiring HTTPS origins, secure cookies, one active generation, and conservative request/session/rate/lifetime bounds.
- Replaced detailed Pydantic request-validation responses with a stable generic message so invalid request input is not echoed to the client.
- Added generated `X-Request-ID` response headers plus `nosniff` and `no-referrer` headers on normal middleware responses.
- Added a best-effort `/api/session/close` signal with a 30-second grace period. A refresh/new request clears the close signal; if there is no further activity, the normal cleanup sweep removes the session. The explicit “End session” action still deletes immediately. Idle/absolute TTLs remain authoritative fallbacks.

### API and frontend behavior

- `/api/health` now checks web-process liveness only and does not call Ollama.
- Added `/api/ready` to check runtime connectivity and the configured model inventory without blocking the ASGI event loop.
- A continuation that finishes after navigation now reloads the active conversation history if its original DOM node has been replaced.
- Failed-send Retry reuses the existing user message element rather than rendering a second copy of that message.
- Updated visible privacy copy so it describes temporary chats and warns against sending secrets.
- Documented that browser chat currently uses the non-streaming path. Browser SSE streaming, its disconnect semantics and UI terminal events remain deferred rather than being represented as implemented.

### Project setup and automation

- Populated `requirements.txt` and `requirements-dev.txt`.
- Added `.env.example` and a non-empty `.gitignore`.
- Added a GitHub Actions workflow for Python checks, JavaScript syntax checking and deterministic tests. The workflow excludes the real-Ollama smoke modules because a hosted CI runner does not have the user's local Ollama process/model.
- Added a root README with setup, limits, privacy/session behavior, test commands, architecture and current limitations.

## Default resource policy

| Control | Default | Notes |
|---|---:|---|
| API body | 65,536 bytes | Checked before route payload parsing |
| Message | 4,000 characters | Existing API schema maximum |
| Active sessions | 25 | In-memory process capacity |
| Conversations/session | 5 | In-memory session capacity |
| Idle expiry | 900 seconds | Activity refreshes the idle timer |
| Absolute expiry | 14,400 seconds | Hard cap even with continued activity |
| Page-close grace | 30 seconds | Refresh/new activity cancels pending close |
| Per-session generation rate | 10/60 seconds | Sliding window |
| Process-wide generation rate | 20/60 seconds | Shared across all sessions |
| Active model generations | 1 | Busy response rather than queueing |

The defaults are conservative guardrails, not measured hardware capacity or a guarantee against deliberate abuse. Public mode additionally fails startup validation if limits are configured outside the allowed policy.

## Test evidence

After the hardening changes and four additional regression tests, the audit environment ran:

```text
355 passed, 5 skipped in 4.22s
```

Command used:

```text
PYTHONPATH=<temporary import shim> python -m pytest -q \
  --ignore=tests/test_ollama_runtime_smoke.py \
  --ignore=tests/test_ollama_streaming_smoke.py
```

**Test-environment limitations:** the isolated audit container does not have the `tokenizers` wheel or the user's Ollama process. A temporary import shim was used only to permit module import; the token-counter fixture tests skipped because `assets/tokenizers/qwen3.8-27b/tokenizer.json` was omitted from the audit archive due to the file collection size limit. The two real-Ollama smoke modules were excluded because `127.0.0.1:11434` is not running in the audit container. These are not claims that the live tests passed. Python compilation and `node --check frontend/app.js` passed.

The uploaded baseline archive separately recorded 337 passes. That historical result is not substituted for the current patch result. The Windows user environment must run the full test suite after applying the patch, where its actual tokenizer JSON and local Ollama are available.

## Verification still required on the Windows working copy

1. Apply the patch using the supplied PowerShell installer, which first creates timestamped backups.
2. Confirm the installer reports a clean syntax and full test run. If real Ollama tests fail, inspect whether the service/model is unavailable before attributing the error to application code.
3. Repeat the user's successful navigation test: generate in A, switch to B, return to A before completion, and check that the final text is in A.
4. Trigger a controlled generation failure, press Retry once, and confirm the backend history and visible UI contain one user turn, not duplicates.
5. If a response can be continued, start continuation, navigate away/back, then verify the completed answer is visible and its continuation button state is correct.
6. Run two sessions and verify conversation history is isolated. Test busy/rate-limit and 413 responses through the API.
7. Refresh the browser during a temporary session and confirm the grace period preserves the session; close the page without activity and verify expiry after the grace/sweep window.
8. Verify `/api/health` remains 200 when Ollama is offline and `/api/ready` becomes unavailable.
9. Do not enable Tailscale Funnel until the local public-mode settings, external-network tests and shutdown procedure pass.

## Not claimed complete

- No browser-streaming integration was added; the non-streaming path is the supported fallback.
- No real-browser Playwright/accessibility/mobile regression suite is present in this patch.
- No manual real-model smoke run was performed in the audit container.
- Tailscale Funnel/public sharing was not started and must remain off until its gates pass.
- No Git repository was created because the supplied working copy is not a Git repository.
