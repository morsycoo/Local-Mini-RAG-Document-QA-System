"""Run live prompt regression evaluations against the configured Ollama model."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.cli import create_conversation_service
from app.config import load_settings
from app.conversation import ConversationManager
from app.prompt_builder import PromptBuilder
from app.runtime.exceptions import RuntimeErrorBase
from app.runtime.ollama_runtime import OllamaRuntime
from app.services.conversation_service import ConversationService

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET = PROJECT_ROOT / "data" / "prompt_regression_cases.json"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"

DEFAULT_CASE_IDS = [
    "prompt_002",
    "prompt_003",
    "prompt_005",
    "prompt_006",
    "prompt_008",
    "prompt_009",
]


def evaluate_response(
    response: str,
    checks: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate generated text against automated regression checks."""

    import re

    normalized_response = response.casefold()
    results: dict[str, bool] = {}

    for phrase in checks.get("must_include", []):
        results[f"must_include:{phrase}"] = phrase.casefold() in normalized_response

    for phrase in checks.get("must_not_include", []):
        results[f"must_not_include:{phrase}"] = (
            phrase.casefold() not in normalized_response
        )

    for check_name in checks.get("automated", []):
        if check_name == "non_empty_output":
            results[check_name] = bool(response.strip())

        elif check_name == "contains_required_phrase":
            phrases = checks.get("must_include", [])
            results[check_name] = bool(phrases) and all(
                phrase.casefold() in normalized_response for phrase in phrases
            )

        elif check_name == "exactly_three_bullet_lines":
            results[check_name] = _count_bullet_lines(response) == 3

        elif check_name == "exactly_two_bullet_lines":
            results[check_name] = _count_bullet_lines(response) == 2

        elif check_name == "at_most_two_sentences_approx":
            results[check_name] = _count_sentences_approximately(response) <= 2

        elif check_name == "contains_question_mark":
            results[check_name] = "?" in response or "؟" in response

        elif check_name == "valid_json_object":
            try:
                parsed = json.loads(response)
                results[check_name] = isinstance(parsed, dict)
            except (json.JSONDecodeError, TypeError):
                results[check_name] = False

        elif check_name == "required_keys_present":
            try:
                parsed = json.loads(response)
            except (json.JSONDecodeError, TypeError):
                parsed = None

            results[check_name] = isinstance(parsed, dict) and all(
                key in parsed for key in checks.get("must_include", [])
            )

        elif check_name == "string_values":
            try:
                parsed = json.loads(response)
            except (json.JSONDecodeError, TypeError):
                parsed = None

            results[check_name] = isinstance(parsed, dict) and all(
                isinstance(value, str) for value in parsed.values()
            )

        else:
            raise ValueError(f"Unsupported automated check: {check_name}")

    return {
        "passed": bool(results) and all(results.values()),
        "checks": results,
    }


def _count_bullet_lines(response: str) -> int:
    """Count Markdown bullet lines."""

    import re

    return sum(1 for line in response.splitlines() if re.match(r"^\s*[-*+]\s+\S", line))


def _count_sentences_approximately(response: str) -> int:
    """Estimate sentence count using common terminal punctuation."""

    import re

    return len(
        [
            sentence
            for sentence in re.split(r"[.!?؟]+", response.strip())
            if sentence.strip()
        ]
    )


def _load_cases(dataset_path: Path) -> list[dict[str, Any]]:
    """Load and validate the regression dataset."""

    with dataset_path.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    if not isinstance(cases, list):
        raise ValueError("The regression dataset must contain a JSON array.")

    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each regression case must be a JSON object.")

        if not isinstance(case.get("id"), str):
            raise ValueError("Every regression case must have a string ID.")

        messages = case.get("messages")
        checks = case.get("checks")

        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{case['id']}: messages must be a non-empty list.")

        if not isinstance(checks, dict):
            raise ValueError(f"{case['id']}: checks must be a JSON object.")

    return cases


def _run_case(
    runtime: OllamaRuntime,
    service: ConversationService,
    case: dict[str, Any],
    *,
    max_new_tokens: int | None = None,
) -> dict[str, Any]:
    """Run one case through the real conversation service."""

    messages = case["messages"]
    system_messages = [
        message["content"] for message in messages if message.get("role") == "system"
    ]

    if len(system_messages) > 1:
        raise ValueError(f"{case['id']}: multiple system messages are not supported.")

    system_prompt = system_messages[0] if system_messages else None

    conversation = ConversationManager(system_prompt=system_prompt)
    prompt_builder = PromptBuilder()

    service = ConversationService(
        runtime=runtime,
        conversation=conversation,
        prompt_builder=prompt_builder,
        settings=runtime.settings,
    )

    conversation_messages = [
        message for message in messages if message.get("role") != "system"
    ]

    if not conversation_messages or conversation_messages[-1].get("role") != "user":
        raise ValueError(
            f"{case['id']}: the final conversation message must be from the user."
        )

    for message in conversation_messages[:-1]:
        role = message.get("role")
        content = message.get("content")

        if not isinstance(content, str):
            raise ValueError(f"{case['id']}: message content must be a string.")

        if role == "user":
            conversation.add_user_message(content)
        elif role == "assistant":
            conversation.add_assistant_message(content)
        else:
            raise ValueError(f"{case['id']}: unsupported role {role!r}.")

    final_user_message = conversation_messages[-1]["content"]
    generation_options = {}

    if max_new_tokens is not None:
        generation_options["max_new_tokens"] = max_new_tokens

    result = service.send_message(final_user_message, **generation_options)

    evaluation = evaluate_response(result.text, case["checks"])

    # A length-limited response is incomplete even if its text passes
    # the content checks.
    if result.done_reason == "length":
        evaluation["checks"]["generation_completed"] = False
        evaluation["passed"] = False
    else:
        evaluation["checks"]["generation_completed"] = bool(result.done)
        evaluation["passed"] = (
            evaluation["passed"] and evaluation["checks"]["generation_completed"]
        )

    return {
        "case_id": case["id"],
        "category": case.get("category"),
        "status": "completed",
        "model": result.model,
        "done": result.done,
        "done_reason": result.done_reason,
        "response": result.text,
        "evaluation": evaluation,
        "human_review": {
            "status": "pending",
            "criteria": case["checks"].get("human_review", []),
        },
        "metrics": {
            "client_latency_ms": result.client_latency_ms,
            "prompt_eval_count": result.prompt_eval_count,
            "eval_count": result.eval_count,
            "total_duration_ns": result.total_duration_ns,
            "load_duration_ns": result.load_duration_ns,
            "prompt_eval_duration_ns": result.prompt_eval_duration_ns,
            "eval_duration_ns": result.eval_duration_ns,
        },
    }


def run_regression(
    case_ids: list[str],
    dataset_path: Path = DEFAULT_DATASET,
    report_dir: Path = DEFAULT_REPORT_DIR,
    max_new_tokens: int | None = None,
) -> Path:
    """Run selected live cases and write a JSON report."""

    if max_new_tokens is not None and not 1 <= max_new_tokens <= 512:
        raise ValueError("max_new_tokens must be between 1 and 512.")

    cases = _load_cases(dataset_path)
    cases_by_id = {case["id"]: case for case in cases}

    missing_ids = [case_id for case_id in case_ids if case_id not in cases_by_id]
    if missing_ids:
        raise ValueError(f"Unknown regression case IDs: {missing_ids}")

    settings = load_settings()
    runtime, service = create_conversation_service(settings)

    results: list[dict[str, Any]] = []

    try:
        for case_id in case_ids:
            case = cases_by_id[case_id]
            print(f"Running {case_id} ({case.get('category', 'unknown')})...")

            try:
                result = _run_case(
                    runtime,
                    service,
                    case,
                    max_new_tokens=max_new_tokens,
                )
            except Exception as error:
                result = {
                    "case_id": case_id,
                    "category": case.get("category"),
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "human_review": {
                        "status": "not_run",
                        "criteria": case["checks"].get("human_review", []),
                    },
                }

            results.append(result)

            if result["status"] == "completed":
                print(
                    f"  Automated checks: "
                    f"{'PASS' if result['evaluation']['passed'] else 'FAIL'}"
                )
            else:
                print(f"  ERROR: {result['error_type']}: {result['error']}")

    finally:
        runtime.close()

    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report_path = report_dir / f"day6_prompt_regression_{timestamp}.json"

    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "model_configured": settings.model_name,
        "ollama_host": settings.ollama_host,
        "request_timeout_seconds": settings.request_timeout_seconds,
        "num_ctx": settings.num_ctx,
        "max_new_tokens": max_new_tokens or settings.max_new_tokens,
        "temperature": settings.temperature,
        "case_ids": case_ids,
        "summary": {
            "total": len(results),
            "completed": sum(result["status"] == "completed" for result in results),
            "errors": sum(result["status"] == "error" for result in results),
            "passed": sum(
                result.get("evaluation", {}).get("passed", False) for result in results
            ),
            "failed": sum(
                result["status"] == "completed" and not result["evaluation"]["passed"]
                for result in results
            ),
        },
        "results": results,
    }

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nReport saved to: {report_path}")
    print(f"Summary: {report['summary']}")

    return report_path


def main() -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser(
        description="Run live prompt regression tests against Ollama."
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="case_ids",
        help="Regression case ID. May be provided multiple times.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the regression dataset JSON file.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory in which to save the JSON report.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override the maximum number of generated tokens.",
    )

    args = parser.parse_args()
    case_ids = args.case_ids or DEFAULT_CASE_IDS

    try:
        run_regression(
            case_ids=case_ids or DEFAULT_CASE_IDS,
            dataset_path=args.dataset,
            report_dir=args.report_dir,
            max_new_tokens=args.max_new_tokens,
        )
    except (OSError, ValueError, RuntimeErrorBase) as error:
        print(f"Regression run failed: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
