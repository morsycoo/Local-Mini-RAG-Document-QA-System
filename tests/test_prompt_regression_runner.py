"""Tests for prompt regression evaluation."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from live_regression_runner import _run_case, evaluate_response


@pytest.mark.parametrize(
    ("response", "checks", "expected_passed"),
    [
        (
            "Unit testing verifies individual units of software.",
            {
                "must_include": ["unit testing"],
                "must_not_include": [],
                "automated": ["non_empty_output", "contains_required_phrase"],
            },
            True,
        ),
        (
            "Automated testing improves reliability.\n"
            "Automated testing catches regressions.\n"
            "Automated testing saves time.",
            {
                "must_include": ["automated testing"],
                "must_not_include": [],
                "automated": [
                    "non_empty_output",
                    "contains_required_phrase",
                    "exactly_three_bullet_lines",
                ],
            },
            False,
        ),
        (
            "- Automated testing catches regressions.\n"
            "- Automated testing improves reliability.\n"
            "- Automated testing saves time.",
            {
                "must_include": ["automated testing"],
                "must_not_include": [],
                "automated": [
                    "non_empty_output",
                    "contains_required_phrase",
                    "exactly_three_bullet_lines",
                ],
            },
            True,
        ),
        (
            '{"category": "runtime", "summary": "Connection failed", '
            '"recommended_action": "Check the server"}',
            {
                "must_include": [
                    "category",
                    "summary",
                    "recommended_action",
                ],
                "must_not_include": ["```"],
                "automated": [
                    "non_empty_output",
                    "valid_json_object",
                    "required_keys_present",
                    "string_values",
                ],
            },
            True,
        ),
        (
            '{"category": "runtime", "summary": 123, '
            '"recommended_action": "Check the server"}',
            {
                "must_include": [
                    "category",
                    "summary",
                    "recommended_action",
                ],
                "must_not_include": ["```"],
                "automated": [
                    "non_empty_output",
                    "valid_json_object",
                    "required_keys_present",
                    "string_values",
                ],
            },
            False,
        ),
        (
            "",
            {
                "must_include": [],
                "must_not_include": [],
                "automated": ["non_empty_output"],
            },
            False,
        ),
        (
            "I need more information. What would you like me to improve?",
            {
                "must_include": [],
                "must_not_include": [],
                "automated": ["non_empty_output", "contains_question_mark"],
            },
            True,
        ),
    ],
)
def test_evaluate_response(
    response: str,
    checks: dict,
    expected_passed: bool,
) -> None:
    result = evaluate_response(response, checks)

    assert result["passed"] is expected_passed


def test_evaluator_reports_individual_check_results() -> None:
    checks = {
        "must_include": ["GraphQL"],
        "must_not_include": [],
        "automated": ["non_empty_output", "contains_required_phrase"],
    }

    result = evaluate_response(
        "You mentioned REST.",
        checks,
    )

    assert result["passed"] is False
    assert result["checks"]["non_empty_output"] is True
    assert result["checks"]["contains_required_phrase"] is False


def test_evaluator_rejects_invalid_json() -> None:
    checks = {
        "must_include": [],
        "must_not_include": [],
        "automated": [
            "valid_json_object",
            "required_keys_present",
            "string_values",
        ],
    }

    result = evaluate_response(
        '{"category": "runtime",}',
        checks,
    )

    assert result["passed"] is False
    assert result["checks"]["valid_json_object"] is False


def test_evaluator_requires_a_json_object() -> None:
    checks = {
        "must_include": [],
        "must_not_include": [],
        "automated": [
            "valid_json_object",
            "required_keys_present",
            "string_values",
        ],
    }

    result = evaluate_response(
        '["runtime", "connection failure"]',
        checks,
    )

    assert result["passed"] is False
    assert result["checks"]["valid_json_object"] is False


def test_evaluator_passes_valid_response() -> None:
    checks = {
        "must_include": [],
        "must_not_include": [],
        "automated": ["non_empty_output"],
    }

    result = evaluate_response("A valid response.", checks)

    assert result["passed"] is True


def test_evaluator_does_not_pass_empty_response() -> None:
    checks = {
        "must_include": [],
        "must_not_include": [],
        "automated": ["non_empty_output"],
    }

    result = evaluate_response("", checks)

    assert result["passed"] is False


def test_run_case_fails_when_generation_stops_at_length() -> None:
    runtime = Mock()
    runtime.settings = SimpleNamespace(
        model="test-model",
        host="http://127.0.0.1:11434",
        timeout_seconds=120.0,
        num_ctx=4096,
        max_new_tokens=16,
        temperature=0.0,
    )

    runtime.generate.return_value = SimpleNamespace(
        text="A valid response.",
        model="test-model",
        done=True,
        done_reason="length",
        client_latency_ms=10.0,
        prompt_eval_count=5,
        eval_count=10,
        total_duration_ns=100,
        load_duration_ns=10,
        prompt_eval_duration_ns=20,
        eval_duration_ns=70,
    )

    case = {
        "id": "test_incomplete",
        "category": "test",
        "messages": [
            {"role": "system", "content": "Answer briefly."},
            {"role": "user", "content": "Say hello."},
        ],
        "checks": {
            "must_include": [],
            "must_not_include": [],
            "automated": ["non_empty_output"],
            "human_review": [],
        },
    }

    result = _run_case(runtime, None, case)

    assert result["status"] == "completed"
    assert result["done_reason"] == "length"
    assert result["evaluation"]["passed"] is False
    assert result["evaluation"]["checks"]["generation_completed"] is False


def test_run_case_passes_when_generation_stops_normally() -> None:
    runtime = Mock()
    runtime.settings = SimpleNamespace(
        model="test-model",
        host="http://127.0.0.1:11434",
        timeout_seconds=120.0,
        num_ctx=4096,
        max_new_tokens=16,
        temperature=0.0,
    )

    runtime.generate.return_value = SimpleNamespace(
        text="Hello.",
        model="test-model",
        done=True,
        done_reason="stop",
        client_latency_ms=10.0,
        prompt_eval_count=5,
        eval_count=3,
        total_duration_ns=100,
        load_duration_ns=10,
        prompt_eval_duration_ns=20,
        eval_duration_ns=70,
    )

    case = {
        "id": "test_complete",
        "category": "test",
        "messages": [
            {"role": "system", "content": "Answer briefly."},
            {"role": "user", "content": "Say hello."},
        ],
        "checks": {
            "must_include": [],
            "must_not_include": [],
            "automated": ["non_empty_output"],
            "human_review": [],
        },
    }

    result = _run_case(runtime, None, case)

    assert result["status"] == "completed"
    assert result["done_reason"] == "stop"
    assert result["evaluation"]["passed"] is True
    assert result["evaluation"]["checks"]["generation_completed"] is True
