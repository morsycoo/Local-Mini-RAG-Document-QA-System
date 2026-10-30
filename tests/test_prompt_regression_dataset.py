"""Tests for the prompt regression dataset."""

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "prompt_regression_cases.json"

EXPECTED_CATEGORIES = {
    "role_priority",
    "format_constraint",
    "conciseness",
    "language_constraint",
    "conversation_context",
    "uncertainty_handling",
    "technical_explanation",
    "clarification_request",
    "structured_output",
    "context_order",
}


@pytest.fixture(scope="module")
def regression_cases() -> list[dict]:
    """Load the prompt regression dataset."""
    with DATASET_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def test_regression_dataset_contains_ten_cases(
    regression_cases: list[dict],
) -> None:
    assert len(regression_cases) == 10


def test_regression_case_ids_are_unique(
    regression_cases: list[dict],
) -> None:
    case_ids = [case["id"] for case in regression_cases]

    assert len(case_ids) == len(set(case_ids))


def test_regression_dataset_covers_expected_categories(
    regression_cases: list[dict],
) -> None:
    categories = {case["category"] for case in regression_cases}

    assert categories == EXPECTED_CATEGORIES


@pytest.mark.parametrize(
    "required_key",
    ["id", "category", "messages", "checks"],
)
def test_each_regression_case_has_required_fields(
    regression_cases: list[dict],
    required_key: str,
) -> None:
    for case in regression_cases:
        assert required_key in case, (
            f"{case.get('id', '<unknown>')} is missing {required_key!r}"
        )


def test_regression_cases_have_valid_message_roles(
    regression_cases: list[dict],
) -> None:
    allowed_roles = {"system", "user", "assistant"}

    for case in regression_cases:
        assert case["messages"], f"{case['id']} has no messages"

        for message in case["messages"]:
            assert message["role"] in allowed_roles
            assert isinstance(message["content"], str)
            assert message["content"].strip()


def test_regression_cases_define_automated_checks(
    regression_cases: list[dict],
) -> None:
    for case in regression_cases:
        checks = case["checks"]

        assert isinstance(checks.get("automated"), list)
        assert checks["automated"], f"{case['id']} has no automated checks"
        assert isinstance(checks.get("human_review"), list)
