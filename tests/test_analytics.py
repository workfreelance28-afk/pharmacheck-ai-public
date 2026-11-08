from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import pytest

from pharmacheck.analytics import (
    ANALYTICS_SCHEMA_VERSION,
    AnalyticsOutcome,
    AnalyticsResolution,
    DemoAnalyticsRecorder,
    derive_anonymous_user,
    outcome_for_result,
    resolution_for_outcome,
    validate_analytics_key,
)
from pharmacheck.contracts import AnalysisResult, AnalysisStatus, InteractionLevel

KEY = b"x" * 32

ALLOWED_SCHEMA_FIELDS = {
    "schema_version",
    "event",
    "period",
    "anonymous_user",
    "outcome",
    "interaction_level",
    "resolution",
    "latency_ms",
    "usage_mode",
}

PROHIBITED_VALUES = [
    "12345",  # a raw Telegram sender/chat/update id used in the fixtures below
    "Секретный препарат",
    "Альфа",
    "внутренний секрет",
    "bot-token-value",
    "webhook-secret-value",
    "https://example.test/label",
]


def _capturing_logger():
    logger = logging.getLogger("test.analytics." + str(id(object())))
    logger.setLevel(logging.INFO)
    records: list[str] = []

    class Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    logger.addHandler(Handler())
    logger.propagate = False
    return logger, records


def _answer_result(*, level=InteractionLevel.GREEN, status=AnalysisStatus.ANSWER) -> AnalysisResult:
    return AnalysisResult.model_validate(
        {
            "status": status.value,
            "interaction_level": level.value,
            "headline": "Значимого известного взаимодействия не найдено",
            "identified_drugs": [
                {"trade_name": "Альфа", "active_ingredients": ["вещество альфа"]},
                {"trade_name": "Бета", "active_ingredients": ["вещество бета"]},
            ],
            "summary": "В проверенных источниках взаимодействие не найдено.",
            "mechanism_or_effect": None,
            "recommended_user_action": "Не изменяйте лечение без консультации врача.",
            "limitations": ["Индивидуальные факторы не оценивались."],
            "conflicts": [],
            "missing_information": [],
            "sources": [
                {
                    "source_id": "openfda.label.abc",
                    "title": "FDA label",
                    "url": "https://example.test/label",
                    "supports": "внутренний секрет из источника",
                }
            ],
        }
    )


def _white_result(status: AnalysisStatus) -> AnalysisResult:
    return AnalysisResult.model_validate(
        {
            "status": status.value,
            "interaction_level": "white",
            "headline": "Нужно уточнить препараты",
            "identified_drugs": [],
            "summary": "Медицинский вывод пока не сформирован.",
            "mechanism_or_effect": None,
            "recommended_user_action": "Уточните названия препаратов.",
            "limitations": [],
            "conflicts": [],
            "missing_information": ["Секретный препарат не распознан."],
            "sources": [],
        }
    )


def test_derive_anonymous_user_is_stable_within_month_and_changes_across_months():
    august = datetime(2026, 8, 15, tzinfo=UTC)
    september = datetime(2026, 9, 1, tzinfo=UTC)

    first = derive_anonymous_user(KEY, 12345, now=august)
    second = derive_anonymous_user(KEY, 12345, now=datetime(2026, 8, 1, tzinfo=UTC))
    third = derive_anonymous_user(KEY, 12345, now=september)

    assert first == second
    assert first[0] == "2026-08"
    assert third[0] == "2026-09"
    assert first[1] != third[1]
    assert len(first[1]) == 16
    assert all(character in "0123456789abcdef" for character in first[1])


def test_derive_anonymous_user_differs_by_sender():
    now = datetime(2026, 8, 15, tzinfo=UTC)
    first = derive_anonymous_user(KEY, 111, now=now)
    second = derive_anonymous_user(KEY, 222, now=now)
    assert first[1] != second[1]


def test_derive_anonymous_user_requires_the_correct_key():
    now = datetime(2026, 8, 15, tzinfo=UTC)
    first = derive_anonymous_user(KEY, 12345, now=now)
    second = derive_anonymous_user(b"y" * 32, 12345, now=now)
    assert first[1] != second[1]


def test_validate_analytics_key_rejects_short_key():
    with pytest.raises(ValueError):
        validate_analytics_key("short-key")


def test_validate_analytics_key_rejects_reused_secret():
    key = "x" * 32
    with pytest.raises(ValueError):
        validate_analytics_key(key, other_secrets=(key, "webhook-secret-value"))


def test_validate_analytics_key_accepts_a_distinct_long_key():
    encoded = validate_analytics_key(
        "y" * 32, other_secrets=("bot-token-value", "webhook-secret-value")
    )
    assert encoded == ("y" * 32).encode("utf-8")


@pytest.mark.parametrize(
    "status,expected_outcome",
    [
        (AnalysisStatus.ANSWER, AnalyticsOutcome.ANSWER),
        (AnalysisStatus.URGENT, AnalyticsOutcome.URGENT),
        (AnalysisStatus.INSUFFICIENT_DATA, AnalyticsOutcome.INSUFFICIENT_DATA),
        (AnalysisStatus.CLARIFICATION_NEEDED, AnalyticsOutcome.CLARIFICATION_NEEDED),
    ],
)
def test_outcome_for_result_maps_every_status(status, expected_outcome):
    if status in {
        AnalysisStatus.INSUFFICIENT_DATA,
        AnalysisStatus.CLARIFICATION_NEEDED,
        AnalysisStatus.URGENT,
    }:
        result = _white_result(status)
    else:
        result = _answer_result(status=status)

    outcome, level = outcome_for_result(result)

    assert outcome is expected_outcome
    assert level is result.interaction_level


def test_resolution_for_outcome_matches_the_confirmed_d013_mapping():
    """`resolution` tracks drug identification, `outcome` tracks the medical
    result — independent axes, confirmed 2026-08-12 (D-013). A pair that was
    identified but had insufficient evidence is `resolved`, not
    `needs_clarification`: only an unidentified/ambiguous pair needs the
    user to act.
    """
    assert resolution_for_outcome(AnalyticsOutcome.ANSWER) is AnalyticsResolution.RESOLVED
    assert resolution_for_outcome(AnalyticsOutcome.URGENT) is AnalyticsResolution.RESOLVED
    assert (
        resolution_for_outcome(AnalyticsOutcome.INSUFFICIENT_DATA)
        is AnalyticsResolution.RESOLVED
    ), "insufficient_data means identification succeeded; only evidence was lacking (D-013)"
    assert (
        resolution_for_outcome(AnalyticsOutcome.CLARIFICATION_NEEDED)
        is AnalyticsResolution.NEEDS_CLARIFICATION
    )
    assert (
        resolution_for_outcome(AnalyticsOutcome.TECHNICAL_ERROR)
        is AnalyticsResolution.UNKNOWN_ON_ERROR
    )


def test_recorder_emits_only_allowed_fields_and_no_prohibited_data():
    logger, records = _capturing_logger()
    recorder = DemoAnalyticsRecorder(
        hmac_key=KEY, usage_mode="official_labels_demo", logger=logger
    )

    recorder.record(
        sender_id=12345,
        outcome=AnalyticsOutcome.ANSWER,
        interaction_level=InteractionLevel.GREEN,
        resolution=AnalyticsResolution.RESOLVED,
        latency_ms=1234.5,
    )

    assert len(records) == 1
    event = json.loads(records[0])
    assert set(event) <= ALLOWED_SCHEMA_FIELDS
    assert event["schema_version"] == ANALYTICS_SCHEMA_VERSION
    assert event["event"] == "demo_check"
    assert event["outcome"] == "answer"
    assert event["interaction_level"] == "green"
    assert event["resolution"] == "resolved"
    assert event["usage_mode"] == "official_labels_demo"
    assert isinstance(event["latency_ms"], int)
    assert event["latency_ms"] >= 0
    assert len(event["anonymous_user"]) == 16
    for forbidden in PROHIBITED_VALUES:
        assert forbidden not in records[0]
    assert "update_id" not in event
    assert "12345" not in records[0]


def test_default_recorder_enables_info_events_for_cloud_runtime():
    logger = logging.getLogger("pharmacheck.analytics")
    previous_level = logger.level
    previous_propagate = logger.propagate
    records: list[str] = []

    class Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    handler = Handler()
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    logger.propagate = False
    try:
        recorder = DemoAnalyticsRecorder(
            hmac_key=KEY,
            usage_mode="official_labels_demo",
        )
        recorder.record(
            sender_id=12345,
            outcome=AnalyticsOutcome.ANSWER,
            interaction_level=InteractionLevel.RED,
            resolution=AnalyticsResolution.RESOLVED,
            latency_ms=1234.5,
        )
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate

    assert len(records) == 1
    event = json.loads(records[0])
    assert event["event"] == "demo_check"
    assert event["interaction_level"] == "red"
    assert "12345" not in records[0]


def test_recorder_omits_anonymous_user_without_a_sender_id():
    logger, records = _capturing_logger()
    recorder = DemoAnalyticsRecorder(
        hmac_key=KEY, usage_mode="official_labels_demo", logger=logger
    )

    recorder.record(
        sender_id=None,
        outcome=AnalyticsOutcome.TECHNICAL_ERROR,
        interaction_level=None,
        resolution=AnalyticsResolution.UNKNOWN_ON_ERROR,
        latency_ms=5.0,
    )

    event = json.loads(records[0])
    assert "anonymous_user" not in event
    assert "period" in event
    assert "interaction_level" not in event


def test_recorder_disabled_without_a_key_emits_nothing():
    logger, records = _capturing_logger()
    recorder = DemoAnalyticsRecorder(hmac_key=None, usage_mode="noncommercial_mvp", logger=logger)

    recorder.record(
        sender_id=12345,
        outcome=AnalyticsOutcome.ANSWER,
        interaction_level=InteractionLevel.GREEN,
        resolution=AnalyticsResolution.RESOLVED,
        latency_ms=1.0,
    )

    assert records == []
    assert recorder.enabled is False


def test_recorder_never_raises_when_logging_fails():
    class BrokenLogger:
        def info(self, message):
            raise RuntimeError("log sink unavailable")

        def warning(self, message):
            raise RuntimeError("log sink unavailable too")

    recorder = DemoAnalyticsRecorder(
        hmac_key=KEY, usage_mode="official_labels_demo", logger=BrokenLogger()
    )

    recorder.record(
        sender_id=12345,
        outcome=AnalyticsOutcome.ANSWER,
        interaction_level=InteractionLevel.GREEN,
        resolution=AnalyticsResolution.RESOLVED,
        latency_ms=1.0,
    )
