"""Privacy-minimizing demo analytics for accepted medicine checks (D-012).

Emits at most one compact JSON log record per accepted check, through the
existing Yandex Cloud Function log stream. Never stores raw Telegram
identifiers, message text, medicine names, or health details. Analytics
must never change, delay, or suppress the Telegram reply: every failure
mode here is caught and swallowed.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from datetime import UTC, datetime
from enum import StrEnum

from pharmacheck.contracts import AnalysisResult, AnalysisStatus, InteractionLevel

ANALYTICS_SCHEMA_VERSION = 1
ANALYTICS_EVENT_NAME = "demo_check"
ANONYMOUS_USER_PREFIX = "pharmacheck-demo"
ANONYMOUS_USER_HEX_LENGTH = 16
MIN_ANALYTICS_KEY_BYTES = 32


class AnalyticsOutcome(StrEnum):
    ANSWER = "answer"
    INSUFFICIENT_DATA = "insufficient_data"
    CLARIFICATION_NEEDED = "clarification_needed"
    URGENT = "urgent"
    TECHNICAL_ERROR = "technical_error"


class AnalyticsResolution(StrEnum):
    RESOLVED = "resolved"
    NEEDS_CLARIFICATION = "needs_clarification"
    UNKNOWN_ON_ERROR = "unknown_on_error"


_OUTCOME_BY_STATUS: dict[AnalysisStatus, AnalyticsOutcome] = {
    AnalysisStatus.ANSWER: AnalyticsOutcome.ANSWER,
    AnalysisStatus.INSUFFICIENT_DATA: AnalyticsOutcome.INSUFFICIENT_DATA,
    AnalysisStatus.CLARIFICATION_NEEDED: AnalyticsOutcome.CLARIFICATION_NEEDED,
    AnalysisStatus.URGENT: AnalyticsOutcome.URGENT,
}

# `resolution` describes drug-identification outcome; `outcome` describes
# the medical-analysis result. They are independent axes: `resolution` is
# "resolved" whenever the two drugs were successfully identified, even if
# the medical outcome itself is inconclusive. `insufficient_data` means
# identification succeeded but evidence did not, so it is `resolved`; only
# `clarification_needed` means identification itself needs the user to act,
# so it alone maps to `needs_clarification`. Confirmed 2026-08-12 (D-013).
_RESOLUTION_BY_OUTCOME: dict[AnalyticsOutcome, AnalyticsResolution] = {
    AnalyticsOutcome.ANSWER: AnalyticsResolution.RESOLVED,
    AnalyticsOutcome.URGENT: AnalyticsResolution.RESOLVED,
    AnalyticsOutcome.INSUFFICIENT_DATA: AnalyticsResolution.RESOLVED,
    AnalyticsOutcome.CLARIFICATION_NEEDED: AnalyticsResolution.NEEDS_CLARIFICATION,
    AnalyticsOutcome.TECHNICAL_ERROR: AnalyticsResolution.UNKNOWN_ON_ERROR,
}


def validate_analytics_key(
    value: str, *, other_secrets: tuple[str, ...] = ()
) -> bytes:
    """Validate a candidate `PHARMACHECK_ANALYTICS_HMAC_KEY` value.

    This only checks length and that the key differs from the other
    deployment secrets it must stay independent from (bot token, webhook
    secret) — code cannot verify actual randomness. The value stored in
    Lockbox at deployment time must be generated with a cryptographically
    secure random generator (e.g. `secrets.token_urlsafe(32)` or
    `openssl rand -base64 32`), not typed or derived by hand; a
    low-entropy 32+ byte string (e.g. a repeated character or a
    dictionary phrase) passes this check but provides no real HMAC
    security.

    Raises `ValueError` if the key is shorter than
    `MIN_ANALYTICS_KEY_BYTES` or equal to another secret.
    """
    encoded = value.encode("utf-8")
    if len(encoded) < MIN_ANALYTICS_KEY_BYTES:
        raise ValueError(
            f"analytics key must be at least {MIN_ANALYTICS_KEY_BYTES} bytes long "
            "(length is checked; the deployed value must still be generated with a "
            "cryptographically secure random generator, not just any long string)"
        )
    if any(value == other for other in other_secrets):
        raise ValueError(
            "analytics key must be separate from the Telegram bot token and webhook secret"
        )
    return encoded


def derive_anonymous_user(
    hmac_key: bytes,
    sender_id: int,
    *,
    now: datetime | None = None,
) -> tuple[str, str]:
    """Return `(period, anonymous_user)` for one monthly-rotating pseudonym."""
    moment = now or datetime.now(UTC)
    period = moment.strftime("%Y-%m")
    message = f"{ANONYMOUS_USER_PREFIX}:{period}:{sender_id}".encode()
    digest = hmac.new(hmac_key, message, hashlib.sha256).hexdigest()
    return period, digest[:ANONYMOUS_USER_HEX_LENGTH]


def outcome_for_result(result: AnalysisResult) -> tuple[AnalyticsOutcome, InteractionLevel]:
    """Map a validated `AnalysisResult` to its analytics outcome and level."""
    return _OUTCOME_BY_STATUS[result.status], result.interaction_level


def resolution_for_outcome(outcome: AnalyticsOutcome) -> AnalyticsResolution:
    return _RESOLUTION_BY_OUTCOME[outcome]


class DemoAnalyticsRecorder:
    """Best-effort, privacy-minimizing analytics; never affects the reply.

    Disabled (a silent no-op) whenever `hmac_key` is `None`, which is always
    the case outside `official_labels_demo` (D-012).
    """

    def __init__(
        self,
        *,
        hmac_key: bytes | None,
        usage_mode: str,
        logger: logging.Logger | None = None,
    ) -> None:
        self._hmac_key = hmac_key
        self._usage_mode = usage_mode
        if logger is None:
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
        self._logger = logger

    @property
    def enabled(self) -> bool:
        return self._hmac_key is not None

    def record(
        self,
        *,
        sender_id: int | None,
        outcome: AnalyticsOutcome,
        interaction_level: InteractionLevel | None,
        resolution: AnalyticsResolution,
        latency_ms: float,
    ) -> None:
        if self._hmac_key is None:
            return
        try:
            now = datetime.now(UTC)
            event: dict[str, object] = {
                "schema_version": ANALYTICS_SCHEMA_VERSION,
                "event": ANALYTICS_EVENT_NAME,
                "period": now.strftime("%Y-%m"),
            }
            if sender_id is not None:
                period, anonymous_user = derive_anonymous_user(
                    self._hmac_key, sender_id, now=now
                )
                event["period"] = period
                event["anonymous_user"] = anonymous_user
            event["outcome"] = outcome.value
            if interaction_level is not None:
                event["interaction_level"] = interaction_level.value
            event["resolution"] = resolution.value
            event["latency_ms"] = max(0, round(latency_ms))
            event["usage_mode"] = self._usage_mode
            self._logger.info(json.dumps(event, ensure_ascii=False, sort_keys=True))
        except Exception:
            try:
                self._logger.warning("demo_analytics_event_failed")
            except Exception:
                pass
