"""Telegram webhook routing for a Yandex API Gateway proxy event."""

from __future__ import annotations

import base64
import binascii
import hmac
import ipaddress
import json
import logging
import re
import threading
import time
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pharmacheck.analytics import (
    AnalyticsOutcome,
    DemoAnalyticsRecorder,
    outcome_for_result,
    resolution_for_outcome,
)
from pharmacheck.contracts import AnalysisResult, InteractionLevel
from pharmacheck.telegram import TelegramAPIError, TelegramBotClient, format_analysis_result
from pharmacheck.yandexgpt import YandexGPTError

logger = logging.getLogger(__name__)

MAX_WEBHOOK_BODY_BYTES = 256_000
MAX_USER_TEXT_LENGTH = 4096
SECRET_HEADER = "x-telegram-bot-api-secret-token"

WELCOME_TEXT = (
    "PharmaCheck AI — ограниченная демо-версия. Она проверяет два препарата по "
    "найденным официальным источникам (инструкциям) и показывает потенциальное "
    "взаимодействие.\n\n"
    "Можно указать торговые названия, действующие вещества или одно и другое. "
    "Предпочтительный формат:\nПрепарат А + Препарат Б\n\n"
    "Примеры:\nНурофен + Варфарин\nамиодарон + вориконазол\n\n"
    "Отсутствие найденной записи не доказывает безопасность комбинации.\n\n"
    "Сервис не ставит диагнозы, не назначает, не отменяет, не заменяет и не "
    "изменяет лечение.\n\n"
    "Справка: /help. Конфиденциальность: /privacy."
)
# The private noncommercial_mvp bot is not a demo and also uses DDInter; the
# rest of the text, including every safety statement, matches WELCOME_TEXT.
NONCOMMERCIAL_WELCOME_TEXT = WELCOME_TEXT.replace(
    "PharmaCheck AI — ограниченная демо-версия. Она проверяет два препарата по "
    "найденным официальным источникам (инструкциям) и показывает потенциальное "
    "взаимодействие.",
    "PharmaCheck AI — частный некоммерческий прототип. Он проверяет два препарата "
    "по найденным официальным инструкциям и некоммерческой базе взаимодействий "
    "DDInter и показывает потенциальное взаимодействие.",
)
HELP_TEXT = (
    "В одном сообщении укажите ровно два препарата. Поддерживаемые форматы:\n"
    "• Препарат А + Препарат Б\n"
    "• /check Препарат А + Препарат Б\n"
    "• обычная фраза, например: «Я принимаю Препарат А. Мне назначили "
    "Препарат Б.»\n\n"
    "Если название неоднозначно, добавьте форму и дозировку.\n\n"
    "За один раз проверяется только одна пара препаратов.\n\n"
    "Если препарат не поддерживается или название нельзя определить точно, "
    "бот попросит уточнить, а не будет угадывать."
)
PRIVACY_TEXT = (
    "Используйте бота только в личном чате. Не отправляйте ФИО, телефон, адрес, "
    "номер полиса, фотографии документов и другие персональные данные. Для проверки "
    "достаточно названий препаратов, формы и дозировки."
)
PRIVATE_CHAT_TEXT = (
    "Для защиты медицинской информации проверка доступна только в личном чате с ботом."
)
NON_TEXT_TEXT = "Отправьте названия двух препаратов текстом, например: Препарат А + Препарат Б."
CHECK_USAGE_TEXT = "После /check укажите два препарата: /check Препарат А + Препарат Б"
UNKNOWN_COMMAND_TEXT = "Неизвестная команда. Используйте /help или отправьте два препарата."
TECHNICAL_ERROR_TEXT = (
    "Запрос не обработан из-за технической ошибки: медицинский вывод не "
    "сформирован. Попробуйте позже и не меняйте лечение самостоятельно."
)

COMMAND_RE = re.compile(
    r"^/(?P<command>[a-z0-9_]+)(?:@[a-z0-9_]+)?(?:\s+(?P<payload>.*))?$",
    re.IGNORECASE | re.DOTALL,
)


class TelegramExternalModel(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class TelegramUser(TelegramExternalModel):
    id: int
    is_bot: bool = False


class TelegramChat(TelegramExternalModel):
    id: int
    type: str


class TelegramMessage(TelegramExternalModel):
    message_id: int
    message_thread_id: int | None = None
    sender: TelegramUser | None = Field(default=None, alias="from")
    chat: TelegramChat
    text: str | None = None


class TelegramUpdate(TelegramExternalModel):
    update_id: int
    message: TelegramMessage | None = None


class CheckService(Protocol):
    """Application layer implemented by the complete medical pipeline."""

    def check(self, message: str, *, request_id: str) -> AnalysisResult: ...


class UserFacingCheckError(RuntimeError):
    """Expected safe error or clarification generated before model analysis."""


class UpdateClaim(StrEnum):
    ACQUIRED = "acquired"
    COMPLETED = "completed"
    IN_PROGRESS = "in_progress"


class UpdateStore(Protocol):
    def claim(self, update_id: int) -> UpdateClaim: ...

    def complete(self, update_id: int) -> None: ...

    def release(self, update_id: int) -> None: ...


class RecentUpdateStore:
    """Bounded warm-instance protection against duplicate Telegram deliveries."""

    def __init__(self, *, ttl_seconds: float = 3600, max_entries: int = 10_000) -> None:
        if ttl_seconds <= 0 or max_entries <= 0:
            raise ValueError("update store limits must be positive")
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._records: dict[int, tuple[UpdateClaim, float]] = {}
        self._lock = threading.Lock()

    def claim(self, update_id: int) -> UpdateClaim:
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            existing = self._records.get(update_id)
            if existing is not None:
                return existing[0]
            self._records[update_id] = (UpdateClaim.IN_PROGRESS, now + self._ttl)
            return UpdateClaim.ACQUIRED

    def complete(self, update_id: int) -> None:
        with self._lock:
            self._records[update_id] = (UpdateClaim.COMPLETED, time.monotonic() + self._ttl)

    def release(self, update_id: int) -> None:
        with self._lock:
            self._records.pop(update_id, None)

    def _prune(self, now: float) -> None:
        expired = [key for key, (_, deadline) in self._records.items() if deadline <= now]
        for key in expired:
            self._records.pop(key, None)
        overflow = len(self._records) - self._max_entries + 1
        if overflow > 0:
            oldest = sorted(self._records, key=lambda key: self._records[key][1])[:overflow]
            for key in oldest:
                self._records.pop(key, None)


class TelegramWebhook:
    """Verify, route and acknowledge one Telegram update."""

    def __init__(
        self,
        *,
        webhook_secret: str,
        bot_client: TelegramBotClient,
        check_service: CheckService,
        update_store: UpdateStore | None = None,
        analytics: DemoAnalyticsRecorder | None = None,
        welcome_text: str = WELCOME_TEXT,
        ingredient_names: Mapping[str, str] | None = None,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", webhook_secret):
            raise ValueError("webhook_secret has an invalid Telegram format")
        self._secret = webhook_secret
        self._bot = bot_client
        self._check_service = check_service
        self._welcome_text = welcome_text
        self._ingredient_names = dict(ingredient_names or {})
        self._update_store = update_store or RecentUpdateStore()
        self._analytics = analytics or DemoAnalyticsRecorder(
            hmac_key=None, usage_mode="disabled"
        )

    def handle(self, event: Mapping[str, Any], *, request_id: str) -> dict[str, Any]:
        if not self._valid_secret(event.get("headers")):
            return _http_response(401, {"ok": False})
        try:
            update = TelegramUpdate.model_validate_json(_event_body(event))
        except (ValueError, TypeError, binascii.Error, ValidationError):
            return _http_response(400, {"ok": False})

        message = update.message
        if message is None or (message.sender is not None and message.sender.is_bot):
            return _http_response(200, {"ok": True})

        claim = self._update_store.claim(update.update_id)
        if claim is UpdateClaim.COMPLETED:
            return _http_response(200, {"ok": True})
        if claim is UpdateClaim.IN_PROGRESS:
            logger.warning("telegram_update_already_in_progress update_id=%s", update.update_id)
            return _http_response(200, {"ok": True})

        try:
            reply = self._reply_for(message, update_id=update.update_id, request_id=request_id)
            if reply is not None:
                self._bot.send_message(
                    chat_id=message.chat.id,
                    text=reply,
                    message_thread_id=message.message_thread_id,
                )
        except TelegramAPIError as exc:
            logger.error(
                "telegram_reply_failed update_id=%s code=%s retryable=%s",
                update.update_id,
                exc.code,
                exc.retryable,
            )
            self._update_store.complete(update.update_id)
            return _http_response(200, {"ok": True})
        except Exception as exc:
            logger.error(
                "telegram_update_failed update_id=%s error_type=%s",
                update.update_id,
                type(exc).__name__,
            )
            self._update_store.complete(update.update_id)
            return _http_response(200, {"ok": True})
        self._update_store.complete(update.update_id)
        return _http_response(200, {"ok": True})

    def _reply_for(
        self,
        message: TelegramMessage,
        *,
        update_id: int,
        request_id: str,
    ) -> str | None:
        if message.chat.type != "private":
            return PRIVATE_CHAT_TEXT
        if message.text is None:
            return NON_TEXT_TEXT
        text = message.text.strip()
        if not text:
            return NON_TEXT_TEXT
        if len(text) > MAX_USER_TEXT_LENGTH:
            return "Сообщение слишком длинное. Укажите только два препарата, форму и дозировку."

        command = COMMAND_RE.fullmatch(text)
        if command:
            name = command.group("command").casefold()
            payload = (command.group("payload") or "").strip()
            if name == "start":
                return self._welcome_text
            if name == "help":
                return HELP_TEXT
            if name == "privacy":
                return PRIVACY_TEXT
            if name == "check":
                if not payload:
                    return CHECK_USAGE_TEXT
                text = payload
            else:
                return UNKNOWN_COMMAND_TEXT

        safe_request_id = _request_id(request_id, update_id)
        sender_id = message.sender.id if message.sender is not None else None
        started = time.monotonic()
        try:
            result = self._check_service.check(text, request_id=safe_request_id)
        except UserFacingCheckError as exc:
            self._record_analytics(
                sender_id=sender_id,
                started=started,
                outcome=AnalyticsOutcome.CLARIFICATION_NEEDED,
                interaction_level=None,
            )
            message_text = str(exc).strip()
            return _limited_reply(message_text or TECHNICAL_ERROR_TEXT)
        except YandexGPTError as exc:
            logger.error(
                "medical_check_failed update_id=%s error_type=YandexGPTError "
                "error_code=%s retryable=%s",
                update_id,
                exc.code,
                exc.retryable,
            )
            self._record_analytics(
                sender_id=sender_id,
                started=started,
                outcome=AnalyticsOutcome.TECHNICAL_ERROR,
                interaction_level=None,
            )
            return TECHNICAL_ERROR_TEXT
        except Exception as exc:
            logger.error(
                "medical_check_failed update_id=%s error_type=%s",
                update_id,
                type(exc).__name__,
            )
            self._record_analytics(
                sender_id=sender_id,
                started=started,
                outcome=AnalyticsOutcome.TECHNICAL_ERROR,
                interaction_level=None,
            )
            return TECHNICAL_ERROR_TEXT
        outcome, level = outcome_for_result(result)
        self._record_analytics(
            sender_id=sender_id,
            started=started,
            outcome=outcome,
            interaction_level=level,
        )
        return format_analysis_result(result, ingredient_names=self._ingredient_names)

    def _record_analytics(
        self,
        *,
        sender_id: int | None,
        started: float,
        outcome: AnalyticsOutcome,
        interaction_level: InteractionLevel | None,
    ) -> None:
        # Analytics must never affect the Telegram reply (D-012): guard the
        # call site itself, not just trust the recorder's own error handling.
        try:
            latency_ms = (time.monotonic() - started) * 1000
            self._analytics.record(
                sender_id=sender_id,
                outcome=outcome,
                interaction_level=interaction_level,
                resolution=resolution_for_outcome(outcome),
                latency_ms=latency_ms,
            )
        except Exception:
            logger.warning("demo_analytics_call_failed")

    def _valid_secret(self, headers: object) -> bool:
        if not isinstance(headers, Mapping):
            return False
        normalized = {str(key).casefold(): value for key, value in headers.items()}
        supplied = normalized.get(SECRET_HEADER)
        return isinstance(supplied, str) and hmac.compare_digest(supplied, self._secret)


def webhook_registration_payload(
    *,
    url: str,
    secret_token: str,
    ip_address: str | None = None,
) -> dict[str, object]:
    """Build the stage-8 setWebhook payload without making an external call."""

    if not url.startswith("https://"):
        raise ValueError("Telegram webhook URL must use HTTPS")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", secret_token):
        raise ValueError("secret_token has an invalid Telegram format")
    payload: dict[str, object] = {
        "url": url,
        "allowed_updates": ["message"],
        "drop_pending_updates": False,
        "secret_token": secret_token,
    }
    if ip_address is not None:
        try:
            parsed = ipaddress.ip_address(ip_address)
        except ValueError as exc:
            raise ValueError("ip_address must be a valid IPv4 address") from exc
        if parsed.version != 4:
            raise ValueError("ip_address must be a valid IPv4 address")
        payload["ip_address"] = str(parsed)
    return payload


def bot_commands_payload() -> dict[str, object]:
    """Build a private-chat setMyCommands payload without making an API call."""

    return {
        "commands": [
            {"command": "start", "description": "Что делает бот и как им пользоваться"},
            {"command": "check", "description": "Проверить два препарата"},
            {"command": "help", "description": "Как пользоваться ботом"},
            {"command": "privacy", "description": "Какие данные не нужно отправлять"},
        ],
        "scope": {"type": "all_private_chats"},
        "language_code": "ru",
    }


def _event_body(event: Mapping[str, Any]) -> bytes:
    body = event.get("body")
    if not isinstance(body, str):
        raise ValueError("event body must be a string")
    if event.get("isBase64Encoded") is True:
        decoded = base64.b64decode(body, validate=True)
    else:
        decoded = body.encode("utf-8")
    if len(decoded) > MAX_WEBHOOK_BODY_BYTES:
        raise ValueError("event body is too large")
    return decoded


def _request_id(context_request_id: str, update_id: int) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.:-]+", "-", context_request_id).strip("-")
    return f"tg-{update_id}-{safe or 'request'}"[:120]


def _limited_reply(value: str) -> str:
    if len(value) <= MAX_USER_TEXT_LENGTH:
        return value
    suffix = "\n\nСообщение сокращено."
    return value[: MAX_USER_TEXT_LENGTH - len(suffix)].rstrip() + suffix


def _http_response(status: int, body: dict[str, object]) -> dict[str, Any]:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json; charset=utf-8"},
        "body": json.dumps(body, ensure_ascii=False),
    }
