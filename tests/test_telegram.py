import base64
import json
import logging
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError

import pytest

from pharmacheck.analytics import DemoAnalyticsRecorder
from pharmacheck.cloud_runtime import load_ingredient_names, welcome_text_for
from pharmacheck.contracts import MAX_SOURCE_URL_LENGTH, AnalysisResult
from pharmacheck.telegram import (
    TELEGRAM_TEXT_LIMIT,
    TelegramAPIError,
    TelegramBotClient,
    format_analysis_result,
)
from pharmacheck.telegram_webhook import (
    NONCOMMERCIAL_WELCOME_TEXT,
    PRIVATE_CHAT_TEXT,
    TECHNICAL_ERROR_TEXT,
    WELCOME_TEXT,
    RecentUpdateStore,
    TelegramWebhook,
    UserFacingCheckError,
    bot_commands_payload,
    webhook_registration_payload,
)
from pharmacheck.yandexgpt import YandexGPTError

SECRET = "test-secret"
ANALYTICS_KEY = b"a" * 32


class FakeBot:
    def __init__(self):
        self.messages = []
        self.error = None

    def send_message(self, **payload):
        if self.error:
            raise self.error
        self.messages.append(payload)


class FakeCheckService:
    def __init__(self):
        self.calls = []
        self.error = None
        self.result = None

    def check(self, message, *, request_id):
        self.calls.append((message, request_id))
        if self.error:
            raise self.error
        return self.result or _green_result()


def _green_result() -> AnalysisResult:
    return AnalysisResult.model_validate(
        {
            "status": "answer",
            "interaction_level": "green",
            "headline": "Значимого известного взаимодействия не найдено",
            "identified_drugs": [
                {"trade_name": "Альфа", "active_ingredients": ["вещество альфа"]},
                {"trade_name": "Бета", "active_ingredients": ["вещество бета"]},
            ],
            "summary": "В проверенных источниках значимого известного взаимодействия не "
            "найдено, но это не доказывает безопасность комбинации для конкретного человека.",
            "mechanism_or_effect": None,
            "recommended_user_action": "Не изменяйте лечение без консультации врача.",
            "limitations": ["Индивидуальные факторы не оценивались."],
            "conflicts": [],
            "missing_information": [],
            "sources": [
                {
                    "source_id": "openfda.label",
                    "title": "FDA label",
                    "url": "https://example.test/fda",
                    "supports": "В проверенных разделах прямое упоминание не найдено.",
                }
            ],
        }
    )


def _white_result(status: str, *, headline="Нужно уточнить препараты") -> AnalysisResult:
    return AnalysisResult.model_validate(
        {
            "status": status,
            "interaction_level": "white",
            "headline": headline,
            "identified_drugs": [],
            "summary": "Медицинский вывод пока не сформирован.",
            "mechanism_or_effect": None,
            "recommended_user_action": "Уточните названия препаратов.",
            "limitations": [],
            "conflicts": [],
            "missing_information": ["Название не распознано."],
            "sources": [],
        }
    )


def _urgent_result() -> AnalysisResult:
    # status=urgent always pairs with interaction_level=white: it means the
    # user's message itself reports dangerous acute symptoms, so the
    # interaction check is skipped in favor of urging emergency care — it
    # is never a statement about interaction severity (contracts.py
    # AnalysisResult.status_matches_level enforces this).
    raw = _green_result().model_dump(mode="json")
    raw["status"] = "urgent"
    raw["interaction_level"] = "white"
    raw["headline"] = "Возможны опасные симптомы"
    raw["identified_drugs"] = []
    raw["sources"] = []
    return AnalysisResult.model_validate(raw)


def _capturing_analytics():
    logger = logging.getLogger("test.telegram_webhook.analytics." + str(id(object())))
    logger.setLevel(logging.INFO)
    records: list[str] = []

    class Handler(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logger.addHandler(Handler())
    logger.propagate = False
    recorder = DemoAnalyticsRecorder(
        hmac_key=ANALYTICS_KEY, usage_mode="official_labels_demo", logger=logger
    )
    return recorder, records


def _event(
    text="Альфа + Бета",
    *,
    update_id=99,
    chat_type="private",
    sender_is_bot=False,
    thread_id=None,
    secret=SECRET,
):
    message = {
        "message_id": 7,
        "from": {"id": 42, "is_bot": sender_is_bot},
        "chat": {"id": 42, "type": chat_type},
    }
    if text is not None:
        message["text"] = text
    if thread_id is not None:
        message["message_thread_id"] = thread_id
    return {
        "headers": {"X-Telegram-Bot-Api-Secret-Token": secret},
        "body": json.dumps({"update_id": update_id, "message": message}),
        "isBase64Encoded": False,
    }


def _webhook(*, analytics=None):
    bot = FakeBot()
    service = FakeCheckService()
    return TelegramWebhook(
        webhook_secret=SECRET,
        bot_client=bot,
        check_service=service,
        analytics=analytics,
    ), bot, service


def test_plain_private_message_runs_check_and_formats_result():
    webhook, bot, service = _webhook()

    response = webhook.handle(_event(), request_id="function/request 1")

    assert response["statusCode"] == 200
    assert service.calls == [("Альфа + Бета", "tg-99-function-request-1")]
    assert bot.messages[0]["chat_id"] == 42
    assert bot.messages[0]["text"].startswith("🟢")
    assert "https://example.test/fda" in bot.messages[0]["text"]


def test_check_command_passes_only_payload_and_preserves_thread():
    webhook, bot, service = _webhook()

    webhook.handle(
        _event("/check@PharmaCheckBot Альфа + Бета", thread_id=13),
        request_id="req",
    )

    assert service.calls[0][0] == "Альфа + Бета"
    assert bot.messages[0]["message_thread_id"] == 13


@pytest.mark.parametrize("command", ["/start", "/help", "/privacy", "/check", "/unknown"])
def test_commands_do_not_run_medical_check(command):
    webhook, bot, service = _webhook()

    response = webhook.handle(_event(command), request_id="req")

    assert response["statusCode"] == 200
    assert not service.calls
    assert bot.messages


def test_group_chat_does_not_expose_or_analyze_drug_names():
    webhook, bot, service = _webhook()

    webhook.handle(_event("Секретный препарат + Бета", chat_type="group"), request_id="req")

    assert not service.calls
    assert bot.messages[0]["text"] == PRIVATE_CHAT_TEXT
    assert "Секретный" not in bot.messages[0]["text"]


def test_non_text_message_gets_text_instruction():
    webhook, bot, service = _webhook()
    webhook.handle(_event(None), request_id="req")
    assert not service.calls
    assert "текстом" in bot.messages[0]["text"]


def test_bot_sender_and_unsupported_update_are_acknowledged_without_reply():
    webhook, bot, service = _webhook()
    first = webhook.handle(_event(sender_is_bot=True), request_id="req")
    second_event = {
        "headers": {"x-telegram-bot-api-secret-token": SECRET},
        "body": json.dumps({"update_id": 100, "callback_query": {"id": "x"}}),
    }
    second = webhook.handle(second_event, request_id="req")
    assert first["statusCode"] == second["statusCode"] == 200
    assert not bot.messages
    assert not service.calls


def test_secret_is_checked_before_body_parsing():
    webhook, bot, service = _webhook()
    event = {"headers": {}, "body": "not-json"}
    response = webhook.handle(event, request_id="req")
    assert response["statusCode"] == 401
    assert not bot.messages
    assert not service.calls


def test_base64_api_gateway_body_is_supported():
    webhook, bot, service = _webhook()
    event = _event("Альфа + Бета")
    event["body"] = base64.b64encode(event["body"].encode()).decode()
    event["isBase64Encoded"] = True
    response = webhook.handle(event, request_id="req")
    assert response["statusCode"] == 200
    assert service.calls


def test_invalid_json_returns_400():
    webhook, bot, service = _webhook()
    event = _event()
    event["body"] = "not-json"
    response = webhook.handle(event, request_id="req")
    assert response["statusCode"] == 400
    assert not bot.messages


def test_safe_workflow_error_is_shown_and_unexpected_error_is_sanitized():
    webhook, bot, service = _webhook()
    service.error = UserFacingCheckError("Уточните дозировку второго препарата.")
    webhook.handle(_event(update_id=100), request_id="req")
    assert bot.messages[-1]["text"] == "Уточните дозировку второго препарата."

    service.error = RuntimeError("internal secret")
    webhook.handle(_event(), request_id="req")
    assert bot.messages[-1]["text"] == TECHNICAL_ERROR_TEXT
    assert "internal secret" not in bot.messages[-1]["text"]


def test_safe_workflow_error_is_limited_to_one_telegram_message():
    webhook, bot, service = _webhook()
    service.error = UserFacingCheckError("уточнение " * 1000)
    webhook.handle(_event(), request_id="req")
    assert len(bot.messages[-1]["text"]) <= TELEGRAM_TEXT_LIMIT


def test_telegram_failure_is_acknowledged_without_webhook_retry():
    webhook, bot, service = _webhook()
    bot.error = TelegramAPIError("safe", code="network", retryable=True)
    response = webhook.handle(_event("/start"), request_id="req")
    assert response["statusCode"] == 200


def test_completed_update_is_not_processed_twice():
    webhook, bot, service = _webhook()
    first = webhook.handle(_event(), request_id="first")
    second = webhook.handle(_event(), request_id="second")

    assert first["statusCode"] == second["statusCode"] == 200
    assert len(service.calls) == 1
    assert len(bot.messages) == 1


def test_in_progress_update_is_acknowledged_without_duplicate_analysis():
    store = RecentUpdateStore()
    assert store.claim(99).value == "acquired"
    bot = FakeBot()
    service = FakeCheckService()
    webhook = TelegramWebhook(
        webhook_secret=SECRET,
        bot_client=bot,
        check_service=service,
        update_store=store,
    )

    response = webhook.handle(_event(), request_id="second")

    assert response["statusCode"] == 200
    assert not service.calls
    assert not bot.messages


def test_telegram_failure_completes_update_without_repeating_paid_analysis():
    webhook, bot, service = _webhook()
    bot.error = TelegramAPIError("safe", code="network", retryable=True)
    assert webhook.handle(_event(), request_id="first")["statusCode"] == 200

    bot.error = None
    assert webhook.handle(_event(), request_id="second")["statusCode"] == 200
    assert len(service.calls) == 1
    assert not bot.messages


def test_start_help_privacy_texts_cover_the_specified_content():
    webhook, bot, service = _webhook()

    webhook.handle(_event("/start", update_id=1), request_id="req")
    start_text = bot.messages[-1]["text"]
    assert "Нурофен + Варфарин" in start_text
    assert "амиодарон + вориконазол" in start_text
    assert "Препарат А + Препарат Б" in start_text
    assert "не доказывает безопасность" in start_text
    assert "не ставит диагнозы" in start_text
    assert "не назначает" in start_text
    assert "не отменяет" in start_text
    assert "не заменяет" in start_text
    assert "не изменяет" in start_text
    assert "/help" in start_text
    assert "/privacy" in start_text

    webhook.handle(_event("/help", update_id=2), request_id="req")
    help_text = bot.messages[-1]["text"]
    assert "ровно два препарата" in help_text
    assert "/check" in help_text
    assert "форму и дозировку" in help_text
    assert "одна пара" in help_text
    assert "уточнить" in help_text

    webhook.handle(_event("/privacy", update_id=3), request_id="req")
    assert "личном чате" in bot.messages[-1]["text"]


def test_technical_error_text_states_no_conclusion_and_try_later():
    webhook, bot, service = _webhook()
    service.error = RuntimeError("internal secret")

    webhook.handle(_event(), request_id="req")

    text = bot.messages[-1]["text"]
    assert "не обработан" in text
    assert "медицинский вывод не сформирован" in text
    assert "позже" in text
    assert "не меняйте лечение" in text.casefold()
    assert "internal secret" not in text


def test_bot_commands_include_start_for_the_private_menu():
    payload = bot_commands_payload()
    assert "start" in [item["command"] for item in payload["commands"]]


def test_at_most_three_sources_are_shown_with_omitted_count():
    raw = _green_result().model_dump(mode="json")
    raw["sources"] = [
        {
            "source_id": f"openfda.label.{index}",
            "title": f"FDA label {index}",
            "url": f"https://example.test/fda/{index}",
            "supports": f"ВНУТРЕННЕЕ ПОЯСНЕНИЕ ИСТОЧНИКА {index}",
        }
        for index in range(5)
    ]
    text = format_analysis_result(AnalysisResult.model_validate(raw))

    for index in range(3):
        assert f"https://example.test/fda/{index}" in text
    for index in range(3, 5):
        assert f"https://example.test/fda/{index}" not in text
    assert "Ещё источников: 2." in text
    assert "ВНУТРЕННЕЕ ПОЯСНЕНИЕ ИСТОЧНИКА" not in text


def test_first_source_url_survives_near_limit_truncation():
    raw = _green_result().model_dump(mode="json")
    raw["summary"] = "Очень длинный текст. " * 60
    raw["limitations"] = ["Ограничение. " * 30 for _ in range(4)]
    raw["conflicts"] = ["Расхождение. " * 30 for _ in range(3)]
    raw["missing_information"] = ["Уточнение. " * 30 for _ in range(4)]
    raw["mechanism_or_effect"] = "Механизм. " * 80
    raw["sources"] = [
        {
            "source_id": f"openfda.label.{index}",
            "title": f"FDA label {index}",
            "url": f"https://example.test/fda/{index}",
            "supports": "Найдено прямое упоминание пары. " * 20,
        }
        for index in range(3)
    ]

    text = format_analysis_result(AnalysisResult.model_validate(raw))

    assert len(text) <= TELEGRAM_TEXT_LIMIT
    assert "1. FDA label 0\nhttps://example.test/fda/0" in text
    for line in text.splitlines():
        if line.startswith("http"):
            assert line.strip() == line
            assert line.endswith(tuple(f"/fda/{i}" for i in range(3)))


def test_first_source_url_survives_extreme_over_limit_truncation():
    raw = _green_result().model_dump(mode="json")
    raw["summary"] = "Очень длинный текст без остановки. " * 400
    raw["sources"] = [
        {
            "source_id": "openfda.label.only",
            "title": "FDA label",
            "url": "https://example.test/fda/only",
            "supports": "Найдено прямое упоминание пары. " * 100,
        }
    ]

    text = format_analysis_result(AnalysisResult.model_validate(raw))

    assert len(text) <= TELEGRAM_TEXT_LIMIT
    assert "https://example.test/fda/only" in text
    assert text.count("https://example.test/fda/only") == 1


def test_pathologically_long_mandatory_fields_still_fit_the_limit():
    # Reproduces the ChatGPT/Codex-reported 11,683-character overflow: every
    # *mandatory* field (headline, drug names, ingredients, action, first
    # source title) is unbounded by schema (`NonEmptyText` has no max
    # length), not just the optional `summary`.
    raw = _green_result().model_dump(mode="json")
    raw["headline"] = "Очень длинный и небезопасно длинный заголовок результата. " * 40
    raw["identified_drugs"] = [
        {
            "trade_name": "Очень длинное торговое название препарата номер один. " * 10,
            "active_ingredients": [
                f"Очень длинное действующее вещество номер {index}. " * 8
                for index in range(10)
            ],
        },
        {
            "trade_name": "Очень длинное торговое название препарата номер два. " * 10,
            "active_ingredients": ["Вещество бета"],
        },
    ]
    raw["summary"] = "Очень длинный текст без остановки. " * 400
    raw["recommended_user_action"] = "Обсудите с врачом. " * 200
    raw["mechanism_or_effect"] = "Механизм действия. " * 200
    raw["limitations"] = ["Ограничение. " * 100 for _ in range(10)]
    raw["conflicts"] = ["Расхождение. " * 100 for _ in range(10)]
    raw["missing_information"] = []
    raw["sources"] = [
        {
            "source_id": f"openfda.label.{index}",
            "title": f"Очень длинное название источника номер {index}. " * 20,
            "url": f"https://example.test/fda/{index}",
            "supports": "Найдено прямое упоминание пары. " * 100,
        }
        for index in range(3)
    ]

    text = format_analysis_result(AnalysisResult.model_validate(raw))

    assert 1 <= len(text) <= TELEGRAM_TEXT_LIMIT
    # The first source's URL is mandatory whenever result.sources is
    # non-empty: it must appear, whole, exactly once, on its own line.
    first_url = "https://example.test/fda/0"
    assert text.count(first_url) == 1
    line = next(line for line in text.splitlines() if line == first_url)
    assert line == first_url
    # Sources 2 and 3, if shown at all, are also never partially cut.
    for index in (1, 2):
        url = f"https://example.test/fda/{index}"
        assert text.count(url) in (0, 1)
    # A source count is never shown without at least one displayed source.
    if "Ещё источников:" in text:
        assert "Источники:\n1." in text


def test_long_but_valid_first_source_url_survives_with_two_drugs_and_long_fields():
    # Every mandatory field is long enough, together, that a *fixed*-cap
    # budget for header+drugs+action (as in the pre-shrink implementation)
    # leaves no room for this realistic-but-long DailyMed query-string URL,
    # so the URL would be dropped rather than shrinking the other fields
    # further. This is what actually distinguishes dynamic shrinking from
    # fixed clipping — a shorter combination passes on both.
    raw = _green_result().model_dump(mode="json")
    raw["headline"] = "Очень длинный и небезопасно длинный заголовок результата. " * 40
    raw["identified_drugs"] = [
        {
            "trade_name": "Очень длинное торговое название препарата номер один. " * 10,
            "active_ingredients": [
                f"Очень длинное действующее вещество номер {index}. " * 8
                for index in range(10)
            ],
        },
        {
            "trade_name": "Очень длинное торговое название препарата номер два. " * 10,
            "active_ingredients": ["Вещество бета"],
        },
    ]
    raw["summary"] = "Очень длинный текст без остановки. " * 400
    raw["recommended_user_action"] = "Обсудите с врачом. " * 200
    raw["mechanism_or_effect"] = "Механизм действия. " * 200
    raw["limitations"] = ["Ограничение. " * 100 for _ in range(10)]
    raw["conflicts"] = ["Расхождение. " * 100 for _ in range(10)]
    raw["missing_information"] = []
    long_valid_url = (
        "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json?"
        + "&".join(f"param{index}=value-{index}-detail" for index in range(100))
    )
    raw["sources"] = [
        {
            "source_id": "dailymed.label.long",
            "title": "Очень длинное название источника с описанием. " * 20,
            "url": long_valid_url,
            "supports": "Найдено прямое упоминание пары. " * 100,
        }
    ]

    text = format_analysis_result(AnalysisResult.model_validate(raw))

    assert 1 <= len(text) <= TELEGRAM_TEXT_LIMIT
    assert text.count(long_valid_url) == 1
    line = next(line for line in text.splitlines() if line == long_valid_url)
    assert line == long_valid_url


def test_extremely_long_headline_alone_still_fits_the_limit():
    raw = _green_result().model_dump(mode="json")
    raw["headline"] = "Заголовок. " * 2000
    raw["identified_drugs"] = []
    raw["mechanism_or_effect"] = None
    raw["limitations"] = []
    raw["conflicts"] = []
    raw["missing_information"] = []
    raw["sources"] = []

    text = format_analysis_result(AnalysisResult.model_validate(raw))

    assert 1 <= len(text) <= TELEGRAM_TEXT_LIMIT


def test_source_url_at_the_max_length_boundary_is_shown_in_full():
    # Ties the contract-level MAX_SOURCE_URL_LENGTH (contracts.py) back to
    # the formatter guarantee it was derived to protect: a first-source URL
    # of exactly that length, combined with every other mandatory field at
    # its own worst realistic case, must still fit and be shown whole.
    raw = _green_result().model_dump(mode="json")
    raw["headline"] = "Очень длинный и небезопасно длинный заголовок результата. " * 40
    raw["identified_drugs"] = [
        {
            "trade_name": "Очень длинное торговое название препарата номер один. " * 10,
            "active_ingredients": [
                f"Очень длинное действующее вещество номер {index}. " * 8
                for index in range(10)
            ],
        },
        {
            "trade_name": "Очень длинное торговое название препарата номер два. " * 10,
            "active_ingredients": ["Вещество бета"],
        },
    ]
    raw["summary"] = "Очень длинный текст без остановки. " * 400
    raw["recommended_user_action"] = "Обсудите с врачом. " * 200
    raw["mechanism_or_effect"] = "Механизм действия. " * 200
    raw["limitations"] = ["Ограничение. " * 100 for _ in range(10)]
    raw["conflicts"] = ["Расхождение. " * 100 for _ in range(10)]
    raw["missing_information"] = []
    url_prefix = "https://example.test/"
    boundary_url = url_prefix + "a" * (MAX_SOURCE_URL_LENGTH - len(url_prefix))
    assert len(boundary_url) == MAX_SOURCE_URL_LENGTH
    raw["sources"] = [
        {
            "source_id": "openfda.label.boundary",
            "title": "Очень длинное название источника с описанием. " * 20,
            "url": boundary_url,
            "supports": "Найдено прямое упоминание пары. " * 100,
        }
    ]

    text = format_analysis_result(AnalysisResult.model_validate(raw))

    assert 1 <= len(text) <= TELEGRAM_TEXT_LIMIT
    assert text.count(boundary_url) == 1
    line = next(line for line in text.splitlines() if line == boundary_url)
    assert line == boundary_url


@pytest.mark.parametrize(
    "build_result,expected_outcome,expected_level,expected_resolution",
    [
        (lambda: _green_result(), "answer", "green", "resolved"),
        (
            lambda: _white_result("insufficient_data"),
            "insufficient_data",
            "white",
            "resolved",
        ),
        (lambda: _urgent_result(), "urgent", "white", "resolved"),
    ],
)
def test_analytics_event_for_accepted_check_outcomes(
    build_result, expected_outcome, expected_level, expected_resolution
):
    analytics, records = _capturing_analytics()
    webhook, bot, service = _webhook(analytics=analytics)
    service.result = build_result()

    webhook.handle(_event(), request_id="req")

    assert len(records) == 1
    event = json.loads(records[0])
    assert event["outcome"] == expected_outcome
    assert event["interaction_level"] == expected_level
    assert event["resolution"] == expected_resolution
    assert event["anonymous_user"]


def test_analytics_event_for_clarification_needed_via_user_facing_error():
    analytics, records = _capturing_analytics()
    webhook, bot, service = _webhook(analytics=analytics)
    service.error = UserFacingCheckError("Уточните дозировку второго препарата.")

    webhook.handle(_event(), request_id="req")

    assert len(records) == 1
    event = json.loads(records[0])
    assert event["outcome"] == "clarification_needed"
    assert event["resolution"] == "needs_clarification"
    assert "interaction_level" not in event


@pytest.mark.parametrize(
    "error",
    [RuntimeError("internal secret"), YandexGPTError("bad json", code="invalid_model_response")],
)
def test_analytics_event_for_technical_error(error):
    analytics, records = _capturing_analytics()
    webhook, bot, service = _webhook(analytics=analytics)
    service.error = error

    webhook.handle(_event(), request_id="req")

    assert len(records) == 1
    event = json.loads(records[0])
    assert event["outcome"] == "technical_error"
    assert event["resolution"] == "unknown_on_error"
    assert "interaction_level" not in event


@pytest.mark.parametrize("command", ["/start", "/help", "/privacy", "/check", "/unknown"])
def test_no_analytics_event_for_commands(command):
    analytics, records = _capturing_analytics()
    webhook, bot, service = _webhook(analytics=analytics)

    webhook.handle(_event(command), request_id="req")

    assert records == []


def test_no_analytics_event_for_rejected_secret():
    analytics, records = _capturing_analytics()
    webhook, bot, service = _webhook(analytics=analytics)

    webhook.handle({"headers": {}, "body": "not-json"}, request_id="req")

    assert records == []


def test_no_analytics_event_for_non_private_chat():
    analytics, records = _capturing_analytics()
    webhook, bot, service = _webhook(analytics=analytics)

    webhook.handle(_event("Секретный препарат + Бета", chat_type="group"), request_id="req")

    assert records == []


def test_no_analytics_event_for_ignored_update():
    analytics, records = _capturing_analytics()
    webhook, bot, service = _webhook(analytics=analytics)

    webhook.handle(_event(sender_is_bot=True), request_id="req")

    assert records == []


def test_no_analytics_event_for_duplicate_updates():
    analytics, records = _capturing_analytics()
    webhook, bot, service = _webhook(analytics=analytics)

    webhook.handle(_event(update_id=555), request_id="first")
    webhook.handle(_event(update_id=555), request_id="second")

    assert len(records) == 1


def test_no_analytics_event_for_in_progress_duplicate():
    store = RecentUpdateStore()
    assert store.claim(777).value == "acquired"
    analytics, records = _capturing_analytics()
    bot = FakeBot()
    service = FakeCheckService()
    webhook = TelegramWebhook(
        webhook_secret=SECRET,
        bot_client=bot,
        check_service=service,
        update_store=store,
        analytics=analytics,
    )

    webhook.handle(_event(update_id=777), request_id="req")

    assert records == []


def test_analytics_failure_does_not_affect_the_reply():
    class BrokenRecorder:
        def record(self, **kwargs):
            raise RuntimeError("analytics sink is down")

    webhook, bot, service = _webhook(analytics=BrokenRecorder())

    response = webhook.handle(_event(), request_id="req")

    assert response["statusCode"] == 200
    assert bot.messages
    assert bot.messages[0]["text"].startswith("🟢")


def test_no_analytics_recorder_means_no_events_by_default():
    webhook, bot, service = _webhook()

    webhook.handle(_event(), request_id="req")

    assert bot.messages


def test_bot_client_sends_plain_json_without_paid_broadcast():
    captured = {}

    def transport(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return {"ok": True, "result": {"message_id": 1}}

    client = TelegramBotClient(bot_token="123:token", transport=transport)
    client.send_message(chat_id=42, text="Ответ", message_thread_id=8)

    assert captured["url"].endswith("/bot123:token/sendMessage")
    assert captured["payload"] == {
        "chat_id": 42,
        "text": "Ответ",
        "link_preview_options": {"is_disabled": True},
        "message_thread_id": 8,
    }
    assert "allow_paid_broadcast" not in captured["payload"]


def test_bot_client_registers_webhook_and_private_commands():
    calls = []

    def transport(request, timeout):
        calls.append((request.full_url, json.loads(request.data)))
        return {"ok": True, "result": True}

    client = TelegramBotClient(bot_token="123:token", transport=transport)
    webhook = webhook_registration_payload(
        url="https://gateway.apigw.yandexcloud.net/telegram",
        secret_token=SECRET,
    )
    commands = bot_commands_payload()

    client.set_webhook(webhook)
    client.set_my_commands(commands)

    assert calls == [
        ("https://api.telegram.org/bot123:token/setWebhook", webhook),
        ("https://api.telegram.org/bot123:token/setMyCommands", commands),
    ]


def test_bot_client_reads_safe_deployment_metadata():
    calls = []

    def transport(request, timeout):
        calls.append(request.full_url)
        if request.full_url.endswith("/getMe"):
            return {"ok": True, "result": {"username": "pharmacheck_bot"}}
        return {"ok": True, "result": {"url": "https://gateway/telegram"}}

    client = TelegramBotClient(bot_token="123:token", transport=transport)
    assert client.get_me()["result"]["username"] == "pharmacheck_bot"
    assert client.get_webhook_info()["result"]["url"] == "https://gateway/telegram"
    assert calls[-2:] == [
        "https://api.telegram.org/bot123:token/getMe",
        "https://api.telegram.org/bot123:token/getWebhookInfo",
    ]


def test_bot_client_uses_authenticated_relay_without_token_in_url():
    captured = {}

    def transport(request, timeout):
        captured["url"] = request.full_url
        captured["relay_secret"] = request.headers["X-pharmacheck-relay-secret"]
        captured["user_agent"] = request.headers["User-agent"]
        return {"ok": True, "result": {"message_id": 1}}

    client = TelegramBotClient(
        bot_token="123:very-secret",
        api_base="https://relay.example.test/telegram-api",
        relay_secret="relay-secret",
        transport=transport,
    )
    client.send_message(chat_id=42, text="Ответ")

    assert captured == {
        "url": "https://relay.example.test/telegram-api/sendMessage",
        "relay_secret": "relay-secret",
        "user_agent": "PharmaCheckAI/1.0",
    }
    assert "very-secret" not in captured["url"]


def test_bot_client_requires_secret_for_non_default_api_base():
    with pytest.raises(ValueError):
        TelegramBotClient(
            bot_token="123:token",
            api_base="https://relay.example.test/telegram-api",
        )


def test_bot_token_is_absent_from_safe_http_error():
    def transport(request, timeout):
        raise HTTPError(request.full_url, 500, "token leak", {}, BytesIO(b"secret"))

    client = TelegramBotClient(bot_token="123:very-secret", transport=transport)
    with pytest.raises(TelegramAPIError) as error:
        client.send_message(chat_id=42, text="Ответ")
    assert "very-secret" not in str(error.value)
    assert error.value.retryable


def test_formatter_never_exceeds_telegram_limit():
    raw = _green_result().model_dump(mode="json")
    raw["summary"] = "Очень длинный текст " * 1000
    raw["limitations"] = ["Ограничение " * 500 for _ in range(10)]
    text = format_analysis_result(AnalysisResult.model_validate(raw))
    assert len(text) <= TELEGRAM_TEXT_LIMIT


def test_webhook_registration_is_private_message_only_and_free_of_paid_options():
    payload = webhook_registration_payload(
        url="https://example.test/telegram",
        secret_token=SECRET,
    )
    assert payload["allowed_updates"] == ["message"]
    assert payload["drop_pending_updates"] is False
    assert "max_connections" not in payload


def test_webhook_registration_can_pin_a_valid_ipv4_address():
    payload = webhook_registration_payload(
        url="https://example.test/telegram",
        secret_token=SECRET,
        ip_address="158.160.171.31",
    )
    assert payload["ip_address"] == "158.160.171.31"

    with pytest.raises(ValueError):
        webhook_registration_payload(
            url="https://example.test/telegram",
            secret_token=SECRET,
            ip_address="2001:db8::1",
        )


def test_bot_commands_are_scoped_to_private_chats():
    payload = bot_commands_payload()
    assert payload["scope"] == {"type": "all_private_chats"}
    assert [item["command"] for item in payload["commands"]] == [
        "start",
        "check",
        "help",
        "privacy",
    ]


def test_start_keeps_the_published_demo_text_by_default():
    webhook, bot, _ = _webhook()
    webhook.handle(_event("/start"), request_id="req")
    assert bot.messages[-1]["text"] == WELCOME_TEXT
    assert welcome_text_for("official_labels_demo") == WELCOME_TEXT


def test_private_mvp_start_text_is_not_a_demo_and_keeps_safety_statements():
    assert welcome_text_for("noncommercial_mvp") == NONCOMMERCIAL_WELCOME_TEXT
    assert "демо" not in NONCOMMERCIAL_WELCOME_TEXT.casefold()
    assert "некоммерческ" in NONCOMMERCIAL_WELCOME_TEXT
    assert "DDInter" in NONCOMMERCIAL_WELCOME_TEXT
    # Everything after the product description is identical to the demo text.
    first_paragraph_end = WELCOME_TEXT.index("\n\n")
    assert NONCOMMERCIAL_WELCOME_TEXT.endswith(WELCOME_TEXT[first_paragraph_end:])
    for statement in (
        "Отсутствие найденной записи не доказывает безопасность комбинации.",
        "Сервис не ставит диагнозы",
    ):
        assert statement in NONCOMMERCIAL_WELCOME_TEXT


def test_webhook_sends_the_configured_start_text():
    bot = FakeBot()
    webhook = TelegramWebhook(
        webhook_secret=SECRET,
        bot_client=bot,
        check_service=FakeCheckService(),
        welcome_text=NONCOMMERCIAL_WELCOME_TEXT,
    )
    webhook.handle(_event("/start"), request_id="req")
    assert bot.messages[-1]["text"] == NONCOMMERCIAL_WELCOME_TEXT


def _aspirin_warfarin_result() -> AnalysisResult:
    raw = _green_result().model_dump(mode="json")
    raw["identified_drugs"] = [
        {"trade_name": "Кардиомагнил", "active_ingredients": ["aspirin", "magnesium hydroxide"]},
        {"trade_name": "Варфарин", "active_ingredients": ["warfarin", "unmapped ingredient"]},
    ]
    return AnalysisResult.model_validate(raw)


RUSSIAN_NAMES = {
    "aspirin": "Ацетилсалициловая кислота",
    "magnesium hydroxide": "Магния гидроксид",
    "warfarin": "Варфарин",
}


def test_ingredients_are_shown_in_russian_when_names_are_known():
    text = format_analysis_result(_aspirin_warfarin_result(), ingredient_names=RUSSIAN_NAMES)
    assert "Кардиомагнил — ацетилсалициловая кислота, магния гидроксид" in text
    # An ingredient without a Russian name keeps its English name.
    assert "Варфарин — варфарин, unmapped ingredient" in text
    assert "aspirin" not in text


def test_ingredients_stay_english_without_a_name_table():
    text = format_analysis_result(_aspirin_warfarin_result())
    assert "Кардиомагнил — aspirin, magnesium hydroxide" in text


def test_russian_names_respect_the_telegram_limit():
    long_names = {key: value * 60 for key, value in RUSSIAN_NAMES.items()}
    text = format_analysis_result(_aspirin_warfarin_result(), ingredient_names=long_names)
    assert 1 <= len(text) <= TELEGRAM_TEXT_LIMIT


def test_webhook_formats_results_with_its_ingredient_names():
    bot = FakeBot()
    service = FakeCheckService()
    service.result = _aspirin_warfarin_result()
    webhook = TelegramWebhook(
        webhook_secret=SECRET,
        bot_client=bot,
        check_service=service,
        ingredient_names=RUSSIAN_NAMES,
    )
    webhook.handle(_event(), request_id="req")
    assert "ацетилсалициловая кислота" in bot.messages[-1]["text"]


def test_ingredient_name_loader_is_never_fatal(tmp_path):
    good = tmp_path / "names.json"
    good.write_text(
        json.dumps(
            [
                {"russian_name": "Ацетилсалициловая кислота", "source_name": "aspirin"},
                {"russian_name": " ", "source_name": "blank"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    assert load_ingredient_names(good) == {"aspirin": "Ацетилсалициловая кислота"}
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert load_ingredient_names(broken) == {}
    assert load_ingredient_names(tmp_path / "missing.json") == {}


@pytest.mark.skipif(
    not (Path(__file__).parents[1] / "data/drug_catalog.json").is_file()
    or not (Path(__file__).parents[1] / "data/ingredient_mappings.json").is_file(),
    reason="Private catalog and display mappings are not distributed",
)
def test_repository_has_a_russian_name_for_every_catalog_ingredient():
    names = load_ingredient_names(Path(__file__).parents[1] / "data" / "ingredient_mappings.json")
    catalog = json.loads(
        (Path(__file__).parents[1] / "data" / "drug_catalog.json").read_text(encoding="utf-8")
    )
    ingredients = {item for entry in catalog for item in entry["active_ingredients"]}
    assert ingredients <= set(names)
