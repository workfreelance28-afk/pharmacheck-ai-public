"""Telegram Bot API client and plain-text result formatting."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pharmacheck.contracts import AnalysisResult, AnalysisStatus, InteractionLevel

TELEGRAM_API_BASE = "https://api.telegram.org"
TELEGRAM_TEXT_LIMIT = 4096
TELEGRAM_USER_AGENT = "PharmaCheckAI/1.0"


class TelegramAPIError(RuntimeError):
    """Safe Telegram error that never includes the bot token or response body."""

    def __init__(self, message: str, *, code: str, retryable: bool) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


TelegramTransport = Callable[[Request, float], Mapping[str, Any]]


class TelegramBotClient:
    """Minimal Bot API client needed by the MVP webhook."""

    def __init__(
        self,
        *,
        bot_token: str,
        timeout: float = 10.0,
        transport: TelegramTransport | None = None,
        api_base: str = TELEGRAM_API_BASE,
        relay_secret: str | None = None,
    ) -> None:
        token = bot_token.strip()
        if not token or any(character.isspace() for character in token):
            raise ValueError("bot_token must be a non-empty token without whitespace")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        normalized_base = api_base.rstrip("/")
        if normalized_base == TELEGRAM_API_BASE:
            self._api_base = f"{TELEGRAM_API_BASE}/bot{token}"
            self._extra_headers: dict[str, str] = {}
        else:
            secret = (relay_secret or "").strip()
            if not secret or any(character.isspace() for character in secret):
                raise ValueError("relay_secret must be set for a Telegram API relay")
            self._api_base = normalized_base
            self._extra_headers = {"X-Pharmacheck-Relay-Secret": secret}
        self._timeout = timeout
        self._transport = transport or _default_transport

    def send_message(
        self,
        *,
        chat_id: int,
        text: str,
        message_thread_id: int | None = None,
    ) -> None:
        if not text or len(text) > TELEGRAM_TEXT_LIMIT:
            raise ValueError("Telegram message must contain 1-4096 characters")
        payload: dict[str, object] = {
            "chat_id": chat_id,
            "text": text,
            "link_preview_options": {"is_disabled": True},
        }
        if message_thread_id is not None:
            payload["message_thread_id"] = message_thread_id
        self._call("sendMessage", payload)

    def set_webhook(self, payload: Mapping[str, object]) -> None:
        """Register the HTTPS webhook without exposing the bot token."""

        self._call("setWebhook", payload)

    def set_my_commands(self, payload: Mapping[str, object]) -> None:
        """Register the private-chat command menu."""

        self._call("setMyCommands", payload)

    def get_me(self) -> Mapping[str, Any]:
        """Return Telegram metadata for the configured bot."""

        return self._call("getMe", {})

    def get_webhook_info(self) -> Mapping[str, Any]:
        """Return Telegram webhook metadata for safe deployment checks."""

        return self._call("getWebhookInfo", {})

    def _call(self, method: str, payload: Mapping[str, object]) -> Mapping[str, Any]:
        request = Request(
            f"{self._api_base}/{method}",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "User-Agent": TELEGRAM_USER_AGENT,
                **self._extra_headers,
            },
            method="POST",
        )
        try:
            response = self._transport(request, self._timeout)
        except HTTPError as exc:
            retryable = exc.code == 429 or exc.code >= 500
            raise TelegramAPIError(
                "Telegram Bot API отклонил отправку сообщения.",
                code="telegram_http_error",
                retryable=retryable,
            ) from None
        except (TimeoutError, URLError, UnicodeError, ValueError):
            raise TelegramAPIError(
                "Не удалось связаться с Telegram Bot API.",
                code="telegram_network_error",
                retryable=True,
            ) from None
        if response.get("ok") is not True:
            raise TelegramAPIError(
                "Telegram Bot API не подтвердил отправку сообщения.",
                code="telegram_rejected",
                retryable=False,
            )
        return response


MAX_DISPLAYED_SOURCES = 3
MAX_INGREDIENTS_SHOWN = 5

# (min, max) character caps for the fields that make up the mandatory tier.
# The *first source's URL is never in this list* — it is the one thing that
# is never clipped or dropped. Everything else shrinks dynamically, in a
# single coordinated pass, to guarantee the URL always has room.
HEADLINE_CAP = (30, 200)
DRUG_NAME_CAP = (30, 120)
INGREDIENT_CAP = (20, 80)
ACTION_CAP = (30, 700)
SOURCE_TITLE_CAP = (20, 150)

# Scale factors tried from most to least generous; the largest one whose
# render still fits is kept. 0.0 always maps every field to its floor.
_SHRINK_SCALES = (1.0, 0.8, 0.6, 0.45, 0.3, 0.18, 0.1, 0.0)


def _scaled_cap(bounds: tuple[int, int], scale: float) -> int:
    low, high = bounds
    return low + round((high - low) * scale)


def _display_ingredient(name: str, ingredient_names: Mapping[str, str] | None) -> str:
    """Show the Russian ingredient name when known; the English one otherwise."""
    russian = (ingredient_names or {}).get(name.casefold())
    if not russian:
        return name
    return russian[0].lower() + russian[1:] if russian[1:2].islower() else russian


def format_analysis_result(
    result: AnalysisResult,
    *,
    ingredient_names: Mapping[str, str] | None = None,
) -> str:
    """Format one validated analysis as a single Telegram-safe plain-text message.

    Every field that reaches this function is schema-validated but
    length-unbounded (`NonEmptyText` has no max length). Two guarantees
    must hold at once: `1 <= len(text) <= TELEGRAM_TEXT_LIMIT`, and — when
    `result.sources` is non-empty — the first source's complete, un-clipped
    URL appears exactly once. The URL itself is never touched. To make room
    for it, the headline, drug names, ingredients, and safe action text are
    shrunk dynamically (all four scaled down together, not clipped to a
    fixed cap) until the mandatory tier fits; only after that does an
    explanatory section (summary, mechanism, limitations, conflicts,
    missing information, or a 2nd/3rd source) get added, each dropped whole
    — never partially cut — in priority order, only while it still fits.
    Sources show only their title and complete URL: the internal `supports`
    explanation repeats the result body and is intentionally omitted from
    Telegram. A source count is never shown without at least one source line
    above it. `ingredient_names` maps English ingredient names to Russian
    display names; it changes only how ingredients are shown.
    """

    icon = _result_icon(result)
    shown_sources = result.sources[:MAX_DISPLAYED_SOURCES]

    def build_mandatory(scale: float) -> tuple[str, str | None, str, str | None]:
        header = f"{icon} {_clip(result.headline, _scaled_cap(HEADLINE_CAP, scale))}"

        drugs_block = None
        if result.identified_drugs:
            drug_name_cap = _scaled_cap(DRUG_NAME_CAP, scale)
            ingredient_cap = _scaled_cap(INGREDIENT_CAP, scale)
            drugs = []
            for drug in result.identified_drugs:
                shown_ingredients = drug.active_ingredients[:MAX_INGREDIENTS_SHOWN]
                ingredient_text = ", ".join(
                    _clip(_display_ingredient(item, ingredient_names), ingredient_cap)
                    for item in shown_ingredients
                )
                omitted = len(drug.active_ingredients) - len(shown_ingredients)
                if omitted:
                    ingredient_text += f" и ещё {omitted}"
                drugs.append(f"{_clip(drug.trade_name, drug_name_cap)} — {ingredient_text}")
            drugs_block = "Препараты:\n" + "\n".join(f"• {item}" for item in drugs)

        action_block = "Что делать:\n" + _clip(
            result.recommended_user_action, _scaled_cap(ACTION_CAP, scale)
        )

        first_source_line = None
        if shown_sources:
            title = _clip(shown_sources[0].title, _scaled_cap(SOURCE_TITLE_CAP, scale))
            first_source_line = f"1. {title}\n{shown_sources[0].url}"

        return header, drugs_block, action_block, first_source_line

    def render_mandatory(scale: float) -> str:
        header, drugs_block, action_block, first_source_line = build_mandatory(scale)
        sections = [header]
        if drugs_block:
            sections.append(drugs_block)
        sections.append(action_block)
        if first_source_line:
            sections.append("Источники:\n" + first_source_line)
        return "\n\n".join(sections)

    chosen_scale = _SHRINK_SCALES[-1]
    for candidate_scale in _SHRINK_SCALES:
        if len(render_mandatory(candidate_scale)) <= TELEGRAM_TEXT_LIMIT:
            chosen_scale = candidate_scale
            break
    header, drugs_block, action_block, first_source_line = build_mandatory(chosen_scale)

    summary_block = _clip(result.summary, 1100)
    mechanism_block = (
        "Возможный эффект:\n" + _clip(result.mechanism_or_effect, 700)
        if result.mechanism_or_effect
        else None
    )
    missing_block = (
        _list_section("Нужно уточнить", result.missing_information, 4)
        if result.missing_information
        else None
    )
    conflicts_block = (
        _list_section("Расхождения источников", result.conflicts, 3)
        if result.conflicts
        else None
    )
    limitations_block = (
        _list_section("Ограничения", result.limitations, 4) if result.limitations else None
    )

    source_lines = [first_source_line] if first_source_line else []
    if len(shown_sources) >= 2:
        source_lines.append(f"2. {_clip(shown_sources[1].title, 150)}\n{shown_sources[1].url}")
    if len(shown_sources) >= 3:
        source_lines.append(f"3. {_clip(shown_sources[2].title, 150)}\n{shown_sources[2].url}")
    def render(
        *,
        summary: bool,
        mechanism: bool,
        missing: bool,
        conflicts: bool,
        limitations: bool,
        sources_count: int,
    ) -> str:
        sections = [header]
        if drugs_block:
            sections.append(drugs_block)
        if summary:
            sections.append(summary_block)
        if mechanism and mechanism_block:
            sections.append(mechanism_block)
        sections.append(action_block)
        if missing and missing_block:
            sections.append(missing_block)
        if conflicts and conflicts_block:
            sections.append(conflicts_block)
        if limitations and limitations_block:
            sections.append(limitations_block)
        if sources_count > 0:
            lines = source_lines[:sources_count]
            remaining = len(result.sources) - sources_count
            if remaining:
                lines.append(f"Ещё источников: {remaining}.")
            sections.append("Источники:\n" + "\n".join(lines))
        return "\n\n".join(sections)

    state: dict[str, Any] = {
        "summary": False,
        "mechanism": False,
        "missing": False,
        "conflicts": False,
        "limitations": False,
        "sources_count": 1 if first_source_line else 0,
    }

    def fits(**overrides: Any) -> bool:
        return len(render(**{**state, **overrides})) <= TELEGRAM_TEXT_LIMIT

    if summary_block and fits(summary=True):
        state["summary"] = True
    if mechanism_block and fits(mechanism=True):
        state["mechanism"] = True
    if limitations_block and fits(limitations=True):
        state["limitations"] = True
    if len(source_lines) >= 2 and fits(sources_count=2):
        state["sources_count"] = 2
    if conflicts_block and fits(conflicts=True):
        state["conflicts"] = True
    if missing_block and fits(missing=True):
        state["missing"] = True
    if (
        len(source_lines) >= 3
        and state["sources_count"] == 2
        and fits(sources_count=3)
    ):
        state["sources_count"] = 3

    text = render(**state)
    # Every optional addition above was accepted only after `fits()`
    # confirmed it, so this should already hold; kept as a hard backstop so
    # a future maintenance bug can never violate the Telegram length limit.
    # This backstop is the only place that could ever cut a URL, and only
    # if the shrink search above still left the mandatory tier over the
    # limit — which does not happen for any realistic source URL.
    if len(text) > TELEGRAM_TEXT_LIMIT:
        text = text[:TELEGRAM_TEXT_LIMIT]
    return text or "…"


def _default_transport(request: Request, timeout: float) -> Mapping[str, Any]:
    with urlopen(request, timeout=timeout) as response:
        decoded = json.loads(response.read().decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("Telegram response must be an object")
    return decoded


def _result_icon(result: AnalysisResult) -> str:
    if result.status is AnalysisStatus.URGENT:
        return "🚑"
    return {
        InteractionLevel.GREEN: "🟢",
        InteractionLevel.YELLOW: "🟡",
        InteractionLevel.ORANGE: "🟠",
        InteractionLevel.RED: "🔴",
        InteractionLevel.WHITE: "⚪",
    }[result.interaction_level]


def _list_section(title: str, values: list[str], limit: int) -> str:
    shown = values[:limit]
    lines = [f"• {_clip(value, 350)}" for value in shown]
    if len(values) > len(shown):
        lines.append(f"• Ещё пунктов: {len(values) - len(shown)}.")
    return f"{title}:\n" + "\n".join(lines)


def _clip(value: str, limit: int) -> str:
    value = " ".join(value.split())
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"
