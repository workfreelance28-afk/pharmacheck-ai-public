"""Yandex Cloud Functions runtime composition for PharmaCheck AI."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

from pharmacheck.analytics import DemoAnalyticsRecorder, validate_analytics_key
from pharmacheck.application import PharmaCheckService
from pharmacheck.drug_recognition import DrugCatalog, DrugCatalogEntry
from pharmacheck.sources.dailymed import DailyMedLabelClient
from pharmacheck.sources.openfda import (
    OPENFDA_LABEL_URL,
    OPENFDA_RELAY_LABEL_URL,
    OpenFDALabelClient,
)
from pharmacheck.sources.rxclass import RxClassClient
from pharmacheck.sources.rxnorm import RxNormClient
from pharmacheck.telegram import TELEGRAM_API_BASE, TelegramAPIError, TelegramBotClient
from pharmacheck.telegram_webhook import (
    NONCOMMERCIAL_WELCOME_TEXT,
    WELCOME_TEXT,
    RecentUpdateStore,
    TelegramWebhook,
    bot_commands_payload,
    webhook_registration_payload,
)
from pharmacheck.yandexgpt import YandexGPTClient


class RuntimeConfigurationError(RuntimeError):
    """Safe startup error caused by absent or unsafe production configuration."""


class InvocationBudgetError(RuntimeError):
    """Raised before an expensive operation when the function deadline is too close."""


ADMIN_OPERATION = "register_telegram"
ADMIN_STATUS_OPERATION = "telegram_status"
GATEWAY_URL_RE = re.compile(
    r"https://(?:(?:[a-z0-9-]+\.)+apigw\.yandexcloud\.net|"
    r"pharmacheck-(?:demo-)?telegram-relay\.your-account\.workers\.dev)"
    r"/telegram"
)
TELEGRAM_RELAY_API_BASE = (
    "https://pharmacheck-telegram-relay.your-account.workers.dev/telegram-api"
)
TELEGRAM_DEMO_RELAY_API_BASE = (
    "https://pharmacheck-demo-telegram-relay.your-account.workers.dev/telegram-api"
)
USAGE_MODE_NONCOMMERCIAL_MVP = "noncommercial_mvp"
USAGE_MODE_OFFICIAL_LABELS_DEMO = "official_labels_demo"
APPROVED_TELEGRAM_API_BASES = {
    USAGE_MODE_NONCOMMERCIAL_MVP: frozenset({TELEGRAM_API_BASE, TELEGRAM_RELAY_API_BASE}),
    USAGE_MODE_OFFICIAL_LABELS_DEMO: frozenset({TELEGRAM_DEMO_RELAY_API_BASE}),
}
SUPPORTED_USAGE_MODES = frozenset(
    {USAGE_MODE_NONCOMMERCIAL_MVP, USAGE_MODE_OFFICIAL_LABELS_DEMO}
)
DEPLOYMENT_MANIFEST_NAME = "deployment_manifest.json"
INGREDIENT_NAMES_FILENAME = "ingredient_mappings.json"
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    telegram_bot_token: str
    telegram_webhook_secret: str
    telegram_api_base: str
    folder_id: str
    model_uri: str
    usage_mode: str
    catalog_path: Path
    catalog_manifest_path: Path
    ddinter_path: Path | None
    prompt_path: Path
    analytics_hmac_key: bytes | None
    openfda_label_url: str = OPENFDA_LABEL_URL
    openfda_relay_secret: str | None = None

    @classmethod
    def from_environment(cls, *, root: Path | None = None) -> RuntimeConfig:
        base = root or Path(__file__).resolve().parents[1]
        folder_id = _required_env("YC_FOLDER_ID")
        model_uri = _required_env("YANDEXGPT_MODEL_URI")
        expected_uri = f"gpt://{folder_id}/yandexgpt-5.1"
        if model_uri != expected_uri:
            raise RuntimeConfigurationError("YANDEXGPT_MODEL_URI must use explicit yandexgpt-5.1")
        usage_mode = os.getenv("PHARMACHECK_USAGE_MODE")
        if usage_mode not in SUPPORTED_USAGE_MODES:
            raise RuntimeConfigurationError(
                "PHARMACHECK_USAGE_MODE must be one of: "
                + ", ".join(sorted(SUPPORTED_USAGE_MODES))
            )
        _validate_deployment_manifest(base, usage_mode)
        telegram_api_base = os.getenv("TELEGRAM_API_BASE", TELEGRAM_API_BASE).rstrip("/")
        if telegram_api_base not in APPROVED_TELEGRAM_API_BASES[usage_mode]:
            raise RuntimeConfigurationError(
                f"TELEGRAM_API_BASE is not an approved endpoint for {usage_mode}"
            )
        telegram_bot_token = _required_env("TELEGRAM_BOT_TOKEN")
        telegram_webhook_secret = _required_env("TELEGRAM_WEBHOOK_SECRET")
        analytics_hmac_key = (
            _required_analytics_key(
                other_secrets=(telegram_bot_token, telegram_webhook_secret)
            )
            if usage_mode == USAGE_MODE_OFFICIAL_LABELS_DEMO
            else None
        )
        openfda_label_url = os.getenv("OPENFDA_LABEL_URL", OPENFDA_LABEL_URL)
        openfda_relay_secret = os.getenv("OPENFDA_RELAY_SECRET")
        if openfda_label_url not in {OPENFDA_LABEL_URL, OPENFDA_RELAY_LABEL_URL}:
            raise RuntimeConfigurationError("OPENFDA_LABEL_URL is not an approved endpoint")
        if openfda_label_url == OPENFDA_RELAY_LABEL_URL:
            if not openfda_relay_secret:
                raise RuntimeConfigurationError(
                    "OPENFDA_RELAY_SECRET is required for the relay endpoint"
                )
        elif openfda_relay_secret is not None:
            raise RuntimeConfigurationError(
                "OPENFDA_RELAY_SECRET is only valid for the relay endpoint"
            )
        return cls(
            telegram_bot_token=telegram_bot_token,
            telegram_webhook_secret=telegram_webhook_secret,
            telegram_api_base=telegram_api_base,
            folder_id=folder_id,
            model_uri=model_uri,
            usage_mode=usage_mode,
            catalog_path=Path(
                os.getenv("DRUG_CATALOG_PATH", str(base / "data" / "drug_catalog.json"))
            ),
            catalog_manifest_path=Path(
                os.getenv(
                    "DRUG_CATALOG_MANIFEST_PATH",
                    str(base / "data" / "drug_catalog.manifest.json"),
                )
            ),
            ddinter_path=(
                Path(os.getenv("DDINTER_DATA_DIR", str(base / "data" / "ddinter")))
                if usage_mode == USAGE_MODE_NONCOMMERCIAL_MVP
                else None
            ),
            prompt_path=Path(
                os.getenv(
                    "SYSTEM_PROMPT_PATH",
                    str(base / "prompts" / "system_prompt_v1.txt"),
                )
            ),
            analytics_hmac_key=analytics_hmac_key,
            openfda_label_url=openfda_label_url,
            openfda_relay_secret=openfda_relay_secret,
        )


@dataclass(frozen=True, slots=True)
class RuntimeResources:
    config: RuntimeConfig
    catalog: DrugCatalog
    sources: Mapping[str, object]
    system_prompt: str
    update_store: RecentUpdateStore
    ingredient_names: Mapping[str, str] = field(default_factory=dict)


@lru_cache(maxsize=1)
def load_runtime_resources() -> RuntimeResources:
    config = RuntimeConfig.from_environment()
    catalog = _load_production_catalog(config.catalog_path, config.catalog_manifest_path)
    if not config.prompt_path.is_file():
        raise RuntimeConfigurationError("system prompt file is missing")
    system_prompt = config.prompt_path.read_text(encoding="utf-8").strip()
    if not system_prompt:
        raise RuntimeConfigurationError("system prompt is empty")
    rxnorm = RxNormClient(timeout=8.0, cache_ttl_seconds=86_400)
    rxclass = RxClassClient(timeout=8.0, cache_ttl_seconds=86_400)
    sources = _build_sources(config, rxnorm=rxnorm, rxclass=rxclass)
    ingredient_names = (
        load_ingredient_names(config.catalog_path.parent / INGREDIENT_NAMES_FILENAME)
        if config.usage_mode == USAGE_MODE_NONCOMMERCIAL_MVP
        else {}
    )
    return RuntimeResources(
        config, catalog, sources, system_prompt, RecentUpdateStore(), ingredient_names
    )


def load_ingredient_names(path: Path) -> dict[str, str]:
    """English → Russian ingredient display names; display only, never fatal."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        names = {
            item["source_name"].casefold(): " ".join(item["russian_name"].split())
            for item in raw
            if isinstance(item.get("source_name"), str)
            and isinstance(item.get("russian_name"), str)
            and item["russian_name"].strip()
        }
    except (OSError, UnicodeError, ValueError, TypeError, AttributeError):
        logger.warning("ingredient_names_unavailable")
        return {}
    return names


def _build_sources(
    config: RuntimeConfig,
    *,
    rxnorm: RxNormClient,
    rxclass: RxClassClient,
) -> dict[str, object]:
    """Compose the interaction-search sources for the active usage mode.

    `official_labels_demo` must never import, construct, or call
    `DDInterSource` (D-011): the import is deliberately local to the
    `noncommercial_mvp` branch so the demo deployment archive can omit
    `pharmacheck/sources/ddinter.py` entirely without breaking on cold start.
    """
    dailymed = DailyMedLabelClient(
        rxnorm=rxnorm,
        rxclass=rxclass,
        timeout=12.0,
        max_labels_per_ingredient=2,
    )
    if config.usage_mode == USAGE_MODE_OFFICIAL_LABELS_DEMO:
        return {
            "dailymed.label": dailymed,
            "openfda.label": OpenFDALabelClient(
                label_url=config.openfda_label_url,
                relay_secret=config.openfda_relay_secret,
                timeout=12.0,
                max_results=2,
            ),
        }
    from pharmacheck.sources.ddinter import DDInterSource

    if config.ddinter_path is None:
        raise RuntimeConfigurationError("noncommercial_mvp mode requires a DDInter data path")
    return {
        "dailymed.label": dailymed,
        "ddinter": DDInterSource(config.ddinter_path, commercial_use=False),
    }


def build_webhook(context: Any) -> TelegramWebhook:
    resources = load_runtime_resources()
    access_token = _context_access_token(context)
    analyzer = YandexGPTClient(
        model_uri=resources.config.model_uri,
        token_provider=lambda: access_token,
        system_prompt=resources.system_prompt,
        folder_id=resources.config.folder_id,
        timeout=60.0,
    )
    service = PharmaCheckService(
        catalog=resources.catalog,
        sources=resources.sources,
        analyzer=analyzer,
        budget_check=lambda stage: _check_budget(context, stage),
    )
    bot = _telegram_client(resources, timeout=10.0)
    analytics = DemoAnalyticsRecorder(
        hmac_key=resources.config.analytics_hmac_key,
        usage_mode=resources.config.usage_mode,
    )
    return TelegramWebhook(
        webhook_secret=resources.config.telegram_webhook_secret,
        bot_client=bot,
        check_service=service,
        update_store=resources.update_store,
        analytics=analytics,
        welcome_text=welcome_text_for(resources.config.usage_mode),
        ingredient_names=resources.ingredient_names,
    )


def welcome_text_for(usage_mode: str) -> str:
    """The private MVP bot is not the demo; the demo keeps its published text."""
    if usage_mode == USAGE_MODE_NONCOMMERCIAL_MVP:
        return NONCOMMERCIAL_WELCOME_TEXT
    return WELCOME_TEXT


def handle_cloud_function(event: object, context: Any) -> dict[str, Any]:
    if not isinstance(event, Mapping):
        return _service_response(400)
    try:
        if event.get("pharmacheck_admin_operation") == ADMIN_OPERATION:
            return _register_telegram(event)
        if event.get("pharmacheck_admin_operation") == ADMIN_STATUS_OPERATION:
            return _telegram_status()
        webhook = build_webhook(context)
        request_id = str(getattr(context, "request_id", "cloud-function"))
        return webhook.handle(event, request_id=request_id)
    except (RuntimeConfigurationError, InvocationBudgetError):
        return _service_response(503)
    except Exception:
        return _service_response(500)


def _register_telegram(event: Mapping[str, Any]) -> dict[str, Any]:
    """Register Telegram from a direct IAM-authorized function invocation."""

    gateway_url = event.get("gateway_url")
    gateway_ip = event.get("gateway_ip")
    if not isinstance(gateway_url, str) or GATEWAY_URL_RE.fullmatch(gateway_url) is None:
        return _service_response(400)
    if gateway_ip is not None and not isinstance(gateway_ip, str):
        return _service_response(400)
    resources = load_runtime_resources()
    bot = _telegram_client(resources, timeout=30.0)
    try:
        bot.set_webhook(
            webhook_registration_payload(
                url=gateway_url,
                secret_token=resources.config.telegram_webhook_secret,
                ip_address=gateway_ip,
            )
        )
    except TelegramAPIError as exc:
        return _service_response(502, error=exc.code, stage="set_webhook")
    try:
        bot.set_my_commands(bot_commands_payload())
    except TelegramAPIError as exc:
        return _service_response(502, error=exc.code, stage="set_commands")
    return _service_response(200, ok=True)


def _telegram_status() -> dict[str, Any]:
    """Return non-secret Telegram metadata for deployment diagnostics."""

    resources = load_runtime_resources()
    bot = _telegram_client(resources, timeout=30.0)
    try:
        me = bot.get_me().get("result")
        webhook = bot.get_webhook_info().get("result")
    except TelegramAPIError as exc:
        return _service_response(502, error=exc.code, stage="telegram_status")
    if not isinstance(me, Mapping) or not isinstance(webhook, Mapping):
        return _service_response(502, error="telegram_rejected", stage="telegram_status")
    details = {
        "bot_username": me.get("username"),
        "webhook_url": webhook.get("url"),
        "pending_update_count": webhook.get("pending_update_count"),
        "last_error_message": webhook.get("last_error_message"),
        "ip_address": webhook.get("ip_address"),
    }
    return _service_response(200, ok=True, details=details)


def _telegram_client(resources: RuntimeResources, *, timeout: float) -> TelegramBotClient:
    relay_secret = None
    if resources.config.telegram_api_base != TELEGRAM_API_BASE:
        relay_secret = resources.config.telegram_webhook_secret
    return TelegramBotClient(
        bot_token=resources.config.telegram_bot_token,
        timeout=timeout,
        api_base=resources.config.telegram_api_base,
        relay_secret=relay_secret,
    )


def _load_production_catalog(path: Path, manifest_path: Path) -> DrugCatalog:
    if not path.is_file():
        raise RuntimeConfigurationError("production drug catalog is missing")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError
        entries = [DrugCatalogEntry.model_validate(item) for item in raw]
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeConfigurationError("production drug catalog is invalid") from exc
    if not entries:
        raise RuntimeConfigurationError("production drug catalog is empty")
    if any(
        entry.source_id.casefold().startswith("synthetic")
        or entry.product_id.casefold().startswith("test.")
        for entry in entries
    ):
        raise RuntimeConfigurationError("synthetic drug catalog is forbidden in production")
    _validate_catalog_manifest(path, manifest_path, entries)
    return DrugCatalog(entries)


def _validate_catalog_manifest(
    catalog_path: Path,
    manifest_path: Path,
    entries: list[DrugCatalogEntry],
) -> None:
    if not manifest_path.is_file():
        raise RuntimeConfigurationError("production drug catalog manifest is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        export_date = date.fromisoformat(manifest["export_date"])
    except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeConfigurationError("production drug catalog manifest is invalid") from exc
    if manifest.get("schema_version") != 1:
        raise RuntimeConfigurationError("production drug catalog manifest schema is unsupported")
    if manifest.get("source_page_url") != "https://grls.rosminzdrav.ru/GRLS.aspx":
        raise RuntimeConfigurationError("production drug catalog source is not GRLS")
    if export_date > datetime.now(UTC).date():
        raise RuntimeConfigurationError("production drug catalog date is in the future")
    if export_date < datetime.now(UTC).date() - timedelta(days=45):
        raise RuntimeConfigurationError("production drug catalog is stale")
    if manifest.get("catalog_entries") != len(entries):
        raise RuntimeConfigurationError("production drug catalog entry count changed")
    digest = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    if manifest.get("catalog_sha256") != digest:
        raise RuntimeConfigurationError("production drug catalog checksum changed")
    source_id = manifest.get("source_id")
    if not isinstance(source_id, str) or any(entry.source_id != source_id for entry in entries):
        raise RuntimeConfigurationError("production drug catalog provenance is inconsistent")


def _context_access_token(context: Any) -> str:
    token = getattr(context, "token", None)
    if isinstance(token, Mapping):
        value = token.get("access_token")
    else:
        value = getattr(token, "access_token", None)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeConfigurationError("function IAM token is unavailable")
    return value


def _check_budget(context: Any, stage: str) -> None:
    getter = getattr(context, "get_remaining_time_in_millis", None)
    if not callable(getter):
        raise InvocationBudgetError("function deadline is unavailable")
    remaining = int(getter())
    minimum = 115_000 if stage == "medical_sources" else 65_000
    if remaining < minimum:
        raise InvocationBudgetError("not enough invocation time for external operation")


def _validate_deployment_manifest(base: Path, usage_mode: str) -> None:
    """Fail closed if the deployed archive was built for a different mode.

    `scripts/build_function_bundle.py` writes `deployment_manifest.json`
    recording which `usage_mode` an archive was built with (D-011). This
    catches a misconfigured deployment (e.g. a `noncommercial_mvp` archive
    started with `PHARMACHECK_USAGE_MODE=official_labels_demo`, or vice
    versa) before it processes any request, instead of relying only on
    `_build_sources` failing or silently doing the wrong thing.
    """
    manifest_path = base / DEPLOYMENT_MANIFEST_NAME
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeConfigurationError(
            "deployment manifest is missing or invalid"
        ) from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise RuntimeConfigurationError("deployment manifest schema is unsupported")
    if raw.get("usage_mode") != usage_mode:
        raise RuntimeConfigurationError(
            "deployment manifest usage_mode does not match PHARMACHECK_USAGE_MODE"
        )


def _required_analytics_key(*, other_secrets: tuple[str, ...]) -> bytes:
    value = _required_env("PHARMACHECK_ANALYTICS_HMAC_KEY")
    try:
        return validate_analytics_key(value, other_secrets=other_secrets)
    except ValueError as exc:
        raise RuntimeConfigurationError(str(exc)) from exc


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeConfigurationError(f"required environment variable is missing: {name}")
    return value.strip()


def _service_response(
    status: int,
    *,
    ok: bool = False,
    error: str | None = None,
    stage: str | None = None,
    details: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    payload: dict[str, object] = {"ok": ok}
    if error is not None:
        payload["error"] = error
    if stage is not None:
        payload["stage"] = stage
    if details is not None:
        payload.update(details)
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json; charset=utf-8"},
        "body": json.dumps(payload, ensure_ascii=False),
    }
