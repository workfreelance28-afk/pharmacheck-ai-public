from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from pharmacheck.cloud_runtime import (
    DEPLOYMENT_MANIFEST_NAME,
    GATEWAY_URL_RE,
    TELEGRAM_DEMO_RELAY_API_BASE,
    TELEGRAM_RELAY_API_BASE,
    USAGE_MODE_NONCOMMERCIAL_MVP,
    USAGE_MODE_OFFICIAL_LABELS_DEMO,
    InvocationBudgetError,
    RuntimeConfig,
    RuntimeConfigurationError,
    _build_sources,
    _check_budget,
    _context_access_token,
    _load_production_catalog,
    _register_telegram,
    _validate_deployment_manifest,
    handle_cloud_function,
)
from pharmacheck.sources.ddinter import DDInterSource
from pharmacheck.sources.openfda import (
    OPENFDA_LABEL_URL,
    OPENFDA_RELAY_LABEL_URL,
    OpenFDALabelClient,
)
from pharmacheck.sources.rxclass import RxClassClient
from pharmacheck.sources.rxnorm import RxNormClient
from pharmacheck.telegram import TELEGRAM_API_BASE


class Context:
    request_id = "request-id"

    def __init__(self, remaining=180_000, token=None):
        self.remaining = remaining
        self.token = token

    def get_remaining_time_in_millis(self):
        return self.remaining


def _catalog(path: Path, *, product_id="drug.alpha", source_id="grls.export.test"):
    path.write_text(
        json.dumps(
            [
                {
                    "product_id": product_id,
                    "trade_name": "Альфа",
                    "aliases": [],
                    "dosage_form": "таблетки",
                    "strength": "10мг",
                    "active_ingredients": ["alpha"],
                    "source_id": source_id,
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest = path.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_id": source_id,
                "source_page_url": "https://grls.rosminzdrav.ru/GRLS.aspx",
                "export_date": datetime.now(UTC).date().isoformat(),
                "catalog_entries": 1,
                "catalog_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _write_deployment_manifest(root: Path, usage_mode: str) -> None:
    (root / DEPLOYMENT_MANIFEST_NAME).write_text(
        json.dumps({"schema_version": 1, "usage_mode": usage_mode}),
        encoding="utf-8",
    )


def test_runtime_config_requires_explicit_model_and_noncommercial_mode(monkeypatch, tmp_path):
    values = {
        "YC_FOLDER_ID": "folder-id",
        "YANDEXGPT_MODEL_URI": "gpt://folder-id/yandexgpt-5.1",
        "PHARMACHECK_USAGE_MODE": "noncommercial_mvp",
        "TELEGRAM_BOT_TOKEN": "123:token",
        "TELEGRAM_WEBHOOK_SECRET": "secret",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    _write_deployment_manifest(tmp_path, "noncommercial_mvp")

    config = RuntimeConfig.from_environment(root=tmp_path)
    assert config.model_uri == values["YANDEXGPT_MODEL_URI"]
    assert config.catalog_path == tmp_path / "data" / "drug_catalog.json"

    monkeypatch.setenv(
        "TELEGRAM_API_BASE",
        "https://pharmacheck-telegram-relay.your-account.workers.dev/telegram-api",
    )
    assert RuntimeConfig.from_environment(root=tmp_path).telegram_api_base.endswith(
        "/telegram-api"
    )

    monkeypatch.setenv("TELEGRAM_API_BASE", TELEGRAM_DEMO_RELAY_API_BASE)
    with pytest.raises(RuntimeConfigurationError, match="for noncommercial_mvp"):
        RuntimeConfig.from_environment(root=tmp_path)

    monkeypatch.setenv("TELEGRAM_API_BASE", "https://untrusted.example.test")
    with pytest.raises(RuntimeConfigurationError):
        RuntimeConfig.from_environment(root=tmp_path)

    monkeypatch.setenv("YANDEXGPT_MODEL_URI", "gpt://folder-id/latest")
    with pytest.raises(RuntimeConfigurationError):
        RuntimeConfig.from_environment(root=tmp_path)


def test_runtime_config_rejects_unsupported_usage_mode(monkeypatch, tmp_path):
    values = {
        "YC_FOLDER_ID": "folder-id",
        "YANDEXGPT_MODEL_URI": "gpt://folder-id/yandexgpt-5.1",
        "PHARMACHECK_USAGE_MODE": "commercial_unrestricted",
        "TELEGRAM_BOT_TOKEN": "123:token",
        "TELEGRAM_WEBHOOK_SECRET": "secret",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(RuntimeConfigurationError):
        RuntimeConfig.from_environment(root=tmp_path)


@pytest.mark.parametrize(
    ("mode", "endpoint"),
    [
        (USAGE_MODE_NONCOMMERCIAL_MVP, None),
        (USAGE_MODE_NONCOMMERCIAL_MVP, TELEGRAM_API_BASE),
        (USAGE_MODE_NONCOMMERCIAL_MVP, TELEGRAM_RELAY_API_BASE),
        (USAGE_MODE_NONCOMMERCIAL_MVP, TELEGRAM_RELAY_API_BASE + "/"),
        (USAGE_MODE_OFFICIAL_LABELS_DEMO, TELEGRAM_DEMO_RELAY_API_BASE),
        (USAGE_MODE_OFFICIAL_LABELS_DEMO, TELEGRAM_DEMO_RELAY_API_BASE + "/"),
    ],
)
def test_runtime_config_accepts_telegram_endpoint_for_its_mode(
    monkeypatch, tmp_path, mode, endpoint
):
    _telegram_mode_environment(monkeypatch, tmp_path, mode, endpoint)

    config = RuntimeConfig.from_environment(root=tmp_path)

    assert config.telegram_api_base == (endpoint or TELEGRAM_API_BASE).rstrip("/")
    assert config.usage_mode == mode
    if mode == USAGE_MODE_OFFICIAL_LABELS_DEMO:
        assert config.ddinter_path is None
        assert config.analytics_hmac_key == b"x" * 32
    else:
        assert config.ddinter_path == tmp_path / "data" / "ddinter"
        assert config.analytics_hmac_key is None


@pytest.mark.parametrize(
    ("mode", "endpoint"),
    [
        (USAGE_MODE_NONCOMMERCIAL_MVP, TELEGRAM_DEMO_RELAY_API_BASE),
        (USAGE_MODE_OFFICIAL_LABELS_DEMO, TELEGRAM_RELAY_API_BASE),
        (USAGE_MODE_OFFICIAL_LABELS_DEMO, TELEGRAM_API_BASE),
        (USAGE_MODE_OFFICIAL_LABELS_DEMO, None),
        (USAGE_MODE_NONCOMMERCIAL_MVP, "https://untrusted.example.test"),
        (USAGE_MODE_OFFICIAL_LABELS_DEMO, "https://untrusted.example.test"),
    ],
)
def test_runtime_config_rejects_telegram_endpoint_mismatch_at_startup(
    monkeypatch, tmp_path, mode, endpoint
):
    _telegram_mode_environment(monkeypatch, tmp_path, mode, endpoint)

    with pytest.raises(RuntimeConfigurationError, match=f"approved endpoint for {mode}"):
        RuntimeConfig.from_environment(root=tmp_path)


def _telegram_mode_environment(monkeypatch, root, mode, endpoint):
    values = {
        "YC_FOLDER_ID": "folder-id",
        "YANDEXGPT_MODEL_URI": "gpt://folder-id/yandexgpt-5.1",
        "PHARMACHECK_USAGE_MODE": mode,
        "TELEGRAM_BOT_TOKEN": "123:token",
        "TELEGRAM_WEBHOOK_SECRET": "secret",
        "PHARMACHECK_ANALYTICS_HMAC_KEY": "x" * 32,
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    if endpoint is None:
        monkeypatch.delenv("TELEGRAM_API_BASE", raising=False)
    else:
        monkeypatch.setenv("TELEGRAM_API_BASE", endpoint)
    _write_deployment_manifest(root, mode)


def test_runtime_config_official_labels_demo_never_gets_a_ddinter_path(
    monkeypatch, tmp_path
):
    values = {
        "YC_FOLDER_ID": "folder-id",
        "YANDEXGPT_MODEL_URI": "gpt://folder-id/yandexgpt-5.1",
        "PHARMACHECK_USAGE_MODE": USAGE_MODE_OFFICIAL_LABELS_DEMO,
        "TELEGRAM_API_BASE": TELEGRAM_DEMO_RELAY_API_BASE,
        "TELEGRAM_BOT_TOKEN": "123:token",
        "TELEGRAM_WEBHOOK_SECRET": "secret",
        "PHARMACHECK_ANALYTICS_HMAC_KEY": "x" * 32,
        # Deliberately set to prove the demo config ignores it regardless.
        "DDINTER_DATA_DIR": str(tmp_path / "data" / "ddinter"),
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    _write_deployment_manifest(tmp_path, USAGE_MODE_OFFICIAL_LABELS_DEMO)

    config = RuntimeConfig.from_environment(root=tmp_path)

    assert config.usage_mode == USAGE_MODE_OFFICIAL_LABELS_DEMO
    assert config.ddinter_path is None
    assert config.analytics_hmac_key == b"x" * 32
    assert config.openfda_label_url == OPENFDA_LABEL_URL
    assert config.openfda_relay_secret is None


def test_runtime_config_accepts_only_protected_approved_openfda_relay(
    monkeypatch, tmp_path
):
    values = {
        "YC_FOLDER_ID": "folder-id",
        "YANDEXGPT_MODEL_URI": "gpt://folder-id/yandexgpt-5.1",
        "PHARMACHECK_USAGE_MODE": USAGE_MODE_OFFICIAL_LABELS_DEMO,
        "TELEGRAM_API_BASE": TELEGRAM_DEMO_RELAY_API_BASE,
        "TELEGRAM_BOT_TOKEN": "123:token",
        "TELEGRAM_WEBHOOK_SECRET": "telegram-secret",
        "PHARMACHECK_ANALYTICS_HMAC_KEY": "x" * 32,
        "OPENFDA_LABEL_URL": OPENFDA_RELAY_LABEL_URL,
        "OPENFDA_RELAY_SECRET": "relay-secret",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    _write_deployment_manifest(tmp_path, USAGE_MODE_OFFICIAL_LABELS_DEMO)

    config = RuntimeConfig.from_environment(root=tmp_path)

    assert config.openfda_label_url == OPENFDA_RELAY_LABEL_URL
    assert config.openfda_relay_secret == "relay-secret"

    monkeypatch.delenv("OPENFDA_RELAY_SECRET")
    with pytest.raises(RuntimeConfigurationError):
        RuntimeConfig.from_environment(root=tmp_path)

    monkeypatch.setenv("OPENFDA_RELAY_SECRET", "relay-secret")
    monkeypatch.setenv("OPENFDA_LABEL_URL", "https://untrusted.example.test")
    with pytest.raises(RuntimeConfigurationError):
        RuntimeConfig.from_environment(root=tmp_path)


def test_runtime_config_rejects_deployment_manifest_mode_mismatch(monkeypatch, tmp_path):
    values = {
        "YC_FOLDER_ID": "folder-id",
        "YANDEXGPT_MODEL_URI": "gpt://folder-id/yandexgpt-5.1",
        "PHARMACHECK_USAGE_MODE": USAGE_MODE_OFFICIAL_LABELS_DEMO,
        "TELEGRAM_BOT_TOKEN": "123:token",
        "TELEGRAM_WEBHOOK_SECRET": "secret",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    # Archive was built for noncommercial_mvp, but the function is
    # configured to run as official_labels_demo: must fail closed.
    _write_deployment_manifest(tmp_path, USAGE_MODE_NONCOMMERCIAL_MVP)

    with pytest.raises(RuntimeConfigurationError):
        RuntimeConfig.from_environment(root=tmp_path)


def test_runtime_config_rejects_missing_deployment_manifest(monkeypatch, tmp_path):
    values = {
        "YC_FOLDER_ID": "folder-id",
        "YANDEXGPT_MODEL_URI": "gpt://folder-id/yandexgpt-5.1",
        "PHARMACHECK_USAGE_MODE": USAGE_MODE_NONCOMMERCIAL_MVP,
        "TELEGRAM_BOT_TOKEN": "123:token",
        "TELEGRAM_WEBHOOK_SECRET": "secret",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(RuntimeConfigurationError):
        RuntimeConfig.from_environment(root=tmp_path)


def test_validate_deployment_manifest_rejects_unsupported_schema_version(tmp_path):
    (tmp_path / DEPLOYMENT_MANIFEST_NAME).write_text(
        json.dumps({"schema_version": 2, "usage_mode": USAGE_MODE_NONCOMMERCIAL_MVP}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeConfigurationError):
        _validate_deployment_manifest(tmp_path, USAGE_MODE_NONCOMMERCIAL_MVP)


def test_build_sources_noncommercial_mvp_uses_ddinter(tmp_path):
    ddinter_path = tmp_path / "ddinter"
    ddinter_path.mkdir()
    config = RuntimeConfig(
        telegram_bot_token="token",
        telegram_webhook_secret="secret",
        telegram_api_base="https://api.telegram.org/bot",
        folder_id="folder-id",
        model_uri="gpt://folder-id/yandexgpt-5.1",
        usage_mode=USAGE_MODE_NONCOMMERCIAL_MVP,
        catalog_path=tmp_path / "drug_catalog.json",
        catalog_manifest_path=tmp_path / "drug_catalog.manifest.json",
        ddinter_path=ddinter_path,
        prompt_path=tmp_path / "prompt.txt",
        analytics_hmac_key=None,
    )

    sources = _build_sources(
        config, rxnorm=RxNormClient(), rxclass=RxClassClient()
    )

    assert set(sources) == {"dailymed.label", "ddinter"}
    assert isinstance(sources["ddinter"], DDInterSource)


def test_build_sources_official_labels_demo_cannot_load_ddinter(tmp_path):
    config = RuntimeConfig(
        telegram_bot_token="token",
        telegram_webhook_secret="secret",
        telegram_api_base="https://api.telegram.org/bot",
        folder_id="folder-id",
        model_uri="gpt://folder-id/yandexgpt-5.1",
        usage_mode=USAGE_MODE_OFFICIAL_LABELS_DEMO,
        catalog_path=tmp_path / "drug_catalog.json",
        catalog_manifest_path=tmp_path / "drug_catalog.manifest.json",
        ddinter_path=None,
        prompt_path=tmp_path / "prompt.txt",
        analytics_hmac_key=b"x" * 32,
    )

    sources = _build_sources(
        config, rxnorm=RxNormClient(), rxclass=RxClassClient()
    )

    assert set(sources) == {"dailymed.label", "openfda.label"}
    assert isinstance(sources["openfda.label"], OpenFDALabelClient)
    assert not any(isinstance(source, DDInterSource) for source in sources.values())


def test_build_sources_noncommercial_mvp_requires_a_ddinter_path(tmp_path):
    config = RuntimeConfig(
        telegram_bot_token="token",
        telegram_webhook_secret="secret",
        telegram_api_base="https://api.telegram.org/bot",
        folder_id="folder-id",
        model_uri="gpt://folder-id/yandexgpt-5.1",
        usage_mode=USAGE_MODE_NONCOMMERCIAL_MVP,
        catalog_path=tmp_path / "drug_catalog.json",
        catalog_manifest_path=tmp_path / "drug_catalog.manifest.json",
        ddinter_path=None,
        prompt_path=tmp_path / "prompt.txt",
        analytics_hmac_key=None,
    )

    with pytest.raises(RuntimeConfigurationError):
        _build_sources(config, rxnorm=RxNormClient(), rxclass=RxClassClient())


def test_production_catalog_rejects_synthetic_and_accepts_verified(tmp_path):
    catalog = tmp_path / "catalog.json"
    manifest = _catalog(catalog, product_id="test.alpha", source_id="synthetic.catalog")
    with pytest.raises(RuntimeConfigurationError):
        _load_production_catalog(catalog, manifest)

    manifest = _catalog(catalog)
    loaded = _load_production_catalog(catalog, manifest)
    assert loaded.resolve_pair


def test_production_catalog_rejects_manifest_checksum_mismatch(tmp_path):
    catalog = tmp_path / "catalog.json"
    manifest = _catalog(catalog)
    catalog.write_text("[]", encoding="utf-8")
    with pytest.raises(RuntimeConfigurationError):
        _load_production_catalog(catalog, manifest)


def test_context_token_supports_yandex_mapping_and_object_shape():
    assert _context_access_token(Context(token={"access_token": "mapping-token"})) == "mapping-token"

    class Token:
        access_token = "object-token"

    assert _context_access_token(Context(token=Token())) == "object-token"


def test_deadline_budget_fails_before_external_calls():
    _check_budget(Context(remaining=115_000), "medical_sources")
    with pytest.raises(InvocationBudgetError):
        _check_budget(Context(remaining=114_999), "medical_sources")
    with pytest.raises(InvocationBudgetError):
        _check_budget(Context(remaining=64_999), "yandexgpt")


def test_cloud_handler_sanitizes_invalid_event_and_missing_configuration(monkeypatch):
    assert handle_cloud_function([], Context())["statusCode"] == 400
    monkeypatch.delenv("YC_FOLDER_ID", raising=False)
    response = handle_cloud_function({"headers": {}, "body": "{}"}, Context())
    assert response["statusCode"] == 503
    assert "YC_FOLDER_ID" not in response["body"]


def test_register_telegram_rejects_non_gateway_url():
    response = _register_telegram(
        {
            "pharmacheck_admin_operation": "register_telegram",
            "gateway_url": "https://example.test/telegram",
        }
    )
    assert response["statusCode"] == 400


def test_gateway_url_accepts_yandex_service_domain_with_technical_segment():
    url = "https://gateway-id.region-id.apigw.yandexcloud.net/telegram"
    assert GATEWAY_URL_RE.fullmatch(url)


def test_gateway_url_accepts_production_cloudflare_relay():
    url = "https://pharmacheck-telegram-relay.your-account.workers.dev/telegram"
    assert GATEWAY_URL_RE.fullmatch(url)


def test_gateway_url_accepts_isolated_demo_cloudflare_relay():
    url = "https://pharmacheck-demo-telegram-relay.your-account.workers.dev/telegram"
    assert GATEWAY_URL_RE.fullmatch(url)


def test_gateway_url_rejects_other_cloudflare_workers():
    url = "https://untrusted.your-account.workers.dev/telegram"
    assert GATEWAY_URL_RE.fullmatch(url) is None
