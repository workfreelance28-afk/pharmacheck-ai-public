"""Curated official_labels_demo control matrix (docs/demo/demo_control_matrix_2026-08-12.md).

Every scenario runs entirely offline against saved evidence fixtures in
`tests/fixtures/demo_control_matrix/`: no network, YandexGPT, Yandex Cloud,
or paid call is made. Required providers are always exactly
`dailymed.label` and `openfda.label`, matching the `official_labels_demo`
deployment mode (D-011); no other interaction source is imported,
constructed, or referenced anywhere in this file or its fixtures.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from pharmacheck.analytics import DemoAnalyticsRecorder
from pharmacheck.application import PharmaCheckService
from pharmacheck.contracts import (
    AnalysisResult,
    AnalysisStatus,
    ContractViolation,
    DrugIdentity,
    EvidenceDocument,
    InteractionLevel,
    ResultSource,
    RetrievalStatus,
    SearchResult,
    SearchResultStatus,
    validate_analysis_against_evidence,
)
from pharmacheck.drug_recognition import DrugCatalog, DrugCatalogEntry
from pharmacheck.evidence import build_evidence_package
from pharmacheck.sources.base import InteractionSearchOutcome
from pharmacheck.sources.dailymed import _extract_excerpt
from pharmacheck.telegram import format_analysis_result
from pharmacheck.telegram_webhook import RecentUpdateStore, TelegramWebhook
from pharmacheck.yandexgpt import YandexGPTAnalysis, YandexGPTUsage

CHECKED_AT = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "demo_control_matrix"
PUBLISHER_BY_PROVIDER = {
    "dailymed.label": "U.S. National Library of Medicine (DailyMed)",
    "openfda.label": "U.S. Food and Drug Administration (openFDA)",
}
REQUIRED_DEMO_PROVIDERS = ("dailymed.label", "openfda.label")


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))


def _excerpt_from_fixture(raw: dict) -> str:
    """Derive the saved excerpt using the real production matcher.

    Fixtures store a ``verified_section_excerpt`` — a minimal fragment
    confirmed against the live DailyMed label, not the full section — rather
    than a hand-typed excerpt, and this calls the same
    ``pharmacheck.sources.dailymed._extract_excerpt`` the DailyMed adapter
    uses in production. This makes each document's ``match_basis`` claim
    mechanically provable instead of merely asserted: a
    ``structured_class`` fixture must fail a direct match on the ingredient
    name and only succeed via the RxClass class name.
    """
    verified_excerpt = raw["verified_section_excerpt"]
    match_basis = raw.get("match_basis", "direct_ingredient")
    if match_basis == "structured_class":
        direct_target = raw["match_target"]
        class_target = raw["class_match_target"]
        assert _extract_excerpt(verified_excerpt, direct_target) is None, (
            f"{raw['source_id']}: fixture text directly names {direct_target!r}; "
            "this contradicts the structured-class match_basis claim"
        )
        excerpt = _extract_excerpt(verified_excerpt, class_target, allow_plural=True)
        assert excerpt is not None, (
            f"{raw['source_id']}: class term {class_target!r} has no match in fixture text"
        )
        return excerpt
    counterpart = raw["counterpart"]
    excerpt = _extract_excerpt(verified_excerpt, counterpart)
    assert excerpt is not None, (
        f"{raw['source_id']}: counterpart {counterpart!r} has no direct match in fixture text"
    )
    return excerpt


def _document_from_fixture(raw: dict, *, provider_id: str, pair: tuple[str, str]) -> EvidenceDocument:
    match_basis = raw.get("match_basis", "direct_ingredient")
    matched_term = raw.get("matched_term")
    if match_basis == "structured_class" and matched_term is None:
        matched_term = raw.get("class_match_target")
    return EvidenceDocument(
        source_id=raw["source_id"],
        title=raw["title"],
        publisher=PUBLISHER_BY_PROVIDER[provider_id],
        url=raw["url"],
        accessed_at=CHECKED_AT,
        source_tier=raw["source_tier"],
        source_type=raw["source_type"],
        language="en",
        applies_to=list(pair),
        section=raw.get("section"),
        spl_set_id=raw.get("spl_set_id"),
        excerpt=_excerpt_from_fixture(raw),
        retrieval_status=RetrievalStatus.OK,
        match_basis=match_basis,
        matched_term=matched_term,
        matched_ingredient=raw.get("matched_ingredient"),
        additional_urls=raw.get("additional_urls", []),
    )


def _outcomes_from_fixture(fixture: dict) -> list[InteractionSearchOutcome]:
    first, second = fixture["pair"]
    outcomes = []
    for raw_outcome in fixture["outcomes"]:
        provider_id = raw_outcome["provider_id"]
        assert provider_id in REQUIRED_DEMO_PROVIDERS, "demo fixtures use only dailymed/openfda"
        result = SearchResultStatus(raw_outcome["result"])
        documents = [
            _document_from_fixture(doc, provider_id=provider_id, pair=(first, second))
            for doc in raw_outcome.get("documents", [])
        ]
        outcomes.append(
            InteractionSearchOutcome(
                provider_id=provider_id,
                ingredient_pair=(first, second),
                search_result=SearchResult(
                    source_id=f"{provider_id}.search.{first}.{second}".casefold(),
                    title=f"{provider_id} search",
                    publisher=PUBLISHER_BY_PROVIDER[provider_id],
                    url=f"https://example.test/{provider_id}/search",
                    accessed_at=CHECKED_AT,
                    query_pairs=[f"{first} + {second}"],
                    searched_sections=["drug_interactions"],
                    result=result,
                ),
                documents=documents,
                complete=raw_outcome["complete"],
                failed_queries=raw_outcome.get("failed_queries", []),
            )
        )
    return outcomes


class FixtureSource:
    """Replays one fixture's saved outcome for a single provider."""

    def __init__(self, provider_id: str, outcome: InteractionSearchOutcome) -> None:
        self.provider_id = provider_id
        self._outcome = outcome
        self.calls: list[tuple[str, str]] = []

    def search_pair(self, ingredient_a, ingredient_b, *, checked_at=None):
        self.calls.append((ingredient_a, ingredient_b))
        return self._outcome


class CapturingAnalyzer:
    """Stands in for YandexGPTClient: returns a pre-built result, never calls a model."""

    def __init__(self, result: AnalysisResult | None = None) -> None:
        self.result = result
        self.calls = 0
        self.evidence = None

    def analyze(self, evidence):
        self.calls += 1
        self.evidence = evidence
        if self.result is None:
            raise AssertionError("CapturingAnalyzer.analyze() called with no result configured")
        return YandexGPTAnalysis(result=self.result, usage=YandexGPTUsage())


class ValidatingAnalyzer:
    """Stands in for YandexGPTClient, including its built-in safety check.

    The real ``YandexGPTClient.analyze`` (``src/pharmacheck/yandexgpt.py``)
    calls ``validate_analysis_against_evidence`` on the model's result before
    returning it. This double does the same against a pre-built result, so a
    full ``PharmaCheckService.check()`` run exercises that protection for
    real instead of only calling ``validate_analysis_against_evidence``
    directly against a hand-assembled evidence package.
    """

    def __init__(self, result: AnalysisResult) -> None:
        self.result = result
        self.calls = 0
        self.evidence = None

    def analyze(self, evidence):
        self.calls += 1
        self.evidence = evidence
        validate_analysis_against_evidence(evidence, self.result)
        return YandexGPTAnalysis(result=self.result, usage=YandexGPTUsage())


def _entry(product_id: str, trade_name: str, ingredient: str) -> DrugCatalogEntry:
    return DrugCatalogEntry(
        product_id=product_id,
        trade_name=trade_name,
        aliases=[],
        dosage_form="таблетки",
        strength=None,
        active_ingredients=[ingredient],
        source_id=f"grls.catalog.{product_id}",
    )


DEMO_CATALOG_ENTRIES = [
    _entry("amiodarone", "Амиодарон", "amiodarone"),
    _entry("voriconazole", "Вориконазол", "voriconazole"),
    _entry("nurofen", "Нурофен", "ibuprofen"),
    _entry("warfarin", "Варфарин", "warfarin"),
    _entry("cetirizine", "Цетиризин", "cetirizine"),
    _entry("levofloxacin", "Левофлоксацин", "levofloxacin"),
]


def _service(
    fixture_name: str | None,
    *,
    analyzer_result: AnalysisResult | None = None,
    analyzer_cls: type = CapturingAnalyzer,
):
    """Build a PharmaCheckService wired to exactly one fixture's saved evidence."""
    sources: dict[str, FixtureSource] = {}
    if fixture_name is not None:
        fixture = _load_fixture(fixture_name)
        outcomes = _outcomes_from_fixture(fixture)
        for provider_id in REQUIRED_DEMO_PROVIDERS:
            outcome = next(o for o in outcomes if o.provider_id == provider_id)
            sources[provider_id] = FixtureSource(provider_id, outcome)
    else:
        # DM-04/DM-05: sources must never be called, but PharmaCheckService
        # still requires at least two configured (official_labels_demo mode).
        for provider_id in REQUIRED_DEMO_PROVIDERS:
            fixture = _load_fixture("dm03_cetirizine_levofloxacin.json")
            outcome = next(
                o for o in _outcomes_from_fixture(fixture) if o.provider_id == provider_id
            )
            sources[provider_id] = FixtureSource(provider_id, outcome)

    analyzer = analyzer_cls(analyzer_result)
    service = PharmaCheckService(
        catalog=DrugCatalog(list(DEMO_CATALOG_ENTRIES)),
        sources=sources,
        analyzer=analyzer,
    )
    return service, sources, analyzer


def _answer_result(
    *,
    level: InteractionLevel,
    headline: str,
    document: EvidenceDocument,
    identified: list[tuple[str, str]],
) -> AnalysisResult:
    return AnalysisResult(
        status=AnalysisStatus.ANSWER,
        interaction_level=level,
        headline=headline,
        identified_drugs=[
            {"trade_name": trade, "active_ingredients": [ingredient]}
            for trade, ingredient in identified
        ],
        summary=f"Вывод основан на сохранённом источнике {document.source_id}.",
        mechanism_or_effect=None,
        recommended_user_action=(
            "Как можно скорее свяжитесь с назначившим врачом или фармацевтом. "
            "Не изменяйте лечение самостоятельно."
        ),
        limitations=["Это тестовый сценарий контрольной матрицы."],
        conflicts=[],
        missing_information=[],
        sources=[
            ResultSource(
                source_id=document.source_id,
                title=document.title,
                url=document.url,
                supports="Подтверждает контрольный сценарий.",
            )
        ],
    )


# --------------------------------------------------------------------------
# DM-01 — Амиодарон + Вориконазол: regulator-label class warning, red allowed
# --------------------------------------------------------------------------


def test_dm01_amiodarone_voriconazole_class_warning_allows_red():
    fixture = _load_fixture("dm01_amiodarone_voriconazole.json")
    raw_document = fixture["outcomes"][0]["documents"][0]
    verified_excerpt = raw_document["verified_section_excerpt"]

    # Prove, using the real production matcher (not a hand-typed assertion),
    # that this fixture's label text never names "voriconazole" directly —
    # only the azole antifungal class — so the structured-class match is
    # mechanically required, not merely claimed.
    assert _extract_excerpt(verified_excerpt, "voriconazole") is None
    assert _extract_excerpt(verified_excerpt, "Azoles", allow_plural=True) is not None

    outcomes = _outcomes_from_fixture(fixture)
    evidence = _evidence_for_pair_fixture(fixture, outcomes)

    assessment = evidence.deterministic_assessment
    assert assessment is not None
    document = evidence.documents[0]

    # Requirement: this must be genuine structured-class provenance, not a
    # plain word match anywhere in free text.
    assert document.section == "drug_interactions"
    assert document.match_basis == "structured_class"
    assert document.matched_term == "Azoles"
    assert document.matched_ingredient == "voriconazole"
    assert "voriconazole" not in document.excerpt.casefold()
    assert document.additional_urls, "structured-class match requires provenance URLs"
    assert "rxclass" in str(document.additional_urls[0]).casefold()

    assert assessment.red_eligible is True
    assert InteractionLevel.RED in assessment.allowed_levels

    result = _answer_result(
        level=InteractionLevel.RED,
        headline="Обнаружено серьёзное взаимодействие",
        document=document,
        identified=[("Амиодарон", "amiodarone"), ("Вориконазол", "voriconazole")],
    )
    validate_analysis_against_evidence(evidence, result)


def test_dm01_full_service_path_resolves_russian_input_and_returns_validated_red():
    """End-to-end PharmaCheckService.check() run for DM-01, not just a
    hand-assembled EvidencePackage/AnalysisResult pair.

    Exercises Russian-input catalog resolution, calling both demo sources
    exactly once each, and a validating analyzer standing in for
    YandexGPTClient's own built-in `validate_analysis_against_evidence`
    safety check (see ValidatingAnalyzer). DDInter is never configured
    (REQUIRED_DEMO_PROVIDERS is exactly dailymed.label/openfda.label).
    """
    fixture = _load_fixture("dm01_amiodarone_voriconazole.json")
    outcomes = _outcomes_from_fixture(fixture)
    evidence = _evidence_for_pair_fixture(fixture, outcomes)
    document = evidence.documents[0]

    result = _answer_result(
        level=InteractionLevel.RED,
        headline="Обнаружено серьёзное взаимодействие",
        document=document,
        identified=[("Амиодарон", "amiodarone"), ("Вориконазол", "voriconazole")],
    )

    service, sources, analyzer = _service(
        "dm01_amiodarone_voriconazole.json",
        analyzer_result=result,
        analyzer_cls=ValidatingAnalyzer,
    )

    outcome = service.check("Амиодарон + Вориконазол", request_id="dm01-service-path")

    assert outcome.status is AnalysisStatus.ANSWER
    assert outcome.interaction_level is InteractionLevel.RED
    assert analyzer.calls == 1, "the analyzer must be invoked exactly once"
    assert all(len(source.calls) == 1 for source in sources.values()), (
        "both demo sources (dailymed.label, openfda.label) must be searched exactly once"
    )
    assert set(sources) == {"dailymed.label", "openfda.label"}
    assert "Не изменяйте лечение самостоятельно" in outcome.recommended_user_action
    rendered = format_analysis_result(outcome)
    assert str(document.url) in rendered, "the first source's full URL must reach the reply text"


# --------------------------------------------------------------------------
# DM-02 — Нурофен + Варфарин: orange allowed, red forbidden without explicit
# avoid/contraindication wording
# --------------------------------------------------------------------------


def test_dm02_ibuprofen_warfarin_allows_orange_forbids_red():
    fixture = _load_fixture("dm02_ibuprofen_warfarin.json")
    outcomes = _outcomes_from_fixture(fixture)
    evidence = _evidence_for_pair_fixture(fixture, outcomes)

    assessment = evidence.deterministic_assessment
    assert assessment is not None
    assert assessment.red_eligible is False
    assert assessment.allowed_levels == [
        InteractionLevel.WHITE,
        InteractionLevel.YELLOW,
        InteractionLevel.ORANGE,
    ]

    document = evidence.documents[0]
    orange_result = _answer_result(
        level=InteractionLevel.ORANGE,
        headline="Обнаружено потенциально значимое взаимодействие",
        document=document,
        identified=[("Нурофен", "ibuprofen"), ("Варфарин", "warfarin")],
    )
    validate_analysis_against_evidence(evidence, orange_result)

    red_result = _answer_result(
        level=InteractionLevel.RED,
        headline="Опасная комбинация",
        document=document,
        identified=[("Нурофен", "ibuprofen"), ("Варфарин", "warfarin")],
    )
    with pytest.raises(ContractViolation, match="prohibited by deterministic evidence"):
        validate_analysis_against_evidence(evidence, red_result)


def test_dm02_full_service_path_resolves_russian_input_and_returns_validated_orange():
    """End-to-end PharmaCheckService.check() run for DM-02 (see DM-01's
    counterpart test for what this covers beyond the hand-assembled-evidence
    test above)."""
    fixture = _load_fixture("dm02_ibuprofen_warfarin.json")
    outcomes = _outcomes_from_fixture(fixture)
    evidence = _evidence_for_pair_fixture(fixture, outcomes)
    document = evidence.documents[0]

    result = _answer_result(
        level=InteractionLevel.ORANGE,
        headline="Обнаружено потенциально значимое взаимодействие",
        document=document,
        identified=[("Нурофен", "ibuprofen"), ("Варфарин", "warfarin")],
    )

    service, sources, analyzer = _service(
        "dm02_ibuprofen_warfarin.json",
        analyzer_result=result,
        analyzer_cls=ValidatingAnalyzer,
    )

    outcome = service.check("Нурофен + Варфарин", request_id="dm02-service-path")

    assert outcome.status is AnalysisStatus.ANSWER
    assert outcome.interaction_level is InteractionLevel.ORANGE
    assert analyzer.calls == 1, "the analyzer must be invoked exactly once"
    assert all(len(source.calls) == 1 for source in sources.values()), (
        "both demo sources (dailymed.label, openfda.label) must be searched exactly once"
    )
    assert set(sources) == {"dailymed.label", "openfda.label"}
    assert "Не изменяйте лечение самостоятельно" in outcome.recommended_user_action
    rendered = format_analysis_result(outcome)
    assert str(document.url) in rendered, "the first source's full URL must reach the reply text"


def test_dm02_full_service_path_rejects_red_via_validating_analyzer():
    """A model attempting red for DM-02 must be blocked before it reaches
    the reply, exactly as the real YandexGPTClient's built-in validation
    would block it — not only when validate_analysis_against_evidence is
    called directly against a hand-assembled evidence package."""
    fixture = _load_fixture("dm02_ibuprofen_warfarin.json")
    outcomes = _outcomes_from_fixture(fixture)
    evidence = _evidence_for_pair_fixture(fixture, outcomes)
    document = evidence.documents[0]

    red_result = _answer_result(
        level=InteractionLevel.RED,
        headline="Опасная комбинация",
        document=document,
        identified=[("Нурофен", "ibuprofen"), ("Варфарин", "warfarin")],
    )

    service, _sources, analyzer = _service(
        "dm02_ibuprofen_warfarin.json",
        analyzer_result=red_result,
        analyzer_cls=ValidatingAnalyzer,
    )

    with pytest.raises(ContractViolation, match="prohibited by deterministic evidence"):
        service.check("Нурофен + Варфарин", request_id="dm02-service-path-red-rejected")
    assert analyzer.calls == 1


# --------------------------------------------------------------------------
# DM-03 — Цетиризин + Левофлоксацин: clean negative pair-literal search,
# fail-closed white without ever calling the model
# --------------------------------------------------------------------------


def test_dm03_cetirizine_levofloxacin_fails_closed_without_model():
    fixture = _load_fixture("dm03_cetirizine_levofloxacin.json")
    outcomes = _outcomes_from_fixture(fixture)
    evidence = _evidence_for_pair_fixture(fixture, outcomes)
    # A clean pair-literal negative search structurally *permits* green at
    # the raw evidence level (D-004 is a deliberate extra guard, not a
    # consequence of the assessment alone) — which is exactly why the
    # service-level fail-closed override below is required.
    assert InteractionLevel.GREEN in evidence.deterministic_assessment.allowed_levels

    service, sources, analyzer = _service("dm03_cetirizine_levofloxacin.json")

    result = service.check("Цетиризин + Левофлоксацин", request_id="dm03")

    assert result.status is AnalysisStatus.INSUFFICIENT_DATA
    assert result.interaction_level is InteractionLevel.WHITE
    assert result.interaction_level is not InteractionLevel.GREEN
    assert analyzer.calls == 0, "a clean negative search must never reach the model"
    assert all(source.calls for source in sources.values())
    assert any(
        "не доказывает безопасность" in limitation for limitation in result.limitations
    )


# --------------------------------------------------------------------------
# DM-04 — заведомо неподдерживаемое название: clarification, no medical calls
# --------------------------------------------------------------------------


def test_dm04_unsupported_drug_returns_clarification_without_medical_calls():
    service, sources, analyzer = _service(None)

    result = service.check("Несуществин + Варфарин", request_id="dm04")

    assert result.status is AnalysisStatus.CLARIFICATION_NEEDED
    assert result.interaction_level is InteractionLevel.WHITE
    assert analyzer.calls == 0
    assert all(source.calls == [] for source in sources.values())


# --------------------------------------------------------------------------
# DM-05 — только один препарат: usage guidance, no medical conclusion
# --------------------------------------------------------------------------


def test_dm05_single_drug_input_requests_two_drugs_without_medical_calls():
    service, sources, analyzer = _service(None)

    result = service.check("Нурофен", request_id="dm05")

    assert result.status is AnalysisStatus.CLARIFICATION_NEEDED
    assert result.interaction_level is InteractionLevel.WHITE
    assert analyzer.calls == 0
    assert all(source.calls == [] for source in sources.values())


# --------------------------------------------------------------------------
# DM-06 — partial required-source failure: insufficient_data/white; a
# colored result must fail validation (regression fix in evidence.py)
# --------------------------------------------------------------------------


def test_dm06_partial_source_failure_stays_white_and_forbids_color():
    fixture = _load_fixture("dm06_partial_source_failure.json")
    outcomes = _outcomes_from_fixture(fixture)
    evidence = _evidence_for_pair_fixture(fixture, outcomes)

    assessment = evidence.deterministic_assessment
    assert assessment is not None
    assert evidence.search_coverage.complete is False
    assert len(evidence.search_coverage.failed_sources) == 1
    assert assessment.red_eligible is False
    assert assessment.allowed_levels == [InteractionLevel.WHITE]
    assert any(
        "неполно" in limitation for limitation in assessment.limitations
    )

    white_result = AnalysisResult(
        status=AnalysisStatus.INSUFFICIENT_DATA,
        interaction_level=InteractionLevel.WHITE,
        headline="Недостаточно данных для безопасного вывода",
        identified_drugs=[],
        summary="Один обязательный источник не завершил поиск.",
        mechanism_or_effect=None,
        recommended_user_action="Обсудите комбинацию с врачом или фармацевтом.",
        limitations=assessment.limitations,
        conflicts=[],
        missing_information=["Обязательный источник openFDA не ответил вовремя."],
        sources=[],
    )
    validate_analysis_against_evidence(evidence, white_result)

    document = evidence.documents[0]
    for level in (InteractionLevel.YELLOW, InteractionLevel.ORANGE, InteractionLevel.RED):
        colored_result = _answer_result(
            level=level,
            headline="Недопустимый вывод",
            document=document,
            identified=[("Нурофен", "ibuprofen"), ("Варфарин", "warfarin")],
        )
        with pytest.raises(ContractViolation, match="prohibited by deterministic evidence"):
            validate_analysis_against_evidence(evidence, colored_result)


def _evidence_for_pair_fixture(fixture: dict, outcomes: list[InteractionSearchOutcome]):
    first_name, second_name = fixture["pair"]
    return build_evidence_package(
        request_id=f"{fixture['scenario_id']}.evidence",
        user_message=f"{first_name} + {second_name}",
        drugs=[
            DrugIdentity(
                input_name=first_name,
                normalized_trade_name=first_name,
                active_ingredients=[first_name],
                identity_status="resolved",
                identity_confidence="high",
                identity_source_ids=[f"grls.catalog.{first_name}"],
            ),
            DrugIdentity(
                input_name=second_name,
                normalized_trade_name=second_name,
                active_ingredients=[second_name],
                identity_status="resolved",
                identity_confidence="high",
                identity_source_ids=[f"grls.catalog.{second_name}"],
            ),
        ],
        outcomes=outcomes,
        required_provider_ids=REQUIRED_DEMO_PROVIDERS,
        checked_at=CHECKED_AT,
    )


# --------------------------------------------------------------------------
# DM-07 — repeated Telegram update_id: exactly one check, one reply, one
# analytics event
# --------------------------------------------------------------------------


class _FakeBot:
    def __init__(self) -> None:
        self.messages = []

    def send_message(self, **payload):
        self.messages.append(payload)


class _FakeCheckService:
    def __init__(self, result: AnalysisResult) -> None:
        self._result = result
        self.calls = []

    def check(self, message, *, request_id):
        self.calls.append((message, request_id))
        return self._result


def test_dm07_duplicate_telegram_update_runs_check_once():
    fixture = _load_fixture("dm03_cetirizine_levofloxacin.json")
    outcomes = _outcomes_from_fixture(fixture)
    evidence = _evidence_for_pair_fixture(fixture, outcomes)
    result = AnalysisResult(
        status=AnalysisStatus.INSUFFICIENT_DATA,
        interaction_level=InteractionLevel.WHITE,
        headline="Недостаточно данных для безопасного вывода",
        identified_drugs=[],
        summary="Прямое упоминание пары не найдено.",
        mechanism_or_effect=None,
        recommended_user_action="Обсудите комбинацию с врачом или фармацевтом.",
        limitations=evidence.deterministic_assessment.limitations,
        conflicts=[],
        missing_information=["Прямое упоминание пары не найдено."],
        sources=[],
    )

    bot = _FakeBot()
    service = _FakeCheckService(result)
    logger_records: list[str] = []

    import logging

    logger = logging.getLogger("test.demo_control_matrix.dm07")
    logger.setLevel(logging.INFO)

    class _Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            logger_records.append(record.getMessage())

    logger.addHandler(_Handler())
    logger.propagate = False
    analytics = DemoAnalyticsRecorder(
        hmac_key=b"x" * 32, usage_mode="official_labels_demo", logger=logger
    )

    webhook = TelegramWebhook(
        webhook_secret="test-secret",
        bot_client=bot,
        check_service=service,
        update_store=RecentUpdateStore(),
        analytics=analytics,
    )

    event = {
        "headers": {"X-Telegram-Bot-Api-Secret-Token": "test-secret"},
        "body": json.dumps(
            {
                "update_id": 4242,
                "message": {
                    "message_id": 1,
                    "from": {"id": 777, "is_bot": False},
                    "chat": {"id": 777, "type": "private"},
                    "text": "Цетиризин + Левофлоксацин",
                },
            }
        ),
        "isBase64Encoded": False,
    }

    first = webhook.handle(event, request_id="dm07-first")
    second = webhook.handle(event, request_id="dm07-second")

    assert first["statusCode"] == second["statusCode"] == 200
    assert len(service.calls) == 1
    assert len(bot.messages) == 1
    assert len(logger_records) == 1
    event_payload = json.loads(logger_records[0])
    assert event_payload["outcome"] == "insufficient_data"


# --------------------------------------------------------------------------
# Mechanical guard: no demo fixture or scenario ever names a source other
# than dailymed.label/openfda.label (requirement 9 of the matrix spec).
# --------------------------------------------------------------------------


def test_demo_fixtures_use_only_dailymed_and_openfda_providers():
    assert REQUIRED_DEMO_PROVIDERS == ("dailymed.label", "openfda.label")
    for path in sorted(FIXTURES_DIR.glob("*.json")):
        fixture = json.loads(path.read_text(encoding="utf-8"))
        provider_ids = {outcome["provider_id"] for outcome in fixture["outcomes"]}
        assert provider_ids <= set(REQUIRED_DEMO_PROVIDERS), path.name
        assert "ddinter" not in path.read_text(encoding="utf-8").casefold()
