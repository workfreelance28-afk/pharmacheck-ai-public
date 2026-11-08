from __future__ import annotations

from datetime import UTC, datetime

from pharmacheck.application import PharmaCheckService
from pharmacheck.contracts import (
    AnalysisResult,
    EvidenceDocument,
    RetrievalStatus,
    SearchResult,
    SearchResultStatus,
)
from pharmacheck.drug_recognition import DrugCatalog, DrugCatalogEntry
from pharmacheck.sources.base import InteractionSearchOutcome
from pharmacheck.yandexgpt import YandexGPTAnalysis, YandexGPTUsage

CHECKED_AT = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)


class NegativeSource:
    def __init__(self, provider_id: str) -> None:
        self.provider_id = provider_id
        self.calls = []

    def search_pair(self, ingredient_a, ingredient_b, *, checked_at=None):
        self.calls.append((ingredient_a, ingredient_b))
        suffix = f"{ingredient_a}.{ingredient_b}".casefold()
        return InteractionSearchOutcome(
            provider_id=self.provider_id,
            ingredient_pair=(ingredient_a, ingredient_b),
            search_result=SearchResult(
                source_id=f"{self.provider_id}.search.{suffix}",
                title=f"{self.provider_id} search",
                publisher=self.provider_id,
                url=f"https://example.test/{self.provider_id}",
                accessed_at=checked_at or CHECKED_AT,
                query_pairs=[f"{ingredient_a} + {ingredient_b}"],
                searched_sections=["interactions"],
                additional_urls=[f"https://example.test/{self.provider_id}/second"],
                result=SearchResultStatus.NO_RELEVANT_INTERACTION_FOUND,
            ),
            complete=True,
        )


class UnknownInteractionSource:
    provider_id = "ddinter"

    def __init__(self) -> None:
        self.calls = []

    def search_pair(self, ingredient_a, ingredient_b, *, checked_at=None):
        self.calls.append((ingredient_a, ingredient_b))
        timestamp = checked_at or CHECKED_AT
        return InteractionSearchOutcome(
            provider_id=self.provider_id,
            ingredient_pair=(ingredient_a, ingredient_b),
            search_result=SearchResult(
                source_id="ddinter.search.unknown",
                title="DDInter search",
                publisher="DDInter",
                url="https://example.test/ddinter/search",
                accessed_at=timestamp,
                query_pairs=[f"{ingredient_a} + {ingredient_b}"],
                searched_sections=["severity"],
                result=SearchResultStatus.INTERACTION_FOUND,
            ),
            documents=[
                EvidenceDocument(
                    source_id="ddinter.record.unknown",
                    title="DDInter unknown interaction",
                    publisher="DDInter",
                    url="https://example.test/ddinter/record",
                    accessed_at=timestamp,
                    source_tier=3,
                    source_type="interaction_database",
                    language="en",
                    applies_to=[ingredient_a, ingredient_b],
                    section="severity",
                    source_severity="unknown",
                    excerpt="The interaction severity is Unknown.",
                    retrieval_status=RetrievalStatus.OK,
                )
            ],
            complete=True,
        )


class KnownInteractionSource(UnknownInteractionSource):
    def __init__(self, severity: str) -> None:
        super().__init__()
        self.severity = severity

    def search_pair(self, ingredient_a, ingredient_b, *, checked_at=None):
        outcome = super().search_pair(
            ingredient_a,
            ingredient_b,
            checked_at=checked_at,
        )
        document = outcome.documents[0].model_copy(
            update={
                "source_id": f"ddinter.record.{self.severity}",
                "title": f"DDInter {self.severity} interaction",
                "source_severity": self.severity,
                "excerpt": f"The interaction severity is {self.severity}.",
            }
        )
        return InteractionSearchOutcome(
            provider_id=outcome.provider_id,
            ingredient_pair=outcome.ingredient_pair,
            search_result=outcome.search_result,
            documents=[document],
            complete=outcome.complete,
        )


class CapturingAnalyzer:
    def __init__(self) -> None:
        self.evidence = None

    def analyze(self, evidence):
        self.evidence = evidence
        result = AnalysisResult.model_validate(
            {
                "status": "answer",
                "interaction_level": "green",
                "headline": "Значимого известного взаимодействия не найдено",
                "identified_drugs": [
                    {"trade_name": "Альфа", "active_ingredients": ["alpha"]},
                    {"trade_name": "Бета", "active_ingredients": ["beta"]},
                ],
                "summary": "В двух проверенных источниках взаимодействие не найдено.",
                "mechanism_or_effect": None,
                "recommended_user_action": "Не изменяйте лечение самостоятельно.",
                "limitations": ["Отсутствие сведений не доказывает безопасность."],
                "conflicts": [],
                "missing_information": [],
                "sources": [
                    {
                        "source_id": evidence.search_results[0].source_id,
                        "title": evidence.search_results[0].title,
                        "url": str(evidence.search_results[0].url),
                        "supports": "В проверенных разделах прямое упоминание не найдено.",
                    }
                ],
            }
        )
        return YandexGPTAnalysis(result=result, usage=YandexGPTUsage())


def _entry(product_id, trade_name, ingredient, *, strength=None):
    return DrugCatalogEntry(
        product_id=product_id,
        trade_name=trade_name,
        aliases=[],
        dosage_form="таблетки",
        strength=strength,
        active_ingredients=[ingredient],
        source_id=f"official.catalog.{product_id}",
    )


def _service(*entries):
    sources = {
        "openfda.label": NegativeSource("openfda.label"),
        "ddinter": NegativeSource("ddinter"),
    }
    analyzer = CapturingAnalyzer()
    stages = []
    service = PharmaCheckService(
        catalog=DrugCatalog(entries),
        sources=sources,
        analyzer=analyzer,
        budget_check=stages.append,
    )
    return service, sources, analyzer, stages


def test_incomplete_message_stops_before_sources_and_model():
    service, sources, analyzer, stages = _service(
        _entry("alpha", "Альфа", "alpha"),
        _entry("beta", "Бета", "beta"),
    )

    result = service.check("Альфа", request_id="request.one")

    assert result.status.value == "clarification_needed"
    assert not stages
    assert analyzer.evidence is None
    assert all(not source.calls for source in sources.values())


def test_unknown_drug_stops_before_sources_and_model():
    service, sources, analyzer, stages = _service(
        _entry("alpha", "Альфа", "alpha"),
        _entry("beta", "Бета", "beta"),
    )

    result = service.check("НесуществующийПрепарат + Бета", request_id="request.unknown")

    assert result.status.value == "clarification_needed"
    assert not stages
    assert analyzer.evidence is None
    assert all(not source.calls for source in sources.values())


def test_ambiguous_name_stops_before_sources_and_lists_candidates():
    service, sources, analyzer, stages = _service(
        _entry("alpha.10", "Альфа", "alpha", strength="10мг"),
        _entry("alpha.20", "Альфа", "alpha", strength="20мг"),
        _entry("beta", "Бета", "beta"),
    )

    result = service.check("Альфа + Бета", request_id="request.two")

    assert result.status.value == "clarification_needed"
    assert "10мг" in result.missing_information[0]
    assert "20мг" in result.missing_information[0]
    assert not stages
    assert analyzer.evidence is None
    assert all(not source.calls for source in sources.values())


def test_pair_only_negative_sources_fail_closed_before_analyzer():
    service, sources, analyzer, stages = _service(
        _entry("alpha", "Альфа", "alpha"),
        _entry("beta", "Бета", "beta"),
    )

    result = service.check("Альфа + Бета", request_id="request.three")

    assert result.status.value == "insufficient_data"
    assert result.interaction_level.value == "white"
    assert stages == ["medical_sources"]
    assert all(source.calls == [("alpha", "beta")] for source in sources.values())
    assert analyzer.evidence is None
    assert result.sources[0].supports.startswith("Дополнительный поиск источника: https://")


def test_amiodarone_voriconazole_cannot_receive_green_from_literal_negatives():
    analyzer = CapturingAnalyzer()
    stages = []
    service = PharmaCheckService(
        catalog=DrugCatalog(
            [
                _entry("amiodarone", "Амиодарон", "amiodarone"),
                _entry("voriconazole", "Вориконазол", "voriconazole"),
            ]
        ),
        sources={
            "dailymed.label": NegativeSource("dailymed.label"),
            "ddinter": NegativeSource("ddinter"),
        },
        analyzer=analyzer,
        budget_check=stages.append,
    )

    result = service.check(
        "Амиодарон + Вориконазол",
        request_id="request.amiodarone-voriconazole",
    )

    assert result.status.value == "insufficient_data"
    assert result.interaction_level.value == "white"
    assert "классы препаратов" in result.summary
    assert analyzer.evidence is None
    assert stages == ["medical_sources"]


def test_unknown_database_severity_returns_white_without_model_call():
    analyzer = CapturingAnalyzer()
    stages = []
    service = PharmaCheckService(
        catalog=DrugCatalog(
            [
                _entry("alpha", "Альфа", "alpha"),
                _entry("beta", "Бета", "beta"),
            ]
        ),
        sources={
            "dailymed.label": NegativeSource("dailymed.label"),
            "ddinter": UnknownInteractionSource(),
        },
        analyzer=analyzer,
        budget_check=stages.append,
    )

    result = service.check("Альфа + Бета", request_id="request.unknown-severity")

    assert result.status.value == "insufficient_data"
    assert result.interaction_level.value == "white"
    assert len(result.sources) == 2
    assert analyzer.evidence is None
    assert stages == ["medical_sources"]


def test_minor_database_severity_returns_yellow_without_model_call():
    analyzer = CapturingAnalyzer()
    stages = []
    service = PharmaCheckService(
        catalog=DrugCatalog(
            [
                _entry("alpha", "Альфа", "alpha"),
                _entry("beta", "Бета", "beta"),
            ]
        ),
        sources={
            "dailymed.label": NegativeSource("dailymed.label"),
            "ddinter": KnownInteractionSource("minor"),
        },
        analyzer=analyzer,
        budget_check=stages.append,
    )

    result = service.check("Альфа + Бета", request_id="request.minor-severity")

    assert result.status.value == "answer"
    assert result.interaction_level.value == "yellow"
    assert len(result.sources) == 2
    assert analyzer.evidence is None
    assert stages == ["medical_sources"]


def test_major_database_severity_returns_orange_without_model_call():
    analyzer = CapturingAnalyzer()
    stages = []
    service = PharmaCheckService(
        catalog=DrugCatalog(
            [
                _entry("alpha", "Альфа", "alpha"),
                _entry("beta", "Бета", "beta"),
            ]
        ),
        sources={
            "dailymed.label": NegativeSource("dailymed.label"),
            "ddinter": KnownInteractionSource("major"),
        },
        analyzer=analyzer,
        budget_check=stages.append,
    )

    result = service.check("Альфа + Бета", request_id="request.major-severity")

    assert result.status.value == "answer"
    assert result.interaction_level.value == "orange"
    assert analyzer.evidence is None
    assert stages == ["medical_sources"]
