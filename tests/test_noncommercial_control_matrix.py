"""Offline noncommercial_mvp release matrix; no real DDInter rows are copied.

The real DDInter adapter reads tiny, synthetic CSVs created only in tmp_path.
Severity assignments and IDs here are test inputs, not medical reference data.
Official-label excerpts reuse the saved, verified DM-01/DM-02 test fixtures.
An analyzer double validates fixed answers; YandexGPT and network are forbidden.
"""

from __future__ import annotations

import csv
import hashlib
import json
import socket
from datetime import UTC, datetime
from pathlib import Path

import pytest

from pharmacheck.application import PharmaCheckService
from pharmacheck.contracts import (
    AnalysisResult,
    AnalysisStatus,
    ContractViolation,
    DrugIdentity,
    EvidenceDocument,
    InteractionLevel,
    SearchResult,
    SearchResultStatus,
    validate_analysis_against_evidence,
)
from pharmacheck.drug_recognition import DrugCatalog, DrugCatalogEntry
from pharmacheck.evidence import build_evidence_package
from pharmacheck.sources.base import InteractionSearchOutcome
from pharmacheck.sources.dailymed import _extract_excerpt
from pharmacheck.sources.ddinter import EXPECTED_COLUMNS, EXPECTED_FILES, DDInterSource
from pharmacheck.telegram import format_analysis_result
from pharmacheck.yandexgpt import (
    YandexGPTAnalysis,
    YandexGPTClient,
    YandexGPTUsage,
    _clamp_disallowed_red_to_orange,
)

CHECKED_AT = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)
LABEL_FIXTURES = Path(__file__).parent / "fixtures" / "demo_control_matrix"
TRADE_NAMES = {
    "amiodarone": "Амиодарон",
    "voriconazole": "Вориконазол",
    "ibuprofen": "Нурофен",
    "warfarin": "Варфарин",
    "loratadine": "Лоратадин",
    "clarithromycin": "Кларитромицин",
}


@pytest.fixture(autouse=True)
def forbid_network_and_yandexgpt(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("release matrix must not call network or YandexGPT")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(YandexGPTClient, "analyze", forbidden)


def _synthetic_snapshot(root, pair, severity, *, missing_name=False):
    root.mkdir()
    files = []
    for index, filename in enumerate(sorted(EXPECTED_FILES)):
        path = root / filename
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(EXPECTED_COLUMNS)
            if index == 0:
                writer.writerow(
                    (
                        "DDInter900001",
                        pair[0],
                        "DDInter900002",
                        "synthetic absent-name sentinel" if missing_name else pair[1],
                        severity,
                    )
                )
        payload = path.read_bytes()
        files.append(
            {
                "name": filename,
                "url": f"https://ddinter2.scbdd.com/static/media/download/{filename}",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_name": "DDInter 2.0",
                "snapshot_version": "synthetic-release-matrix",
                "release_date": None,
                "downloaded_at": CHECKED_AT.isoformat(),
                "license_id": "CC-BY-NC-SA-4.0",
                "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
                "terms_url": "https://ddinter2.scbdd.com/terms/",
                "usage_mode": "noncommercial_mvp",
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    return root


class SavedDailyMedSource:
    def __init__(self, fixture_name):
        self.fixture_name = fixture_name
        self.outcomes = []

    def search_pair(self, first, second, *, checked_at=None):
        documents = []
        if self.fixture_name:
            fixture = json.loads((LABEL_FIXTURES / self.fixture_name).read_text())
            assert fixture["pair"] == [first, second]
            raw_outcome = next(
                item for item in fixture["outcomes"] if item["provider_id"] == "dailymed.label"
            )
            for raw in raw_outcome["documents"]:
                term = raw.get("matched_term", raw.get("counterpart"))
                excerpt = _extract_excerpt(
                    raw["verified_section_excerpt"],
                    term,
                    allow_plural=raw["match_basis"] == "structured_class",
                )
                assert excerpt is not None
                if raw["match_basis"] == "structured_class":
                    assert _extract_excerpt(raw["verified_section_excerpt"], second) is None
                documents.append(
                    EvidenceDocument(
                        source_id=raw["source_id"],
                        title=raw["title"],
                        publisher="U.S. National Library of Medicine (DailyMed)",
                        url=raw["url"],
                        accessed_at=CHECKED_AT,
                        source_tier=raw["source_tier"],
                        source_type=raw["source_type"],
                        spl_set_id=raw["spl_set_id"],
                        language="en",
                        applies_to=[first, second],
                        section=raw["section"],
                        excerpt=excerpt,
                        retrieval_status="ok",
                        match_basis=raw["match_basis"],
                        matched_term=term,
                        matched_ingredient=raw.get("matched_ingredient"),
                        additional_urls=raw.get("additional_urls", []),
                    )
                )
        outcome = InteractionSearchOutcome(
            provider_id="dailymed.label",
            ingredient_pair=(first, second),
            search_result=SearchResult(
                source_id=f"dailymed.label.search.{first}.{second}",
                title="Saved DailyMed label search",
                publisher="U.S. National Library of Medicine (DailyMed)",
                url="https://dailymed.nlm.nih.gov/dailymed/",
                accessed_at=CHECKED_AT,
                query_pairs=[f"{first} + {second}"],
                searched_sections=["drug_interactions"],
                result=(
                    SearchResultStatus.INTERACTION_FOUND
                    if documents
                    else SearchResultStatus.NO_RELEVANT_INTERACTION_FOUND
                ),
            ),
            documents=documents,
            complete=True,
        )
        self.outcomes.append(outcome)
        return outcome


class RecordingDDInterSource(DDInterSource):
    def __init__(self, snapshot):
        super().__init__(snapshot)
        self.outcomes = []

    def search_pair(self, first, second, *, checked_at=None):
        outcome = super().search_pair(first, second, checked_at=CHECKED_AT)
        self.outcomes.append(outcome)
        return outcome


def _result(evidence, level):
    source = next(iter(evidence.documents), evidence.search_results[0])
    return AnalysisResult(
        status=AnalysisStatus.INSUFFICIENT_DATA
        if level is InteractionLevel.WHITE
        else AnalysisStatus.ANSWER,
        interaction_level=level,
        headline="Контрольный ответ офлайн-матрицы",
        identified_drugs=[
            {
                "trade_name": drug.normalized_trade_name,
                "active_ingredients": drug.active_ingredients,
            }
            for drug in evidence.drugs
        ],
        summary="Тестовый ответ по сохранённым свидетельствам.",
        mechanism_or_effect=None,
        recommended_user_action="Не изменяйте лечение самостоятельно.",
        limitations=["Синтетические данные; это не клиническая проверка."],
        conflicts=[],
        missing_information=["Полные результаты обязательного поиска."]
        if level is InteractionLevel.WHITE
        else [],
        sources=[
            {
                "source_id": source.source_id,
                "title": source.title,
                "url": source.url,
                "supports": "Источник из офлайн-фикстуры.",
            }
        ],
    )


class ValidatingAnalyzer:
    """Fixed answer plus the real clamp/validator, without a model client."""

    def __init__(self, level):
        self.level = level
        self.calls = 0

    def analyze(self, evidence):
        self.calls += 1
        assert self.level is not None, "deterministic scenario unexpectedly reached analyzer"
        result = _clamp_disallowed_red_to_orange(evidence, _result(evidence, self.level))
        validate_analysis_against_evidence(evidence, result)
        return YandexGPTAnalysis(result=result, usage=YandexGPTUsage())


def _run(
    tmp_path,
    pair,
    severity,
    *,
    label=None,
    analyzer_level=None,
    missing_name=False,
    message=None,
):
    sources = {
        "dailymed.label": SavedDailyMedSource(label),
        "ddinter": RecordingDDInterSource(
            _synthetic_snapshot(tmp_path / "ddinter", pair, severity, missing_name=missing_name)
        ),
    }
    analyzer = ValidatingAnalyzer(analyzer_level)
    catalog = DrugCatalog(
        [
            DrugCatalogEntry(
                product_id=ingredient,
                trade_name=trade_name,
                active_ingredients=[ingredient],
                source_id=f"grls.catalog.{ingredient}",
            )
            for ingredient, trade_name in TRADE_NAMES.items()
        ]
    )
    service = PharmaCheckService(catalog=catalog, sources=sources, analyzer=analyzer)
    message = message or " + ".join(TRADE_NAMES[name] for name in pair)
    result = service.check(message, request_id="noncommercial-matrix")
    assert all(len(source.outcomes) == 1 for source in sources.values())
    assert all(source.outcomes[0].ingredient_pair == pair for source in sources.values())
    evidence = build_evidence_package(
        request_id="noncommercial-matrix",
        user_message=message,
        drugs=[
            DrugIdentity(
                input_name=TRADE_NAMES[name],
                normalized_trade_name=TRADE_NAMES[name],
                active_ingredients=[name],
                identity_status="resolved",
                identity_confidence="high",
            )
            for name in pair
        ],
        outcomes=[source.outcomes[0] for source in sources.values()],
        required_provider_ids=tuple(sources),
        checked_at=CHECKED_AT,
    )
    validate_analysis_against_evidence(evidence, result)
    rendered = format_analysis_result(result)
    assert str(result.sources[0].url) in rendered
    assert "Не изменяйте лечение самостоятельно" in rendered
    return result, evidence, analyzer


@pytest.mark.parametrize(
    ("pair", "fixture", "level"),
    [
        (("amiodarone", "voriconazole"), "dm01_amiodarone_voriconazole.json", InteractionLevel.RED),
        (("ibuprofen", "warfarin"), "dm02_ibuprofen_warfarin.json", InteractionLevel.ORANGE),
    ],
)
def test_regulator_label_and_ddinter_control_pairs(tmp_path, pair, fixture, level):
    result, evidence, analyzer = _run(tmp_path, pair, "Major", label=fixture, analyzer_level=level)
    assert result.status is AnalysisStatus.ANSWER
    assert result.interaction_level is level
    assert analyzer.calls == 1
    assessment = evidence.deterministic_assessment
    assert assessment.red_eligible is (level is InteractionLevel.RED)
    assert any(doc.source_type == "regulator_label" for doc in evidence.documents)
    assert any(doc.source_severity == "major" for doc in evidence.documents)


@pytest.mark.parametrize(
    ("pair", "severity", "level"),
    [
        (("loratadine", "clarithromycin"), "Minor", InteractionLevel.YELLOW),
        (("loratadine", "clarithromycin"), "Moderate", InteractionLevel.ORANGE),
        (("loratadine", "clarithromycin"), "Major", InteractionLevel.ORANGE),
        (("amiodarone", "voriconazole"), "Major", InteractionLevel.ORANGE),
        (("ibuprofen", "warfarin"), "Major", InteractionLevel.ORANGE),
    ],
)
def test_ddinter_only_is_deterministic_and_never_red(tmp_path, pair, severity, level):
    result, evidence, analyzer = _run(tmp_path, pair, severity)
    assert result.status is AnalysisStatus.ANSWER
    assert result.interaction_level is level
    assert analyzer.calls == 0
    assert all(doc.source_type == "interaction_database" for doc in evidence.documents)
    assert evidence.deterministic_assessment.red_eligible is False
    with pytest.raises(ContractViolation, match="prohibited by deterministic evidence"):
        validate_analysis_against_evidence(evidence, _result(evidence, InteractionLevel.RED))


def test_ddinter_unknown_returns_white_without_analyzer(tmp_path):
    result, evidence, analyzer = _run(tmp_path, ("loratadine", "clarithromycin"), "Unknown")
    assert result.status is AnalysisStatus.INSUFFICIENT_DATA
    assert result.interaction_level is InteractionLevel.WHITE
    assert analyzer.calls == 0
    assert evidence.search_coverage.complete is True
    assert evidence.documents[0].source_severity == "unknown"
    assert "не доказывают безопасность" in result.summary


@pytest.mark.parametrize(
    ("label", "analyzer_calls"),
    # Nothing found anywhere: deterministic white (D-018). A found regulator
    # warning still goes to the analyzer so it is explained in context.
    [(None, 0), ("dm01_amiodarone_voriconazole.json", 1)],
)
def test_ddinter_missing_name_is_inconclusive_and_allows_only_white(
    tmp_path, label, analyzer_calls
):
    result, evidence, analyzer = _run(
        tmp_path,
        ("amiodarone", "voriconazole"),
        "Major",
        label=label,
        analyzer_level=InteractionLevel.WHITE,
        missing_name=True,
    )
    assert result.status is AnalysisStatus.INSUFFICIENT_DATA
    assert result.interaction_level is InteractionLevel.WHITE
    assert analyzer.calls == analyzer_calls
    ddinter_search = next(
        item for item in evidence.search_results if item.source_id.startswith("ddinter.")
    )
    assert ddinter_search.result is SearchResultStatus.INCONCLUSIVE
    assert evidence.search_coverage.complete is False
    assert evidence.deterministic_assessment.red_eligible is False
    assert evidence.deterministic_assessment.allowed_levels == [InteractionLevel.WHITE]
    for level in (
        InteractionLevel.GREEN,
        InteractionLevel.YELLOW,
        InteractionLevel.ORANGE,
        InteractionLevel.RED,
    ):
        with pytest.raises(ContractViolation, match="prohibited by deterministic evidence"):
            validate_analysis_against_evidence(evidence, _result(evidence, level))


def test_unsupported_model_red_for_nurofen_warfarin_is_clamped_offline(tmp_path):
    result, evidence, analyzer = _run(
        tmp_path,
        ("ibuprofen", "warfarin"),
        "Major",
        label="dm02_ibuprofen_warfarin.json",
        analyzer_level=InteractionLevel.RED,
    )
    assert result.interaction_level is InteractionLevel.ORANGE
    assert analyzer.calls == 1
    assert evidence.deterministic_assessment.red_eligible is False


def test_bare_pair_with_incomplete_coverage_answers_white_without_model(tmp_path):
    result, evidence, analyzer = _run(
        tmp_path, ("amiodarone", "voriconazole"), "Major", missing_name=True
    )
    assert analyzer.calls == 0
    assert result.status is AnalysisStatus.INSUFFICIENT_DATA
    assert result.interaction_level is InteractionLevel.WHITE
    assert "не доказывает безопасность" in result.summary
    assert any(source.source_id.startswith("ddinter.") for source in result.sources)
    assert "Не изменяйте лечение самостоятельно" in format_analysis_result(result)


@pytest.mark.parametrize(
    ("severity", "missing_name", "level"),
    [
        ("Minor", False, InteractionLevel.YELLOW),
        ("Unknown", False, InteractionLevel.WHITE),
        ("Major", True, InteractionLevel.WHITE),
    ],
)
def test_free_text_phrase_always_reaches_the_model(tmp_path, severity, missing_name, level):
    # A phrase may describe dangerous symptoms, which only YandexGPT turns into
    # status=urgent, so no deterministic shortcut may answer it (D-018).
    result, _, analyzer = _run(
        tmp_path,
        ("loratadine", "clarithromycin"),
        severity,
        analyzer_level=level,
        missing_name=missing_name,
        message="У меня обморок. Я принимаю Лоратадин, мне назначили Кларитромицин",
    )
    assert analyzer.calls == 1
    assert result.interaction_level is level
