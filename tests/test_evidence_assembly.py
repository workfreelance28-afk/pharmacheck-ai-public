from __future__ import annotations

import re
from datetime import UTC, datetime

import pytest

from pharmacheck.contracts import (
    AnalysisResult,
    AnalysisStatus,
    Confidence,
    ContractViolation,
    DrugIdentity,
    EvidenceDocument,
    EvidenceState,
    IdentityStatus,
    InteractionLevel,
    RetrievalStatus,
    SearchResult,
    SearchResultStatus,
    validate_analysis_against_evidence,
)
from pharmacheck.evidence import EvidenceAssemblyError, build_evidence_package
from pharmacheck.sources.base import InteractionSearchOutcome

CHECKED_AT = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)


def _drug(
    name: str,
    ingredients: list[str],
    *,
    confidence: Confidence = Confidence.HIGH,
) -> DrugIdentity:
    return DrugIdentity(
        input_name=name,
        normalized_trade_name=name,
        active_ingredients=ingredients,
        identity_status=IdentityStatus.RESOLVED,
        identity_confidence=confidence,
        identity_source_ids=[f"catalog.{name.casefold()}"],
    )


def _safe_suffix(first: str, second: str) -> str:
    return re.sub(r"[^a-z0-9]+", ".", f"{first}.{second}".casefold()).strip(".")


def _outcome(
    provider_id: str,
    first: str,
    second: str,
    status: SearchResultStatus,
    *,
    complete: bool = True,
    excerpt: str | None = None,
    source_tier: int = 3,
    source_type: str = "interaction_database",
    source_severity: str | None = None,
    spl_set_id: str | None = None,
) -> InteractionSearchOutcome:
    suffix = _safe_suffix(first, second)
    search_id = f"{provider_id}.search.{suffix}"
    documents: list[EvidenceDocument] = []
    if excerpt is not None:
        documents.append(
            EvidenceDocument(
                source_id=f"{provider_id}.document.{suffix}",
                source_tier=source_tier,
                source_type=source_type,
                title=f"{provider_id}: {first} + {second}",
                publisher=provider_id,
                url=f"https://example.test/{provider_id}/{suffix}",
                accessed_at=CHECKED_AT,
                language="en",
                applies_to=[first, second],
                section="drug_interactions",
                source_severity=source_severity,
                spl_set_id=spl_set_id,
                excerpt=excerpt,
                retrieval_status=RetrievalStatus.OK,
            )
        )
    return InteractionSearchOutcome(
        provider_id=provider_id,
        ingredient_pair=(first, second),
        search_result=SearchResult(
            source_id=search_id,
            title=f"{provider_id} search",
            publisher=provider_id,
            url=f"https://example.test/{provider_id}",
            accessed_at=CHECKED_AT,
            query_pairs=[f"{first} + {second}"],
            searched_sections=["interactions"],
            result=status,
        ),
        documents=documents,
        complete=complete,
        failed_queries=[] if complete else ["technical_failure"],
    )


def _build(*outcomes: InteractionSearchOutcome, drugs=None):
    return build_evidence_package(
        request_id="request.stage5",
        user_message="Проверить два препарата",
        drugs=drugs or [_drug("Alpha", ["alpha"]), _drug("Beta", ["beta"])],
        outcomes=list(outcomes),
        checked_at=CHECKED_AT,
    )


def _negative_pair(first: str = "alpha", second: str = "beta"):
    return (
        _outcome(
            "openfda.label",
            first,
            second,
            SearchResultStatus.NO_RELEVANT_INTERACTION_FOUND,
        ),
        _outcome(
            "ddinter",
            first,
            second,
            SearchResultStatus.NO_RELEVANT_INTERACTION_FOUND,
        ),
    )


def test_complete_two_source_negative_is_green_eligible() -> None:
    evidence = _build(*_negative_pair())

    assessment = evidence.deterministic_assessment
    assert evidence.search_coverage.complete is True
    assert assessment is not None
    assert assessment.state is EvidenceState.COMPLETE_NEGATIVE
    assert assessment.green_eligible is True
    assert assessment.red_eligible is False
    assert assessment.allowed_levels == [
        InteractionLevel.WHITE,
        InteractionLevel.GREEN,
    ]


def test_positive_database_and_negative_label_are_mixed_without_red() -> None:
    evidence = _build(
        _outcome(
            "openfda.label",
            "alpha",
            "beta",
            SearchResultStatus.NO_RELEVANT_INTERACTION_FOUND,
        ),
        _outcome(
            "ddinter",
            "alpha",
            "beta",
            SearchResultStatus.INTERACTION_FOUND,
            excerpt="DDInter classifies this pair as Major.",
            source_severity="major",
        ),
    )

    assessment = evidence.deterministic_assessment
    assert assessment is not None
    assert assessment.state is EvidenceState.MIXED_FINDINGS
    assert assessment.green_eligible is False
    assert assessment.red_eligible is False
    assert InteractionLevel.ORANGE in assessment.allowed_levels
    assert InteractionLevel.RED not in assessment.allowed_levels
    assert assessment.conflicts


def test_explicit_regulator_contraindication_allows_red() -> None:
    evidence = _build(
        _outcome(
            "openfda.label",
            "alpha",
            "beta",
            SearchResultStatus.INTERACTION_FOUND,
            excerpt="Concomitant use with beta is contraindicated.",
            source_tier=2,
            source_type="regulator_label",
        ),
        _outcome(
            "ddinter",
            "alpha",
            "beta",
            SearchResultStatus.INTERACTION_FOUND,
            excerpt="DDInter classifies this pair as Major.",
            source_severity="major",
        ),
    )

    assessment = evidence.deterministic_assessment
    assert assessment is not None
    assert assessment.state is EvidenceState.INTERACTION_EVIDENCE
    assert assessment.red_eligible is True
    assert InteractionLevel.RED in assessment.allowed_levels


def test_structured_class_linked_avoidance_warning_allows_red() -> None:
    label = _outcome(
        "dailymed.label",
        "amiodarone",
        "voriconazole",
        SearchResultStatus.INTERACTION_FOUND,
        excerpt=(
            "Concomitant use of QT prolonging drugs including azole antifungals "
            "should be avoided."
        ),
        source_tier=2,
        source_type="regulator_label",
    )
    class_linked_document = label.documents[0].model_copy(
        update={
            "match_basis": "structured_class",
            "matched_term": "Azole Antifungal",
            "matched_ingredient": "voriconazole",
            "additional_urls": [
                "https://rxnav.nlm.nih.gov/REST/rxclass/class/byRxcui.json?"
                "rxcui=121243&relaSource=FDASPL"
            ],
        }
    )
    label = InteractionSearchOutcome(
        provider_id=label.provider_id,
        ingredient_pair=label.ingredient_pair,
        search_result=label.search_result,
        documents=[class_linked_document],
        complete=True,
    )
    evidence = build_evidence_package(
        request_id="request.class-linked",
        user_message="Амиодарон + Вориконазол",
        drugs=[
            _drug("Amiodarone", ["amiodarone"]),
            _drug("Voriconazole", ["voriconazole"]),
        ],
        outcomes=[
            label,
            _outcome(
                "ddinter",
                "amiodarone",
                "voriconazole",
                SearchResultStatus.NO_RELEVANT_INTERACTION_FOUND,
            ),
        ],
        required_provider_ids=("dailymed.label", "ddinter"),
        checked_at=CHECKED_AT,
    )

    assessment = evidence.deterministic_assessment
    assert assessment is not None
    assert assessment.state is EvidenceState.MIXED_FINDINGS
    assert assessment.red_eligible is True
    assert InteractionLevel.RED in assessment.allowed_levels


def test_negated_contraindication_does_not_allow_red() -> None:
    evidence = _build(
        _outcome(
            "openfda.label",
            "alpha",
            "beta",
            SearchResultStatus.INTERACTION_FOUND,
            excerpt="The combination is not contraindicated, but monitor the patient.",
            source_tier=2,
            source_type="regulator_label",
        ),
        _outcome(
            "ddinter",
            "alpha",
            "beta",
            SearchResultStatus.NO_RELEVANT_INTERACTION_FOUND,
        ),
    )

    assert evidence.deterministic_assessment is not None
    assert evidence.deterministic_assessment.red_eligible is False


def test_unrelated_avoidance_language_does_not_allow_red() -> None:
    label = _outcome(
        "openfda.label",
        "ibuprofen",
        "warfarin",
        SearchResultStatus.INTERACTION_FOUND,
        excerpt=(
            "Warfarin increases bleeding risk. "
            + "Background information. " * 40
            + "This ibuprofen product should not be used with other "
            "ibuprofen-containing products."
        ),
        source_tier=2,
        source_type="regulator_label",
    )
    label.documents[0] = label.documents[0].model_copy(
        update={"matched_term": "warfarin", "matched_ingredient": "warfarin"}
    )
    evidence = _build(
        label,
        _outcome(
            "ddinter",
            "ibuprofen",
            "warfarin",
            SearchResultStatus.INTERACTION_FOUND,
            excerpt="Ibuprofen and warfarin increase bleeding risk.",
        ),
        drugs=[_drug("Ibuprofen", ["ibuprofen"]), _drug("Warfarin", ["warfarin"])],
    )

    assessment = evidence.deterministic_assessment
    assert assessment is not None
    assert assessment.red_eligible is False
    assert assessment.allowed_levels == [
        InteractionLevel.WHITE,
        InteractionLevel.YELLOW,
        InteractionLevel.ORANGE,
    ]


def test_official_contraindications_section_allows_red_candidate() -> None:
    label = _outcome(
        "openfda.label",
        "alpha",
        "beta",
        SearchResultStatus.INTERACTION_FOUND,
        excerpt="Concomitant administration with beta.",
        source_tier=2,
        source_type="regulator_label",
    )
    label.documents[0].section = "contraindications"
    evidence = _build(label, _negative_pair()[1])

    assert evidence.deterministic_assessment is not None
    assert evidence.deterministic_assessment.red_eligible is True


def test_failed_required_source_allows_only_white() -> None:
    evidence = _build(
        _outcome(
            "openfda.label",
            "alpha",
            "beta",
            SearchResultStatus.FAILED,
            complete=False,
        ),
        _outcome(
            "ddinter",
            "alpha",
            "beta",
            SearchResultStatus.NO_RELEVANT_INTERACTION_FOUND,
        ),
    )

    assessment = evidence.deterministic_assessment
    assert evidence.search_coverage.complete is False
    assert len(evidence.search_coverage.failed_sources) == 1
    assert assessment is not None
    assert assessment.state is EvidenceState.INSUFFICIENT
    assert assessment.allowed_levels == [InteractionLevel.WHITE]


def test_positive_result_with_incomplete_coverage_stays_white() -> None:
    # DM-06: DailyMed finds a positive regulator-label document, but the
    # other required source (openFDA) times out. A single channel's
    # finding must not be promoted to a colored level while we still don't
    # know what the failed required source would have said.
    evidence = build_evidence_package(
        request_id="request.dm06",
        user_message="Нурофен + Варфарин",
        drugs=[_drug("Alpha", ["alpha"]), _drug("Beta", ["beta"])],
        outcomes=[
            _outcome(
                "dailymed.label",
                "alpha",
                "beta",
                SearchResultStatus.INTERACTION_FOUND,
                excerpt="Avoid concomitant use with beta.",
                source_tier=2,
                source_type="regulator_label",
            ),
            _outcome(
                "openfda.label",
                "alpha",
                "beta",
                SearchResultStatus.FAILED,
                complete=False,
            ),
        ],
        required_provider_ids=("dailymed.label", "openfda.label"),
        checked_at=CHECKED_AT,
    )

    assessment = evidence.deterministic_assessment
    assert evidence.search_coverage.complete is False
    assert len(evidence.search_coverage.failed_sources) == 1
    assert assessment is not None
    assert assessment.red_eligible is False
    assert assessment.allowed_levels == [InteractionLevel.WHITE]
    assert "Часть обязательных запросов завершилась неполно." in assessment.limitations


@pytest.mark.parametrize(
    "level", [InteractionLevel.YELLOW, InteractionLevel.ORANGE, InteractionLevel.RED]
)
def test_colored_result_is_rejected_when_required_search_is_incomplete(level) -> None:
    # Same DM-06 evidence package as above, but this proves the schema-level
    # enforcement path: even an explicit-avoid-worded document cannot make
    # yellow/orange/red pass validate_analysis_against_evidence while the
    # required search is incomplete.
    evidence = build_evidence_package(
        request_id="request.dm06-reject",
        user_message="Нурофен + Варфарин",
        drugs=[_drug("Alpha", ["alpha"]), _drug("Beta", ["beta"])],
        outcomes=[
            _outcome(
                "dailymed.label",
                "alpha",
                "beta",
                SearchResultStatus.INTERACTION_FOUND,
                excerpt="Avoid concomitant use with beta.",
                source_tier=2,
                source_type="regulator_label",
            ),
            _outcome(
                "openfda.label",
                "alpha",
                "beta",
                SearchResultStatus.FAILED,
                complete=False,
            ),
        ],
        required_provider_ids=("dailymed.label", "openfda.label"),
        checked_at=CHECKED_AT,
    )
    document = evidence.documents[0]
    result = _analysis_result(level, document)

    with pytest.raises(ContractViolation, match="prohibited by deterministic evidence"):
        validate_analysis_against_evidence(evidence, result)


def test_low_identity_confidence_prevents_green() -> None:
    drugs = [
        _drug("Alpha", ["alpha"], confidence=Confidence.MEDIUM),
        _drug("Beta", ["beta"]),
    ]
    evidence = _build(*_negative_pair(), drugs=drugs)

    assert evidence.search_coverage.complete is False
    assert evidence.deterministic_assessment is not None
    assert evidence.deterministic_assessment.allowed_levels == [
        InteractionLevel.WHITE
    ]


def test_missing_provider_pair_is_rejected() -> None:
    with pytest.raises(EvidenceAssemblyError, match="matrix"):
        _build(_negative_pair()[0])


def test_interaction_without_document_is_rejected() -> None:
    with pytest.raises(EvidenceAssemblyError, match="document"):
        _build(
            _outcome(
                "openfda.label",
                "alpha",
                "beta",
                SearchResultStatus.INTERACTION_FOUND,
            ),
            _negative_pair()[1],
        )


def test_document_for_different_pair_is_rejected() -> None:
    outcome = _outcome(
        "openfda.label",
        "alpha",
        "beta",
        SearchResultStatus.INTERACTION_FOUND,
        excerpt="A possible interaction is described.",
        source_tier=2,
        source_type="regulator_label",
    )
    outcome.documents[0].applies_to = ["alpha", "gamma"]

    with pytest.raises(EvidenceAssemblyError, match="does not apply"):
        _build(outcome, _negative_pair()[1])


def test_all_cross_product_pairs_are_required() -> None:
    drugs = [
        _drug("Combo", ["alpha", "gamma"]),
        _drug("Other", ["beta", "delta"]),
    ]
    outcomes = [
        outcome
        for pair in (
            ("alpha", "beta"),
            ("alpha", "delta"),
            ("gamma", "beta"),
            ("gamma", "delta"),
        )
        for outcome in _negative_pair(*pair)
    ]

    evidence = _build(*outcomes, drugs=drugs)

    assert evidence.search_coverage.complete is True
    assert len(evidence.search_results) == 8
    assert len(evidence.search_coverage.query_pairs) == 4
    assert evidence.deterministic_assessment is not None
    assert evidence.deterministic_assessment.green_eligible is True


def test_post_validation_rejects_red_based_only_on_ddinter_major() -> None:
    evidence = _build(
        _outcome(
            "openfda.label",
            "alpha",
            "beta",
            SearchResultStatus.NO_RELEVANT_INTERACTION_FOUND,
        ),
        _outcome(
            "ddinter",
            "alpha",
            "beta",
            SearchResultStatus.INTERACTION_FOUND,
            excerpt="DDInter classifies this pair as Major.",
            source_severity="major",
        ),
    )
    document = evidence.documents[0]
    result = _analysis_result(InteractionLevel.RED, document)

    with pytest.raises(ContractViolation, match="deterministic"):
        validate_analysis_against_evidence(evidence, result)


def test_post_validation_accepts_red_with_explicit_regulator_text() -> None:
    evidence = _build(
        _outcome(
            "openfda.label",
            "alpha",
            "beta",
            SearchResultStatus.INTERACTION_FOUND,
            excerpt="Avoid concomitant use with beta.",
            source_tier=2,
            source_type="regulator_label",
        ),
        _outcome(
            "ddinter",
            "alpha",
            "beta",
            SearchResultStatus.NO_RELEVANT_INTERACTION_FOUND,
        ),
    )
    document = evidence.documents[0]
    result = _analysis_result(InteractionLevel.RED, document)

    validate_analysis_against_evidence(evidence, result)


def test_dailymed_and_openfda_same_spl_record_is_not_double_counted() -> None:
    evidence = build_evidence_package(
        request_id="request.dedup",
        user_message="Проверить два препарата",
        drugs=[_drug("Alpha", ["alpha"]), _drug("Beta", ["beta"])],
        outcomes=[
            _outcome(
                "dailymed.label",
                "alpha",
                "beta",
                SearchResultStatus.INTERACTION_FOUND,
                excerpt="Avoid concomitant use with beta.",
                source_tier=2,
                source_type="regulator_label",
                spl_set_id="3ba0a9f2-062a-401e-82eb-54383a822366",
            ),
            _outcome(
                "openfda.label",
                "alpha",
                "beta",
                SearchResultStatus.INTERACTION_FOUND,
                excerpt="Avoid concomitant use with beta.",
                source_tier=2,
                source_type="regulator_label",
                spl_set_id="3BA0A9F2-062A-401E-82EB-54383A822366",
            ),
        ],
        required_provider_ids=("dailymed.label", "openfda.label"),
        checked_at=CHECKED_AT,
    )

    assert len(evidence.documents) == 1
    assert evidence.documents[0].source_id.startswith("dailymed.")
    assessment = evidence.deterministic_assessment
    assert assessment is not None
    assert any(
        "одной и той же официальной" in limitation
        for limitation in assessment.limitations
    )


def test_dailymed_and_openfda_different_spl_records_are_both_kept() -> None:
    evidence = build_evidence_package(
        request_id="request.no-dedup",
        user_message="Проверить два препарата",
        drugs=[_drug("Alpha", ["alpha"]), _drug("Beta", ["beta"])],
        outcomes=[
            _outcome(
                "dailymed.label",
                "alpha",
                "beta",
                SearchResultStatus.INTERACTION_FOUND,
                excerpt="Avoid concomitant use with beta.",
                source_tier=2,
                source_type="regulator_label",
                spl_set_id="11111111-1111-1111-1111-111111111111",
            ),
            _outcome(
                "openfda.label",
                "alpha",
                "beta",
                SearchResultStatus.INTERACTION_FOUND,
                excerpt="Avoid concomitant use with beta.",
                source_tier=2,
                source_type="regulator_label",
                spl_set_id="22222222-2222-2222-2222-222222222222",
            ),
        ],
        required_provider_ids=("dailymed.label", "openfda.label"),
        checked_at=CHECKED_AT,
    )

    assert len(evidence.documents) == 2
    assessment = evidence.deterministic_assessment
    assert assessment is not None
    assert not any(
        "одной и той же официальной" in limitation
        for limitation in assessment.limitations
    )


def _analysis_result(
    level: InteractionLevel,
    document: EvidenceDocument,
) -> AnalysisResult:
    return AnalysisResult(
        status=AnalysisStatus.ANSWER,
        interaction_level=level,
        headline="Тестовый вывод",
        identified_drugs=[
            {"trade_name": "Alpha", "active_ingredients": ["alpha"]},
            {"trade_name": "Beta", "active_ingredients": ["beta"]},
        ],
        summary="Вывод основан на переданных источниках.",
        mechanism_or_effect=None,
        recommended_user_action="Обсудите комбинацию с врачом.",
        limitations=["Это контрольный сценарий."],
        conflicts=[],
        missing_information=[],
        sources=[
            {
                "source_id": document.source_id,
                "title": document.title,
                "url": str(document.url),
                "supports": "Подтверждает тестовый уровень.",
            }
        ],
    )
