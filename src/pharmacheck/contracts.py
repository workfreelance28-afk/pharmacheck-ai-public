"""Strict input and output contracts for the PharmaCheck AI analysis stage."""

from __future__ import annotations

import re
from datetime import date, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, UrlConstraints, model_validator

# `src/pharmacheck/telegram.py` guarantees that whenever `AnalysisResult.sources`
# is non-empty, the first source's complete, un-clipped URL is always shown
# (never dropped, never partially cut), by dynamically shrinking every other
# mandatory field — headline, drug names, ingredients, safe action text, and
# the source title — down to a documented floor. That guarantee only holds if
# the URL itself cannot exceed the room those floors leave for it, so the
# bound is enforced here, at the contract, rather than left to the formatter.
#
# Worst-case fixed overhead at the formatter's floor scale (icon+space,
# headline at its 30-char floor, two identified drugs at their 30-char name /
# 20-char ingredient floors with all 5 shown ingredient slots and an
# "и ещё 999999" omitted-count suffix, the safe-action text at its 30-char
# floor, and "Источники:\n1. " plus the source title at its 20-char floor):
# 4096 - 439 = 3657 characters. 3600 keeps a deliberate ~57-character margin
# below that exact figure for future formatter changes.
MAX_SOURCE_URL_LENGTH = 3600
BoundedHttpUrl = Annotated[AnyHttpUrl, UrlConstraints(max_length=MAX_SOURCE_URL_LENGTH)]


class StrictModel(BaseModel):
    """Base model that rejects unknown fields instead of silently ignoring them."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class IdentityStatus(StrEnum):
    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    UNRESOLVED = "unresolved"


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SearchResultStatus(StrEnum):
    INTERACTION_FOUND = "interaction_found"
    NO_RELEVANT_INTERACTION_FOUND = "no_relevant_interaction_found"
    INCONCLUSIVE = "inconclusive"
    FAILED = "failed"


class RetrievalStatus(StrEnum):
    OK = "ok"
    PARTIAL = "partial"
    FAILED = "failed"


class AnalysisStatus(StrEnum):
    ANSWER = "answer"
    CLARIFICATION_NEEDED = "clarification_needed"
    INSUFFICIENT_DATA = "insufficient_data"
    URGENT = "urgent"


class InteractionLevel(StrEnum):
    GREEN = "green"
    YELLOW = "yellow"
    ORANGE = "orange"
    RED = "red"
    WHITE = "white"


NonEmptyText = Annotated[str, Field(min_length=1)]
SourceId = Annotated[str, Field(min_length=1, pattern=r"^[A-Za-z0-9_.:-]+$")]


class DrugIdentity(StrictModel):
    input_name: NonEmptyText
    normalized_trade_name: str | None = None
    dosage_form: str | None = None
    strength: str | None = None
    active_ingredients: list[NonEmptyText] = Field(default_factory=list)
    identity_status: IdentityStatus
    identity_confidence: Confidence
    identity_source_ids: list[SourceId] = Field(default_factory=list)

    @model_validator(mode="after")
    def resolved_drug_has_identity(self) -> DrugIdentity:
        if self.identity_status is IdentityStatus.RESOLVED:
            if not self.normalized_trade_name or not self.active_ingredients:
                raise ValueError(
                    "resolved drug requires normalized_trade_name and active_ingredients"
                )
        return self


class FailedSource(StrictModel):
    source: SourceId
    reason: NonEmptyText


class SearchCoverage(StrictModel):
    required_sources: list[SourceId] = Field(min_length=1)
    searched_sources: list[SourceId] = Field(default_factory=list)
    failed_sources: list[FailedSource] = Field(default_factory=list)
    query_pairs: list[NonEmptyText] = Field(default_factory=list)
    complete: bool


class SourceMetadata(StrictModel):
    source_id: SourceId
    title: NonEmptyText
    publisher: NonEmptyText
    url: BoundedHttpUrl
    license_name: str | None = None
    license_url: AnyHttpUrl | None = None
    document_date: date | None = None
    accessed_at: datetime


class SearchResult(SourceMetadata):
    query_pairs: list[NonEmptyText] = Field(min_length=1)
    searched_sections: list[NonEmptyText] = Field(min_length=1)
    additional_urls: list[AnyHttpUrl] = Field(default_factory=list, max_length=4)
    result: SearchResultStatus
    details: str | None = None


class EvidenceDocument(SourceMetadata):
    source_tier: Annotated[int, Field(ge=1, le=3)]
    source_type: Literal[
        "official_label", "regulator_label", "interaction_database"
    ]
    language: NonEmptyText
    applies_to: list[NonEmptyText] = Field(min_length=1)
    section: str | None = None
    source_severity: str | None = None
    spl_set_id: str | None = None
    excerpt: NonEmptyText
    retrieval_status: RetrievalStatus
    match_basis: Literal["direct_ingredient", "structured_class"] = (
        "direct_ingredient"
    )
    matched_term: str | None = None
    matched_ingredient: str | None = None
    additional_urls: list[AnyHttpUrl] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def class_match_has_provenance(self) -> EvidenceDocument:
        if self.match_basis == "structured_class" and (
            not self.matched_term or not self.matched_ingredient or not self.additional_urls
        ):
            raise ValueError(
                "structured class match requires term, ingredient and provenance URL"
            )
        return self


class EvidenceState(StrEnum):
    COMPLETE_NEGATIVE = "complete_negative"
    INTERACTION_EVIDENCE = "interaction_evidence"
    MIXED_FINDINGS = "mixed_findings"
    INSUFFICIENT = "insufficient"


class DeterministicAssessment(StrictModel):
    state: EvidenceState
    green_eligible: bool
    red_eligible: bool
    allowed_levels: list[InteractionLevel] = Field(min_length=1)
    positive_source_ids: list[SourceId] = Field(default_factory=list)
    negative_source_ids: list[SourceId] = Field(default_factory=list)
    incomplete_source_ids: list[SourceId] = Field(default_factory=list)
    conflicts: list[NonEmptyText] = Field(default_factory=list)
    limitations: list[NonEmptyText] = Field(default_factory=list)

    @model_validator(mode="after")
    def eligibility_matches_allowed_levels(self) -> DeterministicAssessment:
        if len(self.allowed_levels) != len(set(self.allowed_levels)):
            raise ValueError("allowed_levels values must be unique")
        if self.green_eligible != (InteractionLevel.GREEN in self.allowed_levels):
            raise ValueError("green eligibility and allowed levels disagree")
        if self.red_eligible != (InteractionLevel.RED in self.allowed_levels):
            raise ValueError("red eligibility and allowed levels disagree")
        if set(self.positive_source_ids).intersection(self.negative_source_ids):
            raise ValueError("positive and negative source groups must not overlap")
        return self


class EvidencePackage(StrictModel):
    request_id: SourceId
    checked_at: datetime
    locale: Literal["ru-RU"] = "ru-RU"
    user_message: NonEmptyText
    drugs: list[DrugIdentity] = Field(min_length=2, max_length=2)
    search_coverage: SearchCoverage
    search_results: list[SearchResult] = Field(default_factory=list)
    documents: list[EvidenceDocument] = Field(default_factory=list)
    deterministic_assessment: DeterministicAssessment | None = None

    @model_validator(mode="after")
    def source_ids_are_consistent(self) -> EvidencePackage:
        search_ids = [item.source_id for item in self.search_results]
        if len(search_ids) != len(set(search_ids)):
            raise ValueError("search_results source_id values must be unique")

        known_ids = set(search_ids) | {item.source_id for item in self.documents}
        required_ids = set(self.search_coverage.required_sources)
        searched_ids = set(self.search_coverage.searched_sources)

        if not searched_ids.issubset(known_ids):
            raise ValueError("searched_sources contains a source absent from evidence")
        if self.search_coverage.complete and not required_ids.issubset(searched_ids):
            raise ValueError("complete coverage requires every required source to be searched")
        if self.deterministic_assessment is not None:
            assessment_ids = {
                *self.deterministic_assessment.positive_source_ids,
                *self.deterministic_assessment.negative_source_ids,
                *self.deterministic_assessment.incomplete_source_ids,
            }
            if not assessment_ids.issubset(set(search_ids)):
                raise ValueError("assessment references unknown search results")
        return self


class IdentifiedDrug(StrictModel):
    trade_name: NonEmptyText
    active_ingredients: list[NonEmptyText] = Field(min_length=1)


class ResultSource(StrictModel):
    source_id: SourceId
    title: NonEmptyText
    url: BoundedHttpUrl
    supports: NonEmptyText


class AnalysisResult(StrictModel):
    status: AnalysisStatus
    interaction_level: InteractionLevel
    headline: NonEmptyText
    identified_drugs: list[IdentifiedDrug] = Field(max_length=2)
    summary: NonEmptyText
    mechanism_or_effect: str | None
    recommended_user_action: NonEmptyText
    limitations: list[NonEmptyText]
    conflicts: list[NonEmptyText]
    missing_information: list[NonEmptyText]
    sources: list[ResultSource]

    @model_validator(mode="after")
    def status_matches_level(self) -> AnalysisResult:
        if self.status in {
            AnalysisStatus.CLARIFICATION_NEEDED,
            AnalysisStatus.INSUFFICIENT_DATA,
            AnalysisStatus.URGENT,
        } and self.interaction_level is not InteractionLevel.WHITE:
            raise ValueError(
                "clarification, insufficient_data, and urgent all require white level"
            )
        if self.status is AnalysisStatus.ANSWER and self.interaction_level is InteractionLevel.WHITE:
            raise ValueError("answer status cannot use white level")
        if self.status in {
            AnalysisStatus.CLARIFICATION_NEEDED,
            AnalysisStatus.INSUFFICIENT_DATA,
        } and not self.missing_information:
            raise ValueError("missing_information is required when no conclusion is possible")
        return self


class ContractViolation(ValueError):
    """Raised when a structurally valid result is unsafe for its evidence package."""


def validate_analysis_against_evidence(
    evidence: EvidencePackage,
    result: AnalysisResult,
) -> None:
    """Apply deterministic safety checks after schema validation."""

    errors: list[str] = []
    metadata = {
        item.source_id: (item.title, str(item.url))
        for item in [*evidence.search_results, *evidence.documents]
    }

    for source in result.sources:
        expected = metadata.get(source.source_id)
        actual = (source.title, str(source.url))
        if expected is None:
            errors.append(f"unknown result source_id: {source.source_id}")
        elif expected != actual:
            errors.append(f"source metadata changed for: {source.source_id}")

    if result.status is AnalysisStatus.ANSWER and not result.sources:
        errors.append("medical answer requires at least one cited source")

    assessment = evidence.deterministic_assessment
    if assessment is not None and result.interaction_level not in assessment.allowed_levels:
        errors.append(
            "interaction level is prohibited by deterministic evidence assessment"
        )

    if result.interaction_level is InteractionLevel.GREEN:
        _validate_green(evidence, result, errors)

    if result.interaction_level in {
        InteractionLevel.YELLOW,
        InteractionLevel.ORANGE,
        InteractionLevel.RED,
    }:
        cited_ids = {source.source_id for source in result.sources}
        document_ids = {
            document.source_id
            for document in evidence.documents
            if document.retrieval_status is RetrievalStatus.OK
        }
        if not cited_ids.intersection(document_ids):
            errors.append("non-green medical level requires a successfully retrieved document")

    free_text = " ".join(
        filter(
            None,
            [
                result.headline,
                result.summary,
                result.mechanism_or_effect,
                result.recommended_user_action,
                *result.limitations,
                *result.conflicts,
                *result.missing_information,
            ],
        )
    )
    if re.search(r"https?://", free_text, flags=re.IGNORECASE):
        errors.append("URLs are allowed only inside the sources array")

    if _contains_unsafe_treatment_instruction(free_text):
        errors.append("result recommends a self-directed treatment change")

    if errors:
        raise ContractViolation("; ".join(errors))


def attach_search_audit_urls(
    evidence: EvidencePackage,
    result: AnalysisResult,
) -> AnalysisResult:
    """Add verified secondary search URLs without trusting model-generated links."""
    additional_by_source = {
        item.source_id: [str(url) for url in item.additional_urls]
        for item in [*evidence.search_results, *evidence.documents]
        if item.additional_urls
    }
    sources = []
    for source in result.sources:
        urls = additional_by_source.get(source.source_id, [])
        if not urls:
            sources.append(source)
            continue
        audit_line = "Дополнительный поиск источника: " + ", ".join(urls)
        sources.append(
            source.model_copy(update={"supports": f"{audit_line}\n{source.supports}"})
        )
    return result.model_copy(update={"sources": sources})


def replace_unsafe_recommended_action(result: AnalysisResult) -> AnalysisResult:
    if result.status is AnalysisStatus.URGENT:
        safe_action = (
            "Немедленно свяжитесь с местной экстренной службой. "
            "Не изменяйте лечение самостоятельно."
        )
    elif result.interaction_level is InteractionLevel.RED:
        safe_action = (
            "Как можно скорее свяжитесь с назначившим врачом или фармацевтом. "
            "Не изменяйте лечение самостоятельно. При опасных симптомах "
            "обратитесь за неотложной помощью."
        )
    else:
        safe_action = (
            "Обсудите результат с врачом или фармацевтом. "
            "Не изменяйте лечение самостоятельно."
        )
    return result.model_copy(update={"recommended_user_action": safe_action})


def _contains_unsafe_treatment_instruction(value: str) -> bool:
    text = " ".join(value.casefold().replace("ё", "е").split())
    patterns = (
        r"\b(?:немедленно\s+)?прекратите\s+"
        r"(?:(?:совместн|одновременн)\w*\s+)?(?:прием|принимать)\b",
        r"\bотмените\s+(?:один\s+из\s+)?(?:препарат|лекарств|лечение)\w*\b",
        r"\bперестаньте\s+принимать\b",
        r"\bне\s+принимайте\s+(?:этот\s+)?(?:препарат|лекарств)\w*\b",
        r"\bзамените\s+(?:препарат|лекарств)\w*\b",
        r"\b(?:измените|снизьте|уменьшите|увеличьте|повыс[ь]те)\s+доз\w*\b",
        r"\bначните\s+принимать\b",
        r"\b(?:immediately\s+)?stop\s+taking\b",
        r"\bdiscontinue\s+(?:the\s+)?(?:drug|medication|treatment)\b",
        r"\bdo\s+not\s+take\s+(?:this\s+)?(?:drug|medication)\b",
        r"\b(?:start|switch|replace)\s+(?:taking\s+)?(?:the\s+)?(?:drug|medication)\b",
        r"\b(?:change|reduce|decrease|increase)\s+(?:the\s+)?dose\b",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def _validate_green(
    evidence: EvidencePackage,
    result: AnalysisResult,
    errors: list[str],
) -> None:
    if result.status is not AnalysisStatus.ANSWER:
        errors.append("green level requires answer status")

    for drug in evidence.drugs:
        if (
            drug.identity_status is not IdentityStatus.RESOLVED
            or drug.identity_confidence is not Confidence.HIGH
            or not drug.active_ingredients
        ):
            errors.append("green level requires two confidently resolved drugs")
            break

    coverage = evidence.search_coverage
    if not coverage.complete or coverage.failed_sources:
        errors.append("green level requires complete coverage without failed sources")

    results = {item.source_id: item.result for item in evidence.search_results}
    for source_id in coverage.required_sources:
        if results.get(source_id) is not SearchResultStatus.NO_RELEVANT_INTERACTION_FOUND:
            errors.append(f"green level lacks a clean negative search result for {source_id}")

    required_warning = (
        "не доказывает безопасность комбинации для конкретного человека"
    )
    if required_warning.casefold() not in result.summary.casefold():
        errors.append("green result is missing the mandatory safety limitation")
