"""Deterministic assembly and assessment of medical-source evidence."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from pharmacheck.contracts import (
    Confidence,
    DeterministicAssessment,
    DrugIdentity,
    EvidenceDocument,
    EvidencePackage,
    EvidenceState,
    FailedSource,
    IdentityStatus,
    InteractionLevel,
    RetrievalStatus,
    SearchCoverage,
    SearchResultStatus,
)
from pharmacheck.sources.base import InteractionSearchOutcome, InteractionSource

DEFAULT_REQUIRED_PROVIDERS = ("openfda.label", "ddinter")


class EvidenceAssemblyError(ValueError):
    """Raised when source outcomes cannot form a trustworthy evidence matrix."""


def collect_evidence_package(
    *,
    request_id: str,
    user_message: str,
    drugs: Sequence[DrugIdentity],
    sources: Mapping[str, InteractionSource],
    checked_at: datetime | None = None,
) -> EvidencePackage:
    """Query every provider for every distinct cross-product ingredient pair."""

    required_providers = tuple(sources)
    if len(required_providers) < 2:
        raise EvidenceAssemblyError("at least two independent providers are required")
    pairs = _expected_pairs(drugs)
    outcomes: list[InteractionSearchOutcome] = []
    for first, second in pairs:
        for provider_id, source in sources.items():
            outcome = source.search_pair(first, second, checked_at=checked_at)
            if outcome.provider_id != provider_id:
                raise EvidenceAssemblyError("source returned an unexpected provider_id")
            outcomes.append(outcome)
    return build_evidence_package(
        request_id=request_id,
        user_message=user_message,
        drugs=drugs,
        outcomes=outcomes,
        required_provider_ids=required_providers,
        checked_at=checked_at,
    )


def build_evidence_package(
    *,
    request_id: str,
    user_message: str,
    drugs: Sequence[DrugIdentity],
    outcomes: Sequence[InteractionSearchOutcome],
    required_provider_ids: Sequence[str] = DEFAULT_REQUIRED_PROVIDERS,
    checked_at: datetime | None = None,
) -> EvidencePackage:
    """Validate the provider/pair matrix and create a model-ready package."""

    checked = checked_at or datetime.now(UTC)
    drug_list = list(drugs)
    if len(drug_list) != 2:
        raise EvidenceAssemblyError("exactly two drugs are required")
    providers = tuple(required_provider_ids)
    if len(providers) < 2 or len(providers) != len(set(providers)):
        raise EvidenceAssemblyError("two or more unique providers are required")

    pairs = _expected_pairs(drug_list)
    expected_matrix = {
        (provider_id, _pair_key(*pair))
        for pair in pairs
        for provider_id in providers
    }
    actual_matrix: dict[
        tuple[str, tuple[str, str]], InteractionSearchOutcome
    ] = {}
    for outcome in outcomes:
        matrix_key = (
            outcome.provider_id,
            _pair_key(*outcome.ingredient_pair),
        )
        if matrix_key in actual_matrix:
            raise EvidenceAssemblyError("duplicate provider and ingredient pair outcome")
        actual_matrix[matrix_key] = outcome

    if set(actual_matrix) != expected_matrix:
        raise EvidenceAssemblyError("source outcome matrix is incomplete or unexpected")

    ordered_outcomes = [actual_matrix[key] for key in sorted(actual_matrix)]
    _validate_outcomes(ordered_outcomes)
    search_results = [outcome.search_result for outcome in ordered_outcomes]
    documents = [
        document
        for outcome in ordered_outcomes
        for document in outcome.documents
    ]
    _require_unique_ids(
        [result.source_id for result in search_results],
        "duplicate search result source_id",
    )
    _require_unique_ids(
        [document.source_id for document in documents],
        "duplicate evidence document source_id",
    )
    documents, related_labels_deduplicated = _deduplicate_related_label_documents(
        documents
    )

    identity_ready = _identity_ready(drug_list)
    failed_sources = _failed_sources(ordered_outcomes)
    coverage_complete = (
        identity_ready
        and not failed_sources
        and all(outcome.complete for outcome in ordered_outcomes)
    )
    assessment = _assess(
        drug_list,
        ordered_outcomes,
        documents,
        coverage_complete=coverage_complete,
        related_labels_deduplicated=related_labels_deduplicated,
    )
    source_ids = [result.source_id for result in search_results]
    query_pairs = [f"{first} + {second}" for first, second in pairs]
    return EvidencePackage(
        request_id=request_id,
        checked_at=checked,
        user_message=user_message,
        drugs=drug_list,
        search_coverage=SearchCoverage(
            required_sources=source_ids,
            searched_sources=source_ids,
            failed_sources=failed_sources,
            query_pairs=query_pairs,
            complete=coverage_complete,
        ),
        search_results=search_results,
        documents=documents,
        deterministic_assessment=assessment,
    )


def _expected_pairs(drugs: Sequence[DrugIdentity]) -> list[tuple[str, str]]:
    if len(drugs) != 2:
        raise EvidenceAssemblyError("exactly two drugs are required")
    if not drugs[0].active_ingredients or not drugs[1].active_ingredients:
        raise EvidenceAssemblyError("both drugs require active ingredients")

    pairs: dict[tuple[str, str], tuple[str, str]] = {}
    for first in drugs[0].active_ingredients:
        for second in drugs[1].active_ingredients:
            key = _pair_key(first, second)
            if key[0] == key[1]:
                continue
            pairs.setdefault(key, (first, second))
    if not pairs:
        raise EvidenceAssemblyError("no distinct cross-product ingredient pairs")
    return [pairs[key] for key in sorted(pairs)]


def _validate_outcomes(outcomes: Sequence[InteractionSearchOutcome]) -> None:
    for outcome in outcomes:
        status = outcome.search_result.result
        if outcome.failed_queries and outcome.complete:
            raise EvidenceAssemblyError("complete outcome contains failed queries")
        if status is SearchResultStatus.INTERACTION_FOUND:
            if not outcome.documents or not any(
                item.retrieval_status is RetrievalStatus.OK
                for item in outcome.documents
            ):
                raise EvidenceAssemblyError(
                    "interaction_found requires a successfully retrieved document"
                )
        elif status is SearchResultStatus.NO_RELEVANT_INTERACTION_FOUND:
            if not outcome.complete or outcome.documents:
                raise EvidenceAssemblyError(
                    "negative outcome must be complete and document-free"
                )
        elif outcome.complete:
            raise EvidenceAssemblyError(
                "failed or inconclusive outcome cannot be complete"
            )

        pair = set(_pair_key(*outcome.ingredient_pair))
        for document in outcome.documents:
            applies_to = {_normalize_name(item) for item in document.applies_to}
            if not pair.issubset(applies_to):
                raise EvidenceAssemblyError(
                    "evidence document does not apply to its ingredient pair"
                )


def _failed_sources(
    outcomes: Sequence[InteractionSearchOutcome],
) -> list[FailedSource]:
    failed: list[FailedSource] = []
    for outcome in outcomes:
        status = outcome.search_result.result
        if outcome.complete and status not in {
            SearchResultStatus.FAILED,
            SearchResultStatus.INCONCLUSIVE,
        }:
            continue
        reasons = [status.value, *outcome.failed_queries]
        failed.append(
            FailedSource(
                source=outcome.search_result.source_id,
                reason=", ".join(dict.fromkeys(reasons)),
            )
        )
    return failed


def _assess(
    drugs: Sequence[DrugIdentity],
    outcomes: Sequence[InteractionSearchOutcome],
    documents: Sequence[EvidenceDocument],
    *,
    coverage_complete: bool,
    related_labels_deduplicated: bool = False,
) -> DeterministicAssessment:
    positive = [
        outcome.search_result.source_id
        for outcome in outcomes
        if outcome.search_result.result is SearchResultStatus.INTERACTION_FOUND
    ]
    negative = [
        outcome.search_result.source_id
        for outcome in outcomes
        if outcome.search_result.result
        is SearchResultStatus.NO_RELEVANT_INTERACTION_FOUND
    ]
    incomplete = [
        outcome.search_result.source_id
        for outcome in outcomes
        if not outcome.complete
        or outcome.search_result.result
        in {SearchResultStatus.FAILED, SearchResultStatus.INCONCLUSIVE}
    ]
    identity_ready = _identity_ready(drugs)
    green_eligible = (
        identity_ready
        and coverage_complete
        and not positive
        and len(negative) == len(outcomes)
    )
    # An incomplete required search (a failed/inconclusive source, or one
    # that did not finish) must stop classification at white, even when
    # another source already found a positive document — a single channel's
    # finding is not sufficient on its own when we do not know what the
    # failed required source would have said (DM-06; system-prompt step 3).
    red_eligible = (
        coverage_complete and bool(positive) and _has_explicit_red_evidence(documents)
    )

    conflicts: list[str] = []
    limitations: list[str] = []
    if positive and negative:
        state = EvidenceState.MIXED_FINDINGS
        conflicts.append(
            "Один источник содержит взаимодействие, а другой не нашёл прямого "
            "упоминания; отрицательный поиск не опровергает положительное свидетельство."
        )
    elif positive:
        state = EvidenceState.INTERACTION_EVIDENCE
    elif green_eligible:
        state = EvidenceState.COMPLETE_NEGATIVE
    else:
        state = EvidenceState.INSUFFICIENT

    if incomplete:
        limitations.append("Часть обязательных запросов завершилась неполно.")
    if not identity_ready:
        limitations.append(
            "Два препарата не разрешены с высокой уверенностью по действующим веществам."
        )
    if negative:
        limitations.append(
            "Отсутствие найденной записи не доказывает безопасность комбинации."
        )
    if related_labels_deduplicated:
        limitations.append(
            "DailyMed и openFDA частично основаны на одной и той же официальной "
            "SPL-записи; совпадающие записи учтены один раз, а не как два "
            "независимых подтверждения."
        )

    if not coverage_complete:
        allowed = [InteractionLevel.WHITE]
    elif positive:
        allowed = [
            InteractionLevel.WHITE,
            InteractionLevel.YELLOW,
            InteractionLevel.ORANGE,
        ]
        if red_eligible:
            allowed.append(InteractionLevel.RED)
    elif green_eligible:
        allowed = [InteractionLevel.WHITE, InteractionLevel.GREEN]
    else:
        allowed = [InteractionLevel.WHITE]

    return DeterministicAssessment(
        state=state,
        green_eligible=green_eligible,
        red_eligible=red_eligible,
        allowed_levels=allowed,
        positive_source_ids=positive,
        negative_source_ids=negative,
        incomplete_source_ids=incomplete,
        conflicts=conflicts,
        limitations=limitations,
    )


def _has_explicit_red_evidence(documents: Sequence[EvidenceDocument]) -> bool:
    positive_patterns = (
        r"\bcontraindicat(?:ed|ion)\b",
        r"\bavoid (?:the )?(?:concomitant |combined )?use\b",
        r"\bshould be avoided\b",
        r"\bshould not be (?:used|administered)\b",
        r"\bdo not use\b",
        r"\bпротивопоказан\w*\b",
        r"\bизбегать совместн\w* применен\w*\b",
        r"\bне применять совместно\b",
    )
    negative_patterns = (
        r"\bnot contraindicated\b",
        r"\bno contraindications?\b",
        r"\bне противопоказан\w*\b",
        r"\bпротивопоказани\w* отсутству\w*\b",
    )
    for document in documents:
        if (
            document.retrieval_status is not RetrievalStatus.OK
            or document.source_tier > 2
            or document.source_type not in {"official_label", "regulator_label"}
        ):
            continue
        text = " ".join(document.excerpt.casefold().split())
        if any(re.search(pattern, text) for pattern in negative_patterns):
            continue
        if document.section in {"contraindications", "do_not_use"}:
            return True
        if _has_red_pattern_near_matched_term(document, text, positive_patterns):
            return True
    return False


def _has_red_pattern_near_matched_term(
    document: EvidenceDocument,
    text: str,
    positive_patterns: Sequence[str],
) -> bool:
    """Require avoidance language to refer to the matched pair or class.

    Official-label sections can contain several unrelated warnings. A phrase
    such as "should not be used with other ibuprofen products" must not make
    an ibuprofen + warfarin result red merely because warfarin appears later
    in the same long section.
    """

    anchors = [document.matched_term] if document.matched_term else document.applies_to
    anchor_spans = [
        span
        for anchor in anchors
        if anchor
        for span in _term_spans(text, anchor)
    ]
    if not anchor_spans:
        return False
    red_spans = [
        match.span()
        for pattern in positive_patterns
        for match in re.finditer(pattern, text)
    ]
    return any(
        _span_gap(red_span, anchor_span) <= 300
        for red_span in red_spans
        for anchor_span in anchor_spans
    )


def _term_spans(text: str, term: str) -> list[tuple[int, int]]:
    tokens = _normalize_name(term).split()
    if not tokens:
        return []
    flexible_tokens = []
    for token in tokens:
        stem = token[:-1] if token.endswith("s") and len(token) > 4 else token
        flexible_tokens.append(rf"{re.escape(stem)}s?")
    pattern = r"\b" + r"\s+".join(flexible_tokens) + r"\b"
    return [match.span() for match in re.finditer(pattern, text)]


def _span_gap(first: tuple[int, int], second: tuple[int, int]) -> int:
    return max(first[0] - second[1], second[0] - first[1], 0)


def _identity_ready(drugs: Sequence[DrugIdentity]) -> bool:
    return all(
        drug.identity_status is IdentityStatus.RESOLVED
        and drug.identity_confidence is Confidence.HIGH
        and bool(drug.active_ingredients)
        for drug in drugs
    )


def _deduplicate_related_label_documents(
    documents: Sequence[EvidenceDocument],
) -> tuple[list[EvidenceDocument], bool]:
    """Collapse DailyMed/openFDA documents that describe the same official
    SPL record into one, so the same underlying label is never presented as
    two independent confirmations (see D-011). Keeps the first occurrence in
    `documents` order, which is deterministic because it follows the sorted
    provider/pair matrix built by `build_evidence_package`.
    """
    kept: list[EvidenceDocument] = []
    seen_keys: set[tuple[str, str | None, tuple[str, ...]]] = set()
    deduplicated = False
    for document in documents:
        if not document.spl_set_id:
            kept.append(document)
            continue
        key = (
            _normalize_name(document.spl_set_id),
            document.section,
            tuple(sorted(_normalize_name(name) for name in document.applies_to)),
        )
        if key in seen_keys:
            deduplicated = True
            continue
        seen_keys.add(key)
        kept.append(document)
    return kept, deduplicated


def _pair_key(first: str, second: str) -> tuple[str, str]:
    return tuple(sorted((_normalize_name(first), _normalize_name(second))))


def _normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", value).split())


def _require_unique_ids(values: Sequence[str], message: str) -> None:
    if len(values) != len(set(values)):
        raise EvidenceAssemblyError(message)
