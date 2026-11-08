"""End-to-end application service joining stages 2 through 6."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol

from pharmacheck.contracts import (
    AnalysisResult,
    AnalysisStatus,
    DrugIdentity,
    EvidencePackage,
    IdentifiedDrug,
    InteractionLevel,
    ResultSource,
    RetrievalStatus,
    SearchResultStatus,
    attach_search_audit_urls,
    validate_analysis_against_evidence,
)
from pharmacheck.drug_recognition import (
    DrugCatalog,
    DrugResolution,
    PairParseStatus,
    PairResolutionStatus,
    parse_drug_pair,
)
from pharmacheck.evidence import collect_evidence_package
from pharmacheck.sources.base import InteractionSource
from pharmacheck.yandexgpt import YandexGPTAnalysis


class EvidenceAnalyzer(Protocol):
    def analyze(self, evidence: EvidencePackage) -> YandexGPTAnalysis: ...


PAIR_ONLY_NEGATIVE_PROVIDERS = frozenset(
    {"dailymed.label", "openfda.label", "ddinter"}
)


class PharmaCheckService:
    """Resolve drugs, collect evidence and invoke the validated model client."""

    def __init__(
        self,
        *,
        catalog: DrugCatalog,
        sources: Mapping[str, InteractionSource],
        analyzer: EvidenceAnalyzer,
        budget_check: Callable[[str], None] | None = None,
    ) -> None:
        if len(sources) < 2:
            raise ValueError("at least two medical sources are required")
        self._catalog = catalog
        self._sources = dict(sources)
        self._analyzer = analyzer
        self._budget_check = budget_check or (lambda stage: None)

    def check(self, message: str, *, request_id: str) -> AnalysisResult:
        parsed = parse_drug_pair(message)
        if parsed.status is not PairParseStatus.PARSED:
            return _clarification_result(parsed.clarification or "Укажите два препарата.")

        resolved = self._catalog.resolve_pair(parsed)
        if resolved.status is PairResolutionStatus.CLARIFICATION_NEEDED:
            return _resolution_clarification(resolved.drugs, resolved.clarification)

        identities = [_to_identity(resolution) for resolution in resolved.drugs]
        self._budget_check("medical_sources")
        evidence = collect_evidence_package(
            request_id=request_id,
            user_message=message,
            drugs=identities,
            sources=self._sources,
        )
        # D-018: only YandexGPT recognizes dangerous symptoms (status=urgent),
        # so the no-model shortcuts apply only to a bare pair ("A + B"), never
        # to a free-text phrase that may describe symptoms.
        deterministic_result = None
        if not parsed.free_text:
            deterministic_result = _pair_only_negative_result(evidence)
            if deterministic_result is None:
                deterministic_result = _unknown_database_severity_result(evidence)
            if deterministic_result is None:
                deterministic_result = _known_database_severity_result(evidence)
            if deterministic_result is None:
                deterministic_result = _incomplete_coverage_result(evidence)
        if deterministic_result is not None:
            return attach_search_audit_urls(evidence, deterministic_result)
        self._budget_check("yandexgpt")
        return attach_search_audit_urls(evidence, self._analyzer.analyze(evidence).result)


def _to_identity(resolution: DrugResolution) -> DrugIdentity:
    return DrugIdentity(
        input_name=resolution.mention.raw_text,
        normalized_trade_name=resolution.normalized_trade_name,
        dosage_form=resolution.dosage_form,
        strength=resolution.strength,
        active_ingredients=resolution.active_ingredients,
        identity_status=resolution.identity_status,
        identity_confidence=resolution.identity_confidence,
        identity_source_ids=resolution.identity_source_ids,
    )


def _resolution_clarification(
    resolutions: list[DrugResolution],
    clarification: str | None,
) -> AnalysisResult:
    identified = []
    missing = []
    for resolution in resolutions:
        if resolution.normalized_trade_name and resolution.active_ingredients:
            identified.append(
                IdentifiedDrug(
                    trade_name=resolution.normalized_trade_name,
                    active_ingredients=resolution.active_ingredients,
                )
            )
        if resolution.clarification:
            detail = resolution.clarification
            if resolution.candidates:
                candidates = ", ".join(
                    _candidate_label(candidate.trade_name, candidate.dosage_form, candidate.strength)
                    for candidate in resolution.candidates
                )
                detail = f"{detail} Возможные варианты: {candidates}."
            missing.append(detail)
    if clarification and clarification not in missing:
        missing.append(clarification)
    return _clarification_result(
        " ".join(dict.fromkeys(missing)) or "Уточните названия препаратов.",
        identified=identified,
    )


def _clarification_result(
    message: str,
    *,
    identified: list[IdentifiedDrug] | None = None,
) -> AnalysisResult:
    return AnalysisResult(
        status=AnalysisStatus.CLARIFICATION_NEEDED,
        interaction_level=InteractionLevel.WHITE,
        headline="Нужно уточнить препараты",
        identified_drugs=identified or [],
        summary="Медицинский вывод пока не сформирован: препараты не определены однозначно.",
        mechanism_or_effect=None,
        recommended_user_action=(
            "Пришлите уточнённые названия, лекарственную форму и дозировку. "
            "Не изменяйте лечение самостоятельно."
        ),
        limitations=["Поиск взаимодействий и вызов YandexGPT не выполнялись."],
        conflicts=[],
        missing_information=[message],
        sources=[],
    )


def _unknown_database_severity_result(
    evidence: EvidencePackage,
) -> AnalysisResult | None:
    """Stop before the model when database evidence has no classifiable severity."""
    assessment = evidence.deterministic_assessment
    documents = [
        document
        for document in evidence.documents
        if document.retrieval_status is RetrievalStatus.OK
    ]
    if (
        assessment is None
        or not assessment.positive_source_ids
        or not documents
        or any(
            document.source_type != "interaction_database"
            or (document.source_severity or "").strip().casefold() != "unknown"
            for document in documents
        )
    ):
        return None

    cited_sources = [
        ResultSource(
            source_id=document.source_id,
            title=document.title,
            url=document.url,
            supports=(
                "Источник содержит запись о паре препаратов, но указывает "
                "неопределённый уровень взаимодействия."
            ),
        )
        for document in documents
    ]
    cited_sources.extend(
        ResultSource(
            source_id=result.source_id,
            title=result.title,
            url=result.url,
            supports=(
                "В проверенных разделах прямое упоминание взаимодействия не найдено."
            ),
        )
        for result in evidence.search_results
        if result.result is SearchResultStatus.NO_RELEVANT_INTERACTION_FOUND
    )
    result = AnalysisResult(
        status=AnalysisStatus.INSUFFICIENT_DATA,
        interaction_level=InteractionLevel.WHITE,
        headline="Недостаточно данных для оценки взаимодействия",
        identified_drugs=[
            IdentifiedDrug(
                trade_name=drug.normalized_trade_name or drug.input_name,
                active_ingredients=drug.active_ingredients,
            )
            for drug in evidence.drugs
        ],
        summary=(
            "Один источник содержит запись о взаимодействии без определённого уровня "
            "значимости, а другой не нашёл прямого упоминания. Эти данные не позволяют "
            "оценить клиническую значимость и не доказывают безопасность комбинации."
        ),
        mechanism_or_effect=None,
        recommended_user_action=(
            "Обсудите результат с врачом или фармацевтом. "
            "Не изменяйте лечение самостоятельно."
        ),
        limitations=list(
            dict.fromkeys(
                [
                    *assessment.limitations,
                    "Источник взаимодействий не указал уровень клинической значимости.",
                ]
            )
        ),
        conflicts=assessment.conflicts,
        missing_information=[
            "Надёжные данные о клинической значимости взаимодействия."
        ],
        sources=cited_sources,
    )
    validate_analysis_against_evidence(evidence, result)
    return result


def _known_database_severity_result(
    evidence: EvidencePackage,
) -> AnalysisResult | None:
    """Explain a complete DDInter-only finding without a fallible model call."""
    assessment = evidence.deterministic_assessment
    documents = [
        document
        for document in evidence.documents
        if document.retrieval_status is RetrievalStatus.OK
    ]
    if (
        assessment is None
        or not evidence.search_coverage.complete
        or evidence.search_coverage.failed_sources
        or assessment.incomplete_source_ids
        or not assessment.positive_source_ids
        or any(
            not source_id.startswith("ddinter.")
            for source_id in assessment.positive_source_ids
        )
        or not documents
        or any(
            document.source_type != "interaction_database"
            for document in documents
        )
        or any(
            document.retrieval_status is not RetrievalStatus.OK
            for document in evidence.documents
        )
    ):
        return None

    severities = {
        (document.source_severity or "").strip().casefold()
        for document in documents
    }
    if severities == {"minor"}:
        level = InteractionLevel.YELLOW
        headline = "Возможно незначительное взаимодействие"
        summary = (
            "DDInter 2.0 содержит запись уровня Minor для этой пары. "
            "В проверенных разделах выбранных официальных инструкций прямое "
            "упоминание пары не найдено; это не доказывает безопасность комбинации."
        )
        effect = (
            "База классифицирует взаимодействие как незначительное, но не предоставляет "
            "достаточных данных для индивидуальной оценки эффекта."
        )
    elif severities and severities.issubset({"moderate", "major"}):
        level = InteractionLevel.ORANGE
        headline = "Обнаружено потенциально значимое взаимодействие"
        severity_label = "Major" if "major" in severities else "Moderate"
        summary = (
            f"DDInter 2.0 содержит запись уровня {severity_label} для этой пары. "
            "Такое взаимодействие может быть клинически значимым и требует обсуждения "
            "с врачом или фармацевтом."
        )
        effect = (
            "Подключённая база указывает на потенциально клинически значимое "
            "взаимодействие, но текущая запись не содержит достаточного описания "
            "механизма для индивидуального вывода."
        )
    else:
        return None

    if level not in assessment.allowed_levels:
        return None

    cited_sources = [
        ResultSource(
            source_id=document.source_id,
            title=document.title,
            url=document.url,
            supports=(
                "DDInter 2.0 классифицирует взаимодействие как "
                f"{document.source_severity}."
            ),
        )
        for document in documents
    ]
    cited_sources.extend(
        ResultSource(
            source_id=search_result.source_id,
            title=search_result.title,
            url=search_result.url,
            supports=(
                "В проверенных разделах прямое упоминание пары не найдено; "
                "это не доказывает безопасность комбинации."
            ),
        )
        for search_result in evidence.search_results
        if search_result.result is SearchResultStatus.NO_RELEVANT_INTERACTION_FOUND
    )
    result = AnalysisResult(
        status=AnalysisStatus.ANSWER,
        interaction_level=level,
        headline=headline,
        identified_drugs=[
            IdentifiedDrug(
                trade_name=drug.normalized_trade_name or drug.input_name,
                active_ingredients=drug.active_ingredients,
            )
            for drug in evidence.drugs
        ],
        summary=summary,
        mechanism_or_effect=effect,
        recommended_user_action=(
            "Обсудите результат с врачом или фармацевтом. "
            "Не изменяйте лечение самостоятельно."
        ),
        limitations=list(
            dict.fromkeys(
                [
                    *assessment.limitations,
                    "DDInter указывает уровень, но не даёт достаточных данных для "
                    "индивидуальной оценки механизма и риска.",
                    "Индивидуальные факторы и другие препараты не оценивались.",
                ]
            )
        ),
        conflicts=assessment.conflicts,
        missing_information=[],
        sources=cited_sources,
    )
    validate_analysis_against_evidence(evidence, result)
    return result


def _incomplete_coverage_result(evidence: EvidencePackage) -> AnalysisResult | None:
    """Answer white without a model call when incomplete coverage allows only white.

    Used only when no source found any interaction evidence; positive evidence
    still goes to YandexGPT so the found warning is explained in context.
    """
    assessment = evidence.deterministic_assessment
    coverage = evidence.search_coverage
    if (
        assessment is None
        or assessment.allowed_levels != [InteractionLevel.WHITE]
        or assessment.positive_source_ids
        or evidence.documents
        or (
            coverage.complete
            and not coverage.failed_sources
            and not assessment.incomplete_source_ids
        )
    ):
        return None

    cited = [
        search_result
        for search_result in evidence.search_results
        if search_result.result
        in (SearchResultStatus.NO_RELEVANT_INTERACTION_FOUND, SearchResultStatus.INCONCLUSIVE)
    ]
    if not cited:
        return None

    result = AnalysisResult(
        status=AnalysisStatus.INSUFFICIENT_DATA,
        interaction_level=InteractionLevel.WHITE,
        headline="Недостаточно данных для оценки взаимодействия",
        identified_drugs=[
            IdentifiedDrug(
                trade_name=drug.normalized_trade_name or drug.input_name,
                active_ingredients=drug.active_ingredients,
            )
            for drug in evidence.drugs
        ],
        summary=(
            "Не все подключённые источники смогли проверить эту пару: например, "
            "вещество отсутствует в базе взаимодействий или источник был недоступен. "
            "В доступных данных прямое упоминание взаимодействия не найдено, но это "
            "не доказывает безопасность комбинации."
        ),
        mechanism_or_effect=None,
        recommended_user_action=(
            "Обсудите комбинацию с врачом или фармацевтом. "
            "Не изменяйте лечение самостоятельно."
        ),
        limitations=list(
            dict.fromkeys(
                [
                    *assessment.limitations,
                    "Проверка подключённых источников неполная, поэтому вывод о "
                    "взаимодействии не сформирован.",
                ]
            )
        ),
        conflicts=assessment.conflicts,
        missing_information=["Полная проверка пары во всех подключённых источниках."],
        sources=[
            ResultSource(
                source_id=search_result.source_id,
                title=search_result.title,
                url=search_result.url,
                supports=(
                    "Источник не нашёл прямую запись о паре или не смог её проверить; "
                    "это не исключает взаимодействие."
                ),
            )
            for search_result in cited
        ],
    )
    validate_analysis_against_evidence(evidence, result)
    return result


def _pair_only_negative_result(evidence: EvidencePackage) -> AnalysisResult | None:
    """Refuse green when current sources only performed pair-literal lookups."""
    assessment = evidence.deterministic_assessment
    search_results = evidence.search_results
    if (
        assessment is None
        or not assessment.green_eligible
        or not search_results
        or any(
            not any(
                result.source_id.startswith(f"{provider_id}.")
                for provider_id in PAIR_ONLY_NEGATIVE_PROVIDERS
            )
            for result in search_results
        )
        or any(
            result.result is not SearchResultStatus.NO_RELEVANT_INTERACTION_FOUND
            for result in search_results
        )
    ):
        return None

    result = AnalysisResult(
        status=AnalysisStatus.INSUFFICIENT_DATA,
        interaction_level=InteractionLevel.WHITE,
        headline="Недостаточно данных для безопасного вывода",
        identified_drugs=[
            IdentifiedDrug(
                trade_name=drug.normalized_trade_name or drug.input_name,
                active_ingredients=drug.active_ingredients,
            )
            for drug in evidence.drugs
        ],
        summary=(
            "Прямое упоминание пары и запись в подключённой базе не найдены. "
            "Собранных данных недостаточно, чтобы полностью исключить взаимодействия, "
            "описанные через классы препаратов, механизмы или общие предупреждения, "
            "поэтому зелёный вывод о комбинации сформировать нельзя."
        ),
        mechanism_or_effect=None,
        recommended_user_action=(
            "Обсудите комбинацию с врачом или фармацевтом. "
            "Не изменяйте лечение самостоятельно."
        ),
        limitations=list(
            dict.fromkeys(
                [
                    *assessment.limitations,
                    "Полнота классовых и механизм-ориентированных предупреждений ограничена.",
                ]
            )
        ),
        conflicts=assessment.conflicts,
        missing_information=[
            "Дополнительные надёжные данные о классовых и механизм-ориентированных "
            "предупреждениях в официальных инструкциях."
        ],
        sources=[
            ResultSource(
                source_id=search_result.source_id,
                title=search_result.title,
                url=search_result.url,
                supports=(
                    "Источник не нашёл прямую запись или буквальное упоминание пары; "
                    "это не исключает классовое взаимодействие."
                ),
            )
            for search_result in search_results
        ],
    )
    validate_analysis_against_evidence(evidence, result)
    return result


def _candidate_label(name: str, dosage_form: str | None, strength: str | None) -> str:
    details = ", ".join(value for value in (dosage_form, strength) if value)
    return f"{name} ({details})" if details else name
