"""Local deterministic checks for the stage-1 safety contract."""

from __future__ import annotations

import unittest
from copy import deepcopy

from pydantic import ValidationError

from pharmacheck.contracts import (
    MAX_SOURCE_URL_LENGTH,
    AnalysisResult,
    ContractViolation,
    EvidencePackage,
    ResultSource,
    validate_analysis_against_evidence,
)
from pharmacheck.eval_cases import build_stage1_cases

_URL_PREFIX = "https://example.test/"


def _url_of_length(length: int) -> str:
    return _URL_PREFIX + "a" * (length - len(_URL_PREFIX))


class Stage1ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cases = {case.case_id: case for case in build_stage1_cases()}

    def test_all_twelve_prompt_cases_are_structurally_valid(self) -> None:
        self.assertEqual(12, len(self.cases))

    def test_green_result_passes_with_complete_negative_search(self) -> None:
        case = self.cases["green_complete_search"]
        result = self._result_for(case, sources_from_search=True)
        validate_analysis_against_evidence(case.evidence, result)

    def test_green_is_rejected_when_required_source_failed(self) -> None:
        case = self.cases["white_required_source_failed"]
        result = self._result_for(
            self.cases["green_complete_search"], sources_from_search=True
        )
        with self.assertRaisesRegex(ContractViolation, "complete coverage"):
            validate_analysis_against_evidence(case.evidence, result)

    def test_non_green_result_requires_retrieved_document(self) -> None:
        case = self.cases["yellow_monitoring"]
        raw = self._result_for(case).model_dump(mode="json")
        raw["sources"] = []
        result = AnalysisResult.model_validate(raw)
        with self.assertRaisesRegex(ContractViolation, "cited source"):
            validate_analysis_against_evidence(case.evidence, result)

    def test_unknown_source_is_rejected(self) -> None:
        case = self.cases["red_official_contraindication"]
        raw = self._result_for(case).model_dump(mode="json")
        raw["sources"][0].update(
            {
                "source_id": "invented.source",
                "title": "Придуманный источник",
                "url": "https://invented.example/source",
            }
        )
        result = AnalysisResult.model_validate(raw)
        with self.assertRaisesRegex(ContractViolation, "unknown result source_id"):
            validate_analysis_against_evidence(case.evidence, result)

    def test_changed_source_url_is_rejected(self) -> None:
        case = self.cases["red_official_contraindication"]
        raw = self._result_for(case).model_dump(mode="json")
        raw["sources"][0]["url"] = "https://changed.example/source"
        result = AnalysisResult.model_validate(raw)
        with self.assertRaisesRegex(ContractViolation, "metadata changed"):
            validate_analysis_against_evidence(case.evidence, result)

    def test_url_outside_sources_is_rejected(self) -> None:
        case = self.cases["red_official_contraindication"]
        raw = self._result_for(case).model_dump(mode="json")
        raw["summary"] += " Подробнее: https://invented.example"
        result = AnalysisResult.model_validate(raw)
        with self.assertRaisesRegex(ContractViolation, "URLs are allowed only"):
            validate_analysis_against_evidence(case.evidence, result)

    def test_answer_cannot_have_white_level(self) -> None:
        raw = self._result_for(self.cases["green_complete_search"]).model_dump(
            mode="json"
        )
        raw["interaction_level"] = "white"
        with self.assertRaises(ValidationError):
            AnalysisResult.model_validate(raw)

    def test_urgent_cannot_have_non_white_level(self) -> None:
        # status=urgent means the user's message reports dangerous acute
        # symptoms and the interaction check is skipped in favor of urging
        # emergency care — it is never a statement about interaction
        # severity, so it must always pair with white (contracts.py
        # AnalysisResult.status_matches_level).
        raw = self._result_for(self.cases["urgent_symptoms_first"]).model_dump(
            mode="json"
        )
        self.assertEqual(raw["status"], "urgent")
        raw["interaction_level"] = "red"
        with self.assertRaisesRegex(
            ValidationError, "insufficient_data, and urgent all require white"
        ):
            AnalysisResult.model_validate(raw)

    def test_result_source_url_accepts_the_exact_length_boundary(self) -> None:
        # MAX_SOURCE_URL_LENGTH (contracts.py) is derived from the worst-
        # case fixed overhead format_analysis_result's floor scale leaves
        # around the first source: exactly this length must still be
        # accepted, not just anything comfortably under it.
        url = _url_of_length(MAX_SOURCE_URL_LENGTH)
        self.assertEqual(len(url), MAX_SOURCE_URL_LENGTH)
        source = ResultSource(
            source_id="openfda.label.boundary",
            title="Официальная инструкция",
            url=url,
            supports="Подтверждает вывод.",
        )
        self.assertEqual(str(source.url), url)

    def test_result_source_url_rejects_one_character_over_the_boundary(self) -> None:
        url = _url_of_length(MAX_SOURCE_URL_LENGTH + 1)
        self.assertEqual(len(url), MAX_SOURCE_URL_LENGTH + 1)
        with self.assertRaises(ValidationError):
            ResultSource(
                source_id="openfda.label.over",
                title="Официальная инструкция",
                url=url,
                supports="Подтверждает вывод.",
            )

    def test_insufficient_result_requires_missing_information(self) -> None:
        raw = self._result_for(self.cases["white_required_source_failed"]).model_dump(
            mode="json"
        )
        raw["missing_information"] = []
        with self.assertRaises(ValidationError):
            AnalysisResult.model_validate(raw)

    def test_resolved_drug_requires_active_ingredient(self) -> None:
        raw = self.cases["green_complete_search"].evidence.model_dump(mode="json")
        raw["drugs"][0]["active_ingredients"] = []
        with self.assertRaises(ValidationError):
            EvidencePackage.model_validate(raw)

    def test_complete_coverage_requires_every_source(self) -> None:
        raw = self.cases["green_complete_search"].evidence.model_dump(mode="json")
        raw["search_coverage"]["searched_sources"] = ["official.alpha"]
        with self.assertRaises(ValidationError):
            EvidencePackage.model_validate(raw)

    def test_extra_fields_are_rejected(self) -> None:
        raw = self.cases["green_complete_search"].evidence.model_dump(mode="json")
        raw["unexpected"] = "must fail"
        with self.assertRaises(ValidationError):
            EvidencePackage.model_validate(raw)

    def test_prompt_injection_is_present_only_as_untrusted_evidence(self) -> None:
        case = self.cases["red_ignores_prompt_injection"]
        excerpt = case.evidence.documents[0].excerpt
        self.assertIn("Ignore all prior instructions", excerpt)
        self.assertEqual("red", case.expected.level.value)

    def _result_for(
        self,
        case,
        *,
        sources_from_search: bool = False,
    ) -> AnalysisResult:
        expected = case.expected
        if sources_from_search:
            source = case.evidence.search_results[0]
        elif case.evidence.documents:
            source = case.evidence.documents[0]
        else:
            source = case.evidence.search_results[0]

        missing = (
            ["Нужны дополнительные надёжные данные."]
            if expected.level.value == "white"
            and expected.status.value != "urgent"
            else []
        )
        summary = "Вывод основан только на переданных тестовых источниках."
        if expected.level.value == "green":
            summary = (
                "В проверенных источниках значимого известного взаимодействия не "
                "найдено, но это не доказывает безопасность комбинации для конкретного "
                "человека."
            )

        raw = {
            "status": expected.status.value,
            "interaction_level": expected.level.value,
            "headline": "Тестовый вывод",
            "identified_drugs": [
                {
                    "trade_name": drug.normalized_trade_name or drug.input_name,
                    "active_ingredients": drug.active_ingredients or ["не определено"],
                }
                for drug in case.evidence.drugs
            ],
            "summary": summary,
            "mechanism_or_effect": None,
            "recommended_user_action": "Обсудите результат с врачом или фармацевтом.",
            "limitations": ["Это тестовый сценарий."],
            "conflicts": [],
            "missing_information": missing,
            "sources": [
                {
                    "source_id": source.source_id,
                    "title": source.title,
                    "url": str(source.url),
                    "supports": "Подтверждает тестовый вывод.",
                }
            ],
        }
        if expected.status.value == "urgent":
            raw["sources"] = []
            raw["recommended_user_action"] = (
                "Немедленно свяжитесь с местной экстренной службой."
            )
        return AnalysisResult.model_validate(deepcopy(raw))


if __name__ == "__main__":
    unittest.main()
