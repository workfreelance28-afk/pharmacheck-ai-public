"""Stage-2 tests: parsing, exact resolution, ambiguity, and typo safety."""

from __future__ import annotations

import unittest
from pathlib import Path

from pharmacheck.contracts import IdentityStatus
from pharmacheck.drug_recognition import (
    DrugCatalog,
    DrugCatalogEntry,
    PairParseStatus,
    PairResolutionStatus,
    normalize_name,
    parse_drug_pair,
)

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_drug_catalog.json"


class DrugRecognitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = DrugCatalog.from_json(FIXTURE)

    def test_plus_format(self) -> None:
        parsed = parse_drug_pair("Альфа + Бета")
        self.assertEqual(PairParseStatus.PARSED, parsed.status)
        self.assertEqual(["Альфа", "Бета"], [item.name for item in parsed.drugs])

    def test_natural_phrase_format(self) -> None:
        parsed = parse_drug_pair(
            "Я принимаю Тестпрепарат Альфа. Мне назначили Тестпрепарат Бета"
        )
        self.assertEqual(PairParseStatus.PARSED, parsed.status)

    def test_natural_phrase_with_comma(self) -> None:
        parsed = parse_drug_pair("Я принимаю Альфа, мне назначили Бета")
        self.assertEqual(PairParseStatus.PARSED, parsed.status)

    def test_labeled_lines_and_command(self) -> None:
        parsed = parse_drug_pair(
            "/check\nПрепарат 1: Тестпрепарат Альфа\nПрепарат 2: Тестпрепарат Бета"
        )
        self.assertEqual(PairParseStatus.PARSED, parsed.status)

    def test_only_phrase_format_is_marked_free_text(self) -> None:
        # D-018: a phrase may carry symptoms, so it must reach YandexGPT.
        self.assertTrue(parse_drug_pair("Я принимаю Альфа, мне назначили Бета").free_text)
        self.assertTrue(
            parse_drug_pair("Мне плохо. Я принимаю Альфа, мне назначили Бета").free_text
        )
        for message in (
            "Альфа + Бета",
            "/check Альфа + Бета",
            "Препарат 1: Альфа\nПрепарат 2: Бета",
        ):
            self.assertFalse(parse_drug_pair(message).free_text, message)

    def test_incomplete_pair_requests_clarification(self) -> None:
        parsed = parse_drug_pair("Только Альфа")
        self.assertEqual(PairParseStatus.INCOMPLETE, parsed.status)
        self.assertTrue(parsed.clarification)

    def test_more_than_two_plus_parts_are_rejected(self) -> None:
        parsed = parse_drug_pair("Альфа + Бета + Дуал")
        self.assertEqual(PairParseStatus.TOO_MANY, parsed.status)

    def test_strength_and_form_are_extracted(self) -> None:
        parsed = parse_drug_pair("Альфа таблетки 10 мг + Бета капсулы 20мг")
        first, second = parsed.drugs
        self.assertEqual(("10мг", "таблетки"), (first.strength, first.dosage_form))
        self.assertEqual(("20мг", "капсулы"), (second.strength, second.dosage_form))

    def test_case_spaces_quotes_and_yo_are_normalized(self) -> None:
        self.assertEqual("тестпрепарат елка", normalize_name(" «ТЕСТПРЕПАРАТ   ЁЛКА» "))
        parsed = parse_drug_pair("«ТЕСТПРЕПАРАТ ЁЛКА» + Бета")
        resolved = self.catalog.resolve_pair(parsed)
        self.assertEqual(PairResolutionStatus.READY, resolved.status)

    def test_exact_alias_resolves_with_high_confidence(self) -> None:
        resolved = self.catalog.resolve_pair(parse_drug_pair("Альфа + Бета"))
        self.assertEqual(PairResolutionStatus.READY, resolved.status)
        self.assertEqual(
            ["вещество альфа", "вещество бета"],
            [item.active_ingredients[0] for item in resolved.drugs],
        )

    def test_equivalent_aliases_do_not_duplicate_one_catalog_entry(self) -> None:
        entry = DrugCatalogEntry(
            product_id="test.same-normalized-name",
            trade_name="Препарат",
            aliases=["ПРЕПАРАТ", "Препарат®"],
            active_ingredients=["вещество"],
            source_id="synthetic.catalog",
        )
        catalog = DrugCatalog([entry])

        resolved = catalog.resolve(parse_drug_pair("Препарат + Бета").drugs[0])

        self.assertEqual(IdentityStatus.RESOLVED, resolved.identity_status)
        self.assertEqual(["вещество"], resolved.active_ingredients)

    def test_explicit_latin_alias_resolves(self) -> None:
        resolved = self.catalog.resolve_pair(
            parse_drug_pair("Testdrug Alpha + Testdrug Beta")
        )
        self.assertEqual(PairResolutionStatus.READY, resolved.status)

    def test_typo_is_suggested_but_not_auto_resolved(self) -> None:
        resolved = self.catalog.resolve_pair(parse_drug_pair("Альфо + Бета"))
        first = resolved.drugs[0]
        self.assertEqual(IdentityStatus.UNRESOLVED, first.identity_status)
        self.assertTrue(first.candidates)
        self.assertEqual(PairResolutionStatus.CLARIFICATION_NEEDED, resolved.status)

    def test_two_letter_typo_in_long_name_is_suggested_without_auto_resolution(self) -> None:
        catalog = DrugCatalog(
            [
                DrugCatalogEntry(
                    product_id="test.mydocalm",
                    trade_name="Мидокалм",
                    active_ingredients=["tolperisone"],
                    source_id="synthetic.catalog",
                ),
                DrugCatalogEntry(
                    product_id="test.escitalopram",
                    trade_name="Эсциталопрам",
                    active_ingredients=["escitalopram"],
                    source_id="synthetic.catalog",
                ),
            ]
        )

        resolved = catalog.resolve_pair(
            parse_drug_pair("эсцитолопрам + медикалм")
        )

        self.assertEqual(PairResolutionStatus.CLARIFICATION_NEEDED, resolved.status)
        self.assertEqual(
            [IdentityStatus.UNRESOLVED, IdentityStatus.UNRESOLVED],
            [drug.identity_status for drug in resolved.drugs],
        )
        self.assertEqual(
            [["Эсциталопрам"], ["Мидокалм"]],
            [[candidate.trade_name for candidate in drug.candidates] for drug in resolved.drugs],
        )

    def test_short_names_do_not_use_relaxed_typo_matching(self) -> None:
        catalog = DrugCatalog(
            [
                DrugCatalogEntry(
                    product_id="test.short",
                    trade_name="Кит",
                    active_ingredients=["substance"],
                    source_id="synthetic.catalog",
                )
            ]
        )

        resolved = catalog.resolve(parse_drug_pair("Кот + Бета").drugs[0])

        self.assertEqual([], resolved.candidates)

    def test_unknown_name_has_no_fabricated_identity(self) -> None:
        resolved = self.catalog.resolve_pair(parse_drug_pair("Совсемнеизвестный + Бета"))
        first = resolved.drugs[0]
        self.assertEqual(IdentityStatus.UNRESOLVED, first.identity_status)
        self.assertEqual([], first.active_ingredients)

    def test_ambiguous_brand_requests_form(self) -> None:
        resolved = self.catalog.resolve_pair(parse_drug_pair("Дуал + Бета"))
        first = resolved.drugs[0]
        self.assertEqual(IdentityStatus.AMBIGUOUS, first.identity_status)
        self.assertEqual(2, len(first.candidates))

    def test_form_disambiguates_brand(self) -> None:
        resolved = self.catalog.resolve_pair(parse_drug_pair("Дуал раствор 5 мг + Бета"))
        first = resolved.drugs[0]
        self.assertEqual(IdentityStatus.RESOLVED, first.identity_status)
        self.assertEqual(["вещество дельта"], first.active_ingredients)

    def test_strength_disambiguates_brand(self) -> None:
        resolved = self.catalog.resolve_pair(parse_drug_pair("Доза таблетки 10 мг + Бета"))
        first = resolved.drugs[0]
        self.assertEqual(IdentityStatus.RESOLVED, first.identity_status)
        self.assertEqual(["вещество дзета"], first.active_ingredients)

    def test_parse_failure_never_runs_catalog_resolution(self) -> None:
        resolved = self.catalog.resolve_pair(parse_drug_pair("Альфа"))
        self.assertEqual(PairResolutionStatus.CLARIFICATION_NEEDED, resolved.status)
        self.assertEqual([], resolved.drugs)


if __name__ == "__main__":
    unittest.main()
