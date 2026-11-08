from pathlib import Path

import pytest

from pharmacheck.contracts import IdentityStatus
from pharmacheck.drug_recognition import DrugCatalog, PairResolutionStatus, parse_drug_pair

PROJECT_ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not (PROJECT_ROOT / "data/drug_catalog.json").is_file(),
    reason="Private GRLS catalog is not distributed",
)


def test_real_grls_catalog_resolves_control_pair_to_source_query_names():
    catalog = DrugCatalog.from_json(PROJECT_ROOT / "data" / "drug_catalog.json")

    resolved = catalog.resolve_pair(parse_drug_pair("Нурофен + Варфарин"))

    assert resolved.status is PairResolutionStatus.READY
    assert resolved.drugs[0].active_ingredients == ["ibuprofen"]
    assert resolved.drugs[1].active_ingredients == ["warfarin"]
    assert all(
        source_id.startswith("grls.export.")
        for drug in resolved.drugs
        for source_id in drug.identity_source_ids
    )


def test_real_grls_catalog_resolves_generic_names_with_case_only_aliases():
    catalog = DrugCatalog.from_json(PROJECT_ROOT / "data" / "drug_catalog.json")

    resolved = catalog.resolve_pair(parse_drug_pair("Парацетамол + Лизиноприл"))

    assert resolved.status is PairResolutionStatus.READY
    assert resolved.drugs[0].active_ingredients == ["acetaminophen"]
    assert resolved.drugs[1].active_ingredients == ["lisinopril"]


def test_real_grls_catalog_keeps_combination_brand_composition():
    catalog = DrugCatalog.from_json(PROJECT_ROOT / "data" / "drug_catalog.json")

    resolved = catalog.resolve_pair(parse_drug_pair("Нурофен Интенсив + Варфарин"))

    assert resolved.status is PairResolutionStatus.READY
    assert resolved.drugs[0].active_ingredients == ["ibuprofen", "acetaminophen"]


def test_real_grls_catalog_resolves_escitalopram_and_mydocalm():
    catalog = DrugCatalog.from_json(PROJECT_ROOT / "data" / "drug_catalog.json")

    resolved = catalog.resolve_pair(parse_drug_pair("Эсциталопрам + Мидокалм"))

    assert resolved.status is PairResolutionStatus.READY
    assert resolved.drugs[0].active_ingredients == ["escitalopram"]
    assert resolved.drugs[1].active_ingredients == ["tolperisone"]


def test_expanded_grls_catalog_resolves_new_verified_ingredients():
    catalog = DrugCatalog.from_json(PROJECT_ROOT / "data" / "drug_catalog.json")

    resolved = catalog.resolve_pair(parse_drug_pair("Метформин + Амоксициллин"))

    assert resolved.status is PairResolutionStatus.READY
    assert resolved.drugs[0].active_ingredients == ["metformin"]
    assert resolved.drugs[1].active_ingredients == ["amoxicillin"]


def test_second_expansion_resolves_new_verified_ingredients():
    catalog = DrugCatalog.from_json(PROJECT_ROOT / "data" / "drug_catalog.json")

    resolved = catalog.resolve_pair(parse_drug_pair("Цетиризин + Левофлоксацин"))

    assert resolved.status is PairResolutionStatus.READY
    assert resolved.drugs[0].active_ingredients == ["cetirizine"]
    assert resolved.drugs[1].active_ingredients == ["levofloxacin"]


def test_real_grls_catalog_suggests_user_typos_without_resolving_them():
    catalog = DrugCatalog.from_json(PROJECT_ROOT / "data" / "drug_catalog.json")

    resolved = catalog.resolve_pair(parse_drug_pair("эсцитолопрам + медикалм"))

    assert resolved.status is PairResolutionStatus.CLARIFICATION_NEEDED
    assert all(drug.identity_status is IdentityStatus.UNRESOLVED for drug in resolved.drugs)
    assert resolved.drugs[0].candidates[0].trade_name == "Эсциталопрам"
    assert resolved.drugs[1].candidates[0].trade_name == "Мидокалм®"
