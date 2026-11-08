from __future__ import annotations

from datetime import date

from scripts.import_grls_catalog import (
    GRLSRecord,
    IngredientMapping,
    build_catalog,
    collect_unmapped_ingredients,
    normalize_russian_ingredient,
)


def _mapping(russian_name, source_name):
    return IngredientMapping(
        russian_name=russian_name,
        source_name=source_name,
        source_id=f"who.inn.{source_name}",
        source_url="https://extranet.who.int/soinn/",
    )


def _record(
    registration,
    trade_name,
    ingredients,
    *,
    status="active",
    expiration=None,
    cancellation=None,
):
    return GRLSRecord(
        registration_number=registration,
        trade_name=trade_name,
        russian_ingredients=tuple(ingredients),
        status=status,
        expiration_date=expiration,
        cancellation_date=cancellation,
    )


def test_only_fully_mapped_live_products_enter_catalog():
    mappings = {
        "ибупрофен": _mapping("Ибупрофен", "ibuprofen"),
        "парацетамол": _mapping("Парацетамол", "acetaminophen"),
    }
    records = [
        _record("RU-1", "Нурофен", ["ибупрофен"]),
        _record("RU-2", "Комбинация", ["ибупрофен", "неизвестное вещество"]),
        _record("RU-3", "Отменённый", ["ибупрофен"], cancellation=date(2026, 1, 1)),
    ]

    entries, stats = build_catalog(records, mappings, as_of=date(2026, 8, 7))

    assert [entry.trade_name for entry in entries] == ["Нурофен"]
    assert entries[0].active_ingredients == ["ibuprofen"]
    assert entries[0].registration_numbers == ["RU-1"]
    assert stats["unmapped_unique_ingredients"] == 1


def test_live_status_shadows_changed_history_and_expired_changed_is_removed():
    mappings = {"ибупрофен": _mapping("Ибупрофен", "ibuprofen")}
    records = [
        _record("RU-1", "Актуальный", ["ибупрофен"]),
        _record("RU-1", "Старое название", ["ибупрофен"], status="changed"),
        _record(
            "RU-2",
            "Истёкший",
            ["ибупрофен"],
            status="changed",
            expiration=date(2025, 1, 1),
        ),
    ]

    entries, _ = build_catalog(records, mappings, as_of=date(2026, 8, 7))

    assert [entry.trade_name for entry in entries] == ["Актуальный"]


def test_russian_ingredient_normalization_is_conservative():
    assert normalize_russian_ingredient(" [Аскорбиновая   кислота] ") == "аскорбиновая кислота"


def test_unmapped_report_contains_only_live_unverified_ingredients():
    mappings = {"ибупрофен": _mapping("Ибупрофен", "ibuprofen")}
    records = [
        _record("RU-1", "Известный", ["ибупрофен"]),
        _record("RU-2", "Новый", ["эсциталопрам"]),
        _record("RU-3", "Ещё один", ["эсциталопрам", "толперизон"]),
        _record(
            "RU-4",
            "Отменённый",
            ["неактуальное вещество"],
            cancellation=date(2026, 1, 1),
        ),
    ]

    unmapped = collect_unmapped_ingredients(records, mappings, as_of=date(2026, 8, 9))

    assert unmapped == {"эсциталопрам": 2, "толперизон": 1}
