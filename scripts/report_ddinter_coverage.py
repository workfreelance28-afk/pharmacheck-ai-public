"""Report local catalog/DDInter name coverage with the reviewed runtime alias table."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from pharmacheck.drug_recognition import DrugCatalogEntry
from pharmacheck.sources.ddinter import (
    DDINTER_ALIASES_FILENAME,
    ROUTE_WORDS,
    DDInterSource,
    _normalize_name,
    load_name_aliases,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALIASES = PROJECT_ROOT / "data" / "ddinter" / DDINTER_ALIASES_FILENAME


def build_coverage(ingredients, snapshot, aliases):
    """Count unique, unambiguous names using the runtime's normalization.

    Exact-name coverage and reviewed-alias coverage are counted separately.
    Route-qualified candidates count only through a reviewed `route` alias.
    """
    ingredients = sorted(set(ingredients))
    alias_by_name = {_normalize_name(item["catalog_ingredient"]): item for item in aliases}
    ingredient_keys = {_normalize_name(name) for name in ingredients}
    if not set(alias_by_name) <= ingredient_keys:
        raise ValueError("alias source is absent from catalog")
    for alias in aliases:
        target_ids = snapshot.name_to_ids.get(_normalize_name(alias["ddinter_name"]), ())
        if len(target_ids) != 1:
            raise ValueError("reviewed alias target is missing or ambiguous in snapshot")

    routes = {}
    for identifier, name in sorted(snapshot.canonical_names.items()):
        if ROUTE_WORDS.search(name):
            base = _normalize_name(ROUTE_WORDS.sub("", name))
            routes.setdefault(base, []).append({"ddinter_id": identifier, "ddinter_name": name})
    rows = []
    for ingredient in ingredients:
        key = _normalize_name(ingredient)
        exact_ids = snapshot.name_to_ids.get(key, ())
        excluded_routes = routes.get(key, [])
        row = {
            "catalog_ingredient": ingredient,
            "status": "missing",
            "matched_ddinter_name": None,
            "matched_ddinter_id": None,
            "excluded_route_candidates": excluded_routes,
        }
        if len(exact_ids) == 1:
            identifier = next(iter(exact_ids))
            row.update(
                status="exact",
                matched_ddinter_name=snapshot.canonical_names[identifier],
                matched_ddinter_id=identifier,
            )
        elif exact_ids:
            row["status"] = "ambiguous_exact"
        elif key in alias_by_name:
            alias = alias_by_name[key]
            identifier = next(iter(snapshot.name_to_ids[_normalize_name(alias["ddinter_name"])]))
            row.update(
                status="reviewed_" + alias.get("kind", "spelling"),
                matched_ddinter_name=alias["ddinter_name"],
                matched_ddinter_id=identifier,
                identity_source_id=alias["identity_source_id"],
            )
        elif excluded_routes:
            row["status"] = "route_only"
        rows.append(row)
    counts = {
        status: sum(row["status"] == status for row in rows)
        for status in (
            "exact",
            "reviewed_spelling",
            "reviewed_synonym",
            "reviewed_route",
            "route_only",
            "ambiguous_exact",
            "missing",
        )
    }
    covered = (
        counts["exact"]
        + counts["reviewed_spelling"]
        + counts["reviewed_synonym"]
        + counts["reviewed_route"]
    )
    return {
        "alias_table_scope": "runtime_and_coverage",
        "counts": {
            "catalog_ingredients": len(rows),
            "runtime_exact_coverage": counts["exact"],
            "reviewed_table_coverage": covered,
            "unresolved_after_table": len(rows) - covered,
            **counts,
        },
        "ingredients": rows,
    }


def report_coverage(catalog_path: Path, snapshot_path: Path, aliases_path: Path):
    raw_catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(raw_catalog, list) or not raw_catalog:
        raise ValueError("catalog must be a nonempty array")
    entries = [DrugCatalogEntry.model_validate(entry) for entry in raw_catalog]
    ingredients = [ingredient for entry in entries for ingredient in entry.active_ingredients]
    snapshot = DDInterSource(snapshot_path)._get_snapshot()
    aliases = load_name_aliases(aliases_path)
    if snapshot.name_aliases != {
        _normalize_name(item["catalog_ingredient"]): _normalize_name(item["ddinter_name"])
        for item in aliases
    }:
        raise ValueError("report table must match the aliases installed with the snapshot")
    report = build_coverage(ingredients, snapshot, aliases)
    report.update(
        checked_at=datetime.now(UTC).isoformat(),
        snapshot_version=snapshot.manifest.snapshot_version,
        catalog_entries=len(entries),
        provenance_sha256={
            "catalog": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
            "snapshot_manifest": hashlib.sha256(
                (snapshot_path / "manifest.json").read_bytes()
            ).hexdigest(),
            "alias_table": hashlib.sha256(aliases_path.read_bytes()).hexdigest(),
        },
        limitations=[
            "Name coverage does not prove interaction coverage or safety.",
            "Reviewed aliases (spelling, RxNorm synonym D-020, single route D-019) apply "
            "to the local runtime; production changes only after publication.",
            "Other route-qualified candidates stay excluded.",
            "No fuzzy or unreviewed synonym, salt, formulation or route substitution.",
            "Private noncommercial report; keep outside public materials and Git.",
        ],
    )
    return report


def format_markdown(report):
    counts = report["counts"]
    lines = [
        "# Покрытие каталога именами DDInter",
        "",
        f"Снимок: `{report['snapshot_version']}`. Проверено: {report['checked_at']}.",
        f"Каталог: {report['catalog_entries']} записей, {counts['catalog_ingredients']} веществ.",
        "",
        f"Без таблицы: {counts['runtime_exact_coverage']} точных совпадений.",
        f"С проверенной таблицей в отчёте: {counts['reviewed_table_coverage']}; "
        f"не разрешены: {counts['unresolved_after_table']}.",
        "Таблица применяется в локальном DDInterSource; production меняется только после публикации.",
        "Совпадение имени не доказывает полноту данных или безопасность.",
        "Точное совпадение использует нормализацию runtime: NFKC, регистр, пунктуация, пробелы.",
        "",
        "## Проверенные соответствия",
        "",
        "| Вещество каталога | Имя DDInter | Тип | Подтверждение |",
        "| --- | --- | --- | --- |",
    ]
    kinds = {
        "reviewed_spelling": "только написание",
        "reviewed_synonym": "синоним RxNorm (D-020)",
        "reviewed_route": "единственный путь введения (D-019)",
    }
    for row in report["ingredients"]:
        if row["status"] in kinds:
            lines.append(
                f"| {row['catalog_ingredient']} | {row['matched_ddinter_name']} | "
                f"{kinds[row['status']]} | {row['identity_source_id']} |"
            )
    lines += [
        "",
        "## Записи с путём введения",
        "",
        "| Вещество каталога | Записи DDInter | Точное совпадение без этих записей |",
        "| --- | --- | --- |",
    ]
    for row in report["ingredients"]:
        if row["excluded_route_candidates"]:
            names = "; ".join(item["ddinter_name"] for item in row["excluded_route_candidates"])
            lines.append(
                f"| {row['catalog_ingredient']} | {names} | "
                f"{'да' if row['status'] == 'exact' else 'через D-019' if row['status'] == 'reviewed_route' else 'нет'} |"
            )
    lines += ["", "## Остальные неразрешённые имена", ""]
    for row in report["ingredients"]:
        if row["status"] in {"missing", "ambiguous_exact"}:
            lines.append(f"- {row['catalog_ingredient']} ({row['status']})")
    lines += [
        "",
        "Полный список и SHA-256 входных данных — в JSON-отчёте.",
        "Отчёт предназначен только для частного некоммерческого проекта; исключён из Git.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=PROJECT_ROOT / "data/drug_catalog.json")
    parser.add_argument("--snapshot", type=Path, default=PROJECT_ROOT / "data/ddinter")
    parser.add_argument("--aliases", type=Path, default=DEFAULT_ALIASES)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "dist/ddinter-name-coverage.json"
    )
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    report = report_coverage(args.catalog, args.snapshot, args.aliases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(format_markdown(report), encoding="utf-8")
    print(json.dumps(report["counts"], ensure_ascii=False))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
