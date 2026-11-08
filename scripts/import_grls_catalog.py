"""Create a fail-closed production catalog from an official GRLS XLSX export."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
import unicodedata
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from xml.etree import ElementTree as ET

from pydantic import BaseModel, ConfigDict, Field

from pharmacheck.drug_recognition import DrugCatalogEntry, normalize_name

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAPPINGS = PROJECT_ROOT / "data" / "ingredient_mappings.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "drug_catalog.json"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "drug_catalog.manifest.json"
SOURCE_PAGE_URL = "https://grls.rosminzdrav.ru/GRLS.aspx"
XML_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
CELL_COLUMN_RE = re.compile(r"[A-Z]+")

LIVE_STATUS_FILES = {
    "Действующий.xlsx": "active",
    "Изменённый.xlsx": "changed",
    "Выдано_по_правилам_ЕАЭС.xlsx": "eaeu",
    "Действует_на_подтверждении_государственной_регистрации.xlsx": "confirmation",
    "Действует_в_иностранных_упаковках.xlsx": "foreign_packaging",
}


class ImportErrorSafe(RuntimeError):
    """Expected invalid-source error without confidential or row-level data."""


class IngredientMapping(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    russian_name: str = Field(min_length=1)
    source_name: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9 .'-]*$")
    source_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.:-]+$")
    source_url: str = Field(pattern=r"^https://")


@dataclass(frozen=True, slots=True)
class GRLSRecord:
    registration_number: str
    trade_name: str
    russian_ingredients: tuple[str, ...]
    status: str
    expiration_date: date | None
    cancellation_date: date | None


def load_mappings(path: Path) -> dict[str, IngredientMapping]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        mappings = [IngredientMapping.model_validate(item) for item in raw]
    except (OSError, UnicodeError, ValueError) as exc:
        raise ImportErrorSafe("Ingredient mapping file is invalid.") from exc
    result: dict[str, IngredientMapping] = {}
    for mapping in mappings:
        key = normalize_russian_ingredient(mapping.russian_name)
        if key in result:
            raise ImportErrorSafe("Ingredient mapping contains duplicate Russian names.")
        result[key] = mapping
    if not result:
        raise ImportErrorSafe("Ingredient mapping file is empty.")
    return result


def iter_export_records(archive_path: Path) -> tuple[list[GRLSRecord], date]:
    if not archive_path.is_file():
        raise ImportErrorSafe("GRLS export archive is missing.")
    records: list[GRLSRecord] = []
    export_dates: set[date] = set()
    found_statuses: set[str] = set()
    try:
        with zipfile.ZipFile(archive_path) as outer:
            for filename in outer.namelist():
                status = _status_from_filename(filename)
                if status is None:
                    continue
                export_dates.add(_date_from_filename(filename))
                found_statuses.add(status)
                with tempfile.SpooledTemporaryFile(max_size=32 * 1024 * 1024) as nested:
                    nested.write(outer.read(filename))
                    nested.seek(0)
                    with zipfile.ZipFile(nested) as workbook:
                        records.extend(_read_sheet(workbook, status))
    except (OSError, UnicodeError, ValueError, zipfile.BadZipFile) as exc:
        raise ImportErrorSafe("GRLS export archive is invalid.") from exc
    required = {"active", "changed", "eaeu", "confirmation", "foreign_packaging"}
    if found_statuses != required:
        raise ImportErrorSafe("GRLS export does not contain every required live-status file.")
    if len(export_dates) != 1:
        raise ImportErrorSafe("GRLS export files have inconsistent dates.")
    return records, export_dates.pop()


def build_catalog(
    records: list[GRLSRecord],
    mappings: dict[str, IngredientMapping],
    *,
    as_of: date,
) -> tuple[list[DrugCatalogEntry], dict[str, int]]:
    accepted = _filter_live_records(records, as_of=as_of)

    groups: dict[tuple[str, tuple[str, ...]], dict[str, object]] = {}
    unmapped = Counter()
    mapped_rows = 0
    for record in accepted:
        resolved = []
        missing = []
        for russian_name in record.russian_ingredients:
            mapping = mappings.get(normalize_russian_ingredient(russian_name))
            if mapping is None:
                missing.append(normalize_russian_ingredient(russian_name))
            else:
                resolved.append(mapping.source_name)
        if missing:
            unmapped.update(name for name in missing if name)
            continue
        if not resolved:
            continue
        mapped_rows += 1
        ingredients = tuple(dict.fromkeys(resolved))
        key = (normalize_name(record.trade_name), tuple(sorted(ingredients)))
        group = groups.setdefault(
            key,
            {"trade_names": Counter(), "ingredients": ingredients, "registrations": set()},
        )
        group["trade_names"][record.trade_name] += 1
        group["registrations"].add(record.registration_number)

    entries = []
    for (normalized_trade, _), group in sorted(groups.items()):
        trade_name = sorted(
            group["trade_names"],
            key=lambda value: (-group["trade_names"][value], len(value), value.casefold()),
        )[0]
        aliases = sorted(
            {name for name in group["trade_names"] if name != trade_name},
            key=str.casefold,
        )
        registrations = sorted(group["registrations"])
        fingerprint = "|".join((normalized_trade, *group["ingredients"]))
        digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:20]
        entries.append(
            DrugCatalogEntry(
                product_id=f"grls.{digest}",
                trade_name=trade_name,
                aliases=aliases,
                active_ingredients=list(group["ingredients"]),
                source_id=f"grls.export.{as_of:%Y%m%d}",
                registration_numbers=registrations,
            )
        )
    stats = {
        "rows_total": len(records),
        "rows_live_after_status_filter": len(accepted),
        "rows_with_verified_ingredient_mapping": mapped_rows,
        "catalog_entries": len(entries),
        "unmapped_unique_ingredients": len(unmapped),
    }
    return entries, stats


def collect_unmapped_ingredients(
    records: list[GRLSRecord],
    mappings: dict[str, IngredientMapping],
    *,
    as_of: date,
) -> Counter[str]:
    """Return normalized unmapped ingredients from currently live GRLS rows."""

    unmapped = Counter()
    for record in _filter_live_records(records, as_of=as_of):
        for russian_name in record.russian_ingredients:
            normalized = normalize_russian_ingredient(russian_name)
            if normalized and normalized not in mappings:
                unmapped[normalized] += 1
    return unmapped


def _filter_live_records(records: list[GRLSRecord], *, as_of: date) -> list[GRLSRecord]:
    live_registration_numbers = {
        record.registration_number
        for record in records
        if record.status != "changed" and record.cancellation_date is None
    }
    changed_latest: dict[str, GRLSRecord] = {}
    accepted: list[GRLSRecord] = []
    for record in records:
        if record.cancellation_date is not None:
            continue
        if record.status == "changed":
            if record.registration_number in live_registration_numbers:
                continue
            if record.expiration_date is not None and record.expiration_date < as_of:
                continue
            changed_latest[record.registration_number] = record
        else:
            accepted.append(record)
    accepted.extend(changed_latest.values())
    return accepted


def import_catalog(
    *,
    archive_path: Path,
    mappings_path: Path = DEFAULT_MAPPINGS,
    output_path: Path = DEFAULT_OUTPUT,
    manifest_path: Path = DEFAULT_MANIFEST,
    unmapped_output_path: Path | None = None,
) -> tuple[Path, Path]:
    mappings = load_mappings(mappings_path)
    records, export_date = iter_export_records(archive_path)
    if export_date > datetime.now(UTC).date():
        raise ImportErrorSafe("GRLS export date is in the future.")
    entries, stats = build_catalog(records, mappings, as_of=export_date)
    if not entries:
        raise ImportErrorSafe("No catalog entries have verified ingredient mappings.")
    catalog_bytes = (
        json.dumps(
            [entry.model_dump(mode="json") for entry in entries],
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(catalog_bytes)
    manifest = {
        "schema_version": 1,
        "source_id": f"grls.export.{export_date:%Y%m%d}",
        "source_title": "Государственный реестр лекарственных средств",
        "source_page_url": SOURCE_PAGE_URL,
        "export_date": export_date.isoformat(),
        "imported_at": datetime.now(UTC).isoformat(),
        "archive_sha256": _sha256_file(archive_path),
        "mapping_sha256": _sha256_file(mappings_path),
        "catalog_sha256": hashlib.sha256(catalog_bytes).hexdigest(),
        "included_statuses": sorted(LIVE_STATUS_FILES.values()),
        **stats,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if unmapped_output_path is not None:
        unmapped = collect_unmapped_ingredients(records, mappings, as_of=export_date)
        report = {
            "schema_version": 1,
            "source_id": f"grls.export.{export_date:%Y%m%d}",
            "items": [
                {"russian_name": name, "live_row_count": count}
                for name, count in sorted(unmapped.items(), key=lambda item: (-item[1], item[0]))
            ],
        }
        unmapped_output_path.parent.mkdir(parents=True, exist_ok=True)
        unmapped_output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return output_path, manifest_path


def normalize_russian_ingredient(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().replace("ё", "е")
    value = value.strip().strip("[]").strip()
    return " ".join(value.split())


def _status_from_filename(filename: str) -> str | None:
    return next(
        (status for suffix, status in LIVE_STATUS_FILES.items() if filename.endswith(suffix)),
        None,
    )


def _date_from_filename(filename: str) -> date:
    match = re.search(r"grls(\d{4})-(\d{2})-(\d{2})", filename, flags=re.IGNORECASE)
    if match is None:
        raise ValueError("GRLS filename does not contain an export date")
    return date(*(int(value) for value in match.groups()))


def _read_sheet(workbook: zipfile.ZipFile, status: str) -> list[GRLSRecord]:
    strings = _shared_strings(workbook)
    records = []
    with workbook.open("xl/worksheets/sheet1.xml") as stream:
        for _, row in ET.iterparse(stream, events=("end",)):
            if row.tag.rsplit("}", 1)[-1] != "row":
                continue
            values = {}
            for cell in row.findall("m:c", XML_NS):
                match = CELL_COLUMN_RE.match(cell.get("r", ""))
                if match:
                    values[match.group()] = _cell_value(cell, strings)
            row.clear()
            registration = values.get("C", "").strip()
            trade_name = values.get("I", "").strip()
            ingredient_text = values.get("J", "").strip()
            if registration == "Номер регистрационного удостоверения":
                continue
            if not registration or not trade_name or ingredient_text in {"", "~"}:
                continue
            ingredients = tuple(
                item
                for item in (
                    normalize_russian_ingredient(value) for value in ingredient_text.split("+")
                )
                if item and item != "~"
            )
            if not ingredients:
                continue
            records.append(
                GRLSRecord(
                    registration_number=registration,
                    trade_name=trade_name,
                    russian_ingredients=ingredients,
                    status=status,
                    expiration_date=_optional_date(values.get("E", "")),
                    cancellation_date=_optional_date(values.get("F", "")),
                )
            )
    return records


def _shared_strings(workbook: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in workbook.namelist():
        return []
    root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
    return ["".join(node.itertext()) for node in root.findall("m:si", XML_NS)]


def _cell_value(cell: ET.Element, strings: list[str]) -> str:
    if cell.get("t") == "inlineStr":
        inline = cell.find("m:is", XML_NS)
        return "".join(inline.itertext()) if inline is not None else ""
    raw = cell.findtext("m:v", default="", namespaces=XML_NS)
    return strings[int(raw)] if cell.get("t") == "s" and raw else raw


def _optional_date(value: str) -> date | None:
    value = value.strip()
    if not value:
        return None
    return datetime.strptime(value, "%d.%m.%Y").date()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--mappings", type=Path, default=DEFAULT_MAPPINGS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--unmapped-output", type=Path)
    arguments = parser.parse_args()
    try:
        output, manifest = import_catalog(
            archive_path=arguments.archive,
            mappings_path=arguments.mappings,
            output_path=arguments.output,
            manifest_path=arguments.manifest,
            unmapped_output_path=arguments.unmapped_output,
        )
    except ImportErrorSafe as exc:
        parser.error(str(exc))
    print(output)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
