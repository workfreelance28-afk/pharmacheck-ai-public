"""Non-commercial DDInter 2.0 snapshot adapter."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path, PurePath
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, model_validator

from pharmacheck.contracts import (
    EvidenceDocument,
    RetrievalStatus,
    SearchResult,
    SearchResultStatus,
)
from pharmacheck.sources.base import InteractionSearchOutcome

DDINTER_DOWNLOAD_PAGE = "https://ddinter2.scbdd.com/download/"
DDINTER_TERMS_URL = "https://ddinter2.scbdd.com/terms/"
DDINTER_LICENSE_ID = "CC-BY-NC-SA-4.0"
DDINTER_ALIASES_FILENAME = "name_aliases.json"
ROUTE_WORDS = re.compile(r"\b(?:nasal|ophthalmic|topical)\b", re.IGNORECASE)
ROUTE_TARGET_RE = re.compile(
    r"^(?P<base>[^()]+?)\s*\((?P<route>nasal|ophthalmic|topical)\)$", re.IGNORECASE
)
# Every alias is reviewed; each kind has its own objective rule (D-019, D-020).
ALIAS_KINDS = frozenset({"spelling", "synonym", "route"})
SYNONYM_RELATIONS = frozenset({"same_ingredient", "salt_or_form_of"})
PUBLISHER = "Computational Biology & Drug Design Group (DDInter)"

EXPECTED_FILES = frozenset(
    {
        "ddinter_downloads_code_A.csv",
        "ddinter_downloads_code_B.csv",
        "ddinter_downloads_code_D.csv",
        "ddinter_downloads_code_H.csv",
        "ddinter_downloads_code_L.csv",
        "ddinter_downloads_code_P.csv",
        "ddinter_downloads_code_R.csv",
        "ddinter_downloads_code_V.csv",
    }
)
EXPECTED_COLUMNS = (
    "DDInterID_A",
    "Drug_A",
    "DDInterID_B",
    "Drug_B",
    "Level",
)
ALLOWED_SOURCE_HOSTS = frozenset({"ddinter.scbdd.com", "ddinter2.scbdd.com"})


class DDInterLevel(StrEnum):
    MAJOR = "Major"
    MODERATE = "Moderate"
    MINOR = "Minor"
    UNKNOWN = "Unknown"


class DDInterLicenseError(RuntimeError):
    """Raised before data access when a commercial configuration is requested."""


class DDInterSnapshotError(RuntimeError):
    """Safe error for an absent, incomplete, or corrupted local snapshot."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class DDInterManifestFile(_StrictModel):
    name: str = Field(min_length=1)
    url: AnyHttpUrl
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    size_bytes: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_safe_official_file(self) -> DDInterManifestFile:
        if PurePath(self.name).name != self.name or self.name not in EXPECTED_FILES:
            raise ValueError("unexpected DDInter filename")
        parsed = urlsplit(str(self.url))
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_SOURCE_HOSTS:
            raise ValueError("DDInter file URL must use an official HTTPS host")
        if not parsed.path.endswith(f"/{self.name}"):
            raise ValueError("DDInter file URL and filename do not match")
        return self


class DDInterManifest(_StrictModel):
    schema_version: Literal[1]
    dataset_name: Literal["DDInter 2.0"]
    snapshot_version: str = Field(min_length=1)
    release_date: date | None = None
    downloaded_at: datetime
    license_id: Literal["CC-BY-NC-SA-4.0"]
    license_url: AnyHttpUrl
    terms_url: AnyHttpUrl
    usage_mode: Literal["noncommercial_mvp"]
    files: list[DDInterManifestFile] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_complete_snapshot(self) -> DDInterManifest:
        names = [item.name for item in self.files]
        if len(names) != len(set(names)) or set(names) != EXPECTED_FILES:
            raise ValueError("manifest must contain every official DDInter CSV once")
        if urlsplit(str(self.terms_url)).hostname not in ALLOWED_SOURCE_HOSTS:
            raise ValueError("unexpected DDInter terms host")
        return self


@dataclass(frozen=True, slots=True)
class _DDInterRecord:
    drug_a_id: str
    drug_a: str
    drug_b_id: str
    drug_b: str
    level: DDInterLevel
    source_urls: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _DDInterSnapshot:
    manifest: DDInterManifest
    name_to_ids: dict[str, frozenset[str]]
    canonical_names: dict[str, str]
    records: dict[tuple[str, str], _DDInterRecord]
    name_aliases: dict[str, str]


class DDInterSource:
    """Search a verified local DDInter snapshot in non-commercial MVP mode."""

    def __init__(
        self,
        snapshot_dir: str | Path,
        *,
        commercial_use: bool = False,
    ) -> None:
        if commercial_use:
            raise DDInterLicenseError(
                "DDInter CC BY-NC-SA data is disabled for commercial use"
            )
        self._snapshot_dir = Path(snapshot_dir)
        self._snapshot: _DDInterSnapshot | None = None

    def search_pair(
        self,
        ingredient_a: str,
        ingredient_b: str,
        *,
        checked_at: datetime | None = None,
    ) -> InteractionSearchOutcome:
        first = _validate_ingredient(ingredient_a)
        second = _validate_ingredient(ingredient_b)
        first_key = _normalize_name(first)
        second_key = _normalize_name(second)
        if first_key == second_key:
            raise ValueError("ingredients must be different")

        checked = checked_at or datetime.now(UTC)
        try:
            snapshot = self._get_snapshot()
        except DDInterSnapshotError:
            return _failed_outcome(first, second, checked)

        first_ids = snapshot.name_to_ids.get(first_key, frozenset())
        second_ids = snapshot.name_to_ids.get(second_key, frozenset())
        used_aliases = []
        for key, name in ((first_key, first), (second_key, second)):
            # A real (even ambiguous) snapshot name takes precedence. Only
            # the explicitly reviewed table may resolve a missing name.
            if key not in snapshot.name_to_ids and key in snapshot.name_aliases:
                target = snapshot.name_aliases[key]
                ids = snapshot.name_to_ids[target]
                used_aliases.append(f"{name} -> {snapshot.canonical_names[next(iter(ids))]}")
                if key == first_key:
                    first_ids = ids
                else:
                    second_ids = ids
        alias_note = (
            " Проверенное соответствие названия DDInter: " + "; ".join(used_aliases) + "."
            if used_aliases
            else ""
        )
        unresolved = [
            name
            for name, identifiers in ((first, first_ids), (second, second_ids))
            if not identifiers
        ]
        ambiguous = [
            name
            for name, identifiers in ((first, first_ids), (second, second_ids))
            if len(identifiers) > 1
        ]

        if unresolved or ambiguous:
            details: list[str] = []
            if unresolved:
                details.append("не распознаны: " + ", ".join(unresolved))
            if ambiguous:
                details.append("неоднозначны: " + ", ".join(ambiguous))
            return _outcome(
                first,
                second,
                checked,
                snapshot=snapshot,
                status=SearchResultStatus.INCONCLUSIVE,
                details="DDInter: " + "; ".join(details) + "." + alias_note,
                complete=False,
            )

        first_id = next(iter(first_ids))
        second_id = next(iter(second_ids))
        if first_id == second_id:
            raise ValueError("ingredients must be different after alias resolution")
        record = snapshot.records.get(_pair_key(first_id, second_id))
        if record is None:
            return _outcome(
                first,
                second,
                checked,
                snapshot=snapshot,
                status=SearchResultStatus.NO_RELEVANT_INTERACTION_FOUND,
                details=(
                    "В полном CSV-наборе, опубликованном разработчиками DDInter 2.0, "
                    "запись для пары не найдена. Это не доказывает безопасность "
                    "комбинации." + alias_note
                ),
                complete=True,
            )

        document = _record_to_document(
            record,
            snapshot.manifest,
            checked,
            query_pair=(first, second) if used_aliases else None,
        )
        details = f"DDInter 2.0 содержит запись уровня {record.level.value}."
        details += alias_note
        return _outcome(
            first,
            second,
            checked,
            snapshot=snapshot,
            status=SearchResultStatus.INTERACTION_FOUND,
            details=details,
            complete=True,
            documents=[document],
        )

    def _get_snapshot(self) -> _DDInterSnapshot:
        if self._snapshot is None:
            self._snapshot = _load_snapshot(self._snapshot_dir)
        return self._snapshot


def _load_snapshot(snapshot_dir: Path) -> _DDInterSnapshot:
    manifest_path = snapshot_dir / "manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = DDInterManifest.model_validate(payload)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise DDInterSnapshotError("invalid_manifest") from exc

    names: dict[str, set[str]] = {}
    canonical_names: dict[str, str] = {}
    accumulated: dict[tuple[str, str], tuple[_DDInterRecord, set[str]]] = {}

    for file_info in manifest.files:
        path = snapshot_dir / file_info.name
        _verify_file(path, file_info)
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                if tuple(reader.fieldnames or ()) != EXPECTED_COLUMNS:
                    raise DDInterSnapshotError("invalid_csv_header")
                for row in reader:
                    record = _parse_row(row, str(file_info.url))
                    _add_name(names, canonical_names, record.drug_a_id, record.drug_a)
                    _add_name(names, canonical_names, record.drug_b_id, record.drug_b)
                    key = _pair_key(record.drug_a_id, record.drug_b_id)
                    existing = accumulated.get(key)
                    if existing is None:
                        accumulated[key] = (record, {str(file_info.url)})
                    elif existing[0].level is not record.level:
                        raise DDInterSnapshotError("conflicting_pair_levels")
                    else:
                        existing[1].add(str(file_info.url))
        except (OSError, UnicodeError, csv.Error) as exc:
            raise DDInterSnapshotError("invalid_csv") from exc

    if not accumulated:
        raise DDInterSnapshotError("empty_snapshot")

    records = {
        key: _DDInterRecord(
            drug_a_id=value[0].drug_a_id,
            drug_a=value[0].drug_a,
            drug_b_id=value[0].drug_b_id,
            drug_b=value[0].drug_b,
            level=value[0].level,
            source_urls=tuple(sorted(value[1])),
        )
        for key, value in accumulated.items()
    }
    aliases_path = snapshot_dir / DDINTER_ALIASES_FILENAME
    try:
        aliases = load_name_aliases(aliases_path) if aliases_path.exists() else []
    except (OSError, UnicodeError, ValueError) as exc:
        raise DDInterSnapshotError("invalid_alias_table") from exc
    name_aliases = {}
    for alias in aliases:
        source = _normalize_name(alias["catalog_ingredient"])
        target = _normalize_name(alias["ddinter_name"])
        if len(names.get(target, ())) != 1:
            raise DDInterSnapshotError("alias_target_missing_or_ambiguous")
        if alias.get("kind", "spelling") == "route" and not _route_alias_is_unique(
            names, source, target
        ):
            # D-019: only a substance that DDInter lists under exactly one route
            # and never without a route may use that route's records.
            raise DDInterSnapshotError("route_alias_not_unique")
        name_aliases[source] = target
    return _DDInterSnapshot(
        manifest=manifest,
        name_to_ids={key: frozenset(value) for key, value in names.items()},
        canonical_names=canonical_names,
        records=records,
        name_aliases=name_aliases,
    )


def load_name_aliases(path: Path) -> list[dict[str, str]]:
    """Validate the explicit spelling table; never infer aliases or routes."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(raw, dict)
        or raw.get("schema_version") != 1
        or raw.get("usage_mode") != "noncommercial_mvp"
        or raw.get("scope") != "runtime_and_coverage"
        or not isinstance(raw.get("aliases"), list)
    ):
        raise ValueError("invalid DDInter alias table")
    aliases = raw["aliases"]
    seen = set()
    for alias in aliases:
        if not isinstance(alias, dict):
            raise ValueError("alias must be an object")
        for field in (
            "catalog_ingredient",
            "ddinter_name",
            "identity_source_id",
            "identity_source_url",
            "verification",
            "reviewed_on",
        ):
            if not isinstance(alias.get(field), str) or not alias[field].strip():
                raise ValueError(f"alias requires {field}")
        source, target = alias["catalog_ingredient"], alias["ddinter_name"]
        kind = alias.get("kind", "spelling")
        if kind not in ALIAS_KINDS:
            raise ValueError("unknown alias kind")
        if ROUTE_WORDS.search(source):
            raise ValueError("route-qualified aliases are prohibited")
        if kind == "route":
            _validate_route_alias(alias)
        elif ROUTE_WORDS.search(target):
            raise ValueError("route-qualified aliases are prohibited")
        elif kind == "spelling":
            spelling_keys = [
                "".join(unicodedata.normalize("NFKC", name).casefold().split())
                for name in (source, target)
            ]
            if spelling_keys[0] != spelling_keys[1]:
                raise ValueError("alias must differ only in whitespace/case")
        else:
            _validate_synonym_alias(alias)
        key = _normalize_name(source)
        if key in seen:
            raise ValueError("duplicate catalog alias")
        seen.add(key)
    return aliases


def _validate_synonym_alias(alias: dict[str, Any]) -> None:
    """D-020: RxNorm must show the same ingredient or a salt/form of it."""
    relation = alias.get("relation")
    if relation not in SYNONYM_RELATIONS:
        raise ValueError("synonym alias requires a known relation")
    fields = ["catalog_rxcui", "ddinter_name_rxcui"]
    if relation == "salt_or_form_of":
        fields.append("ddinter_ingredient_rxcui")
    for field in fields:
        if not isinstance(alias.get(field), str) or not alias[field].isdigit():
            raise ValueError(f"synonym alias requires {field}")
    if relation == "same_ingredient":
        if alias["ddinter_name_rxcui"] != alias["catalog_rxcui"]:
            raise ValueError("same_ingredient synonym must share one RxCUI")
    elif (
        alias["ddinter_ingredient_rxcui"] != alias["catalog_rxcui"]
        or alias["ddinter_name_rxcui"] == alias["catalog_rxcui"]
    ):
        raise ValueError("salt_or_form_of synonym must resolve to the catalog ingredient")


def _validate_route_alias(alias: dict[str, Any]) -> None:
    """D-019: the target is exactly '<catalog ingredient> (<route>)'."""
    match = ROUTE_TARGET_RE.fullmatch(alias["ddinter_name"].strip())
    if match is None or _normalize_name(match["base"]) != _normalize_name(
        alias["catalog_ingredient"]
    ):
        raise ValueError("route alias target must be the catalog ingredient plus one route")
    if alias.get("route") != match["route"].casefold():
        raise ValueError("route alias must name its route")


def _route_alias_is_unique(
    names: dict[str, set[str]],
    source: str,
    target: str,
) -> bool:
    if source in names:
        return False
    variants = [
        name
        for name in names
        if name.startswith(source + " ") and ROUTE_WORDS.fullmatch(name[len(source) + 1 :])
    ]
    return variants == [target]


def _verify_file(path: Path, file_info: DDInterManifestFile) -> None:
    try:
        size = path.stat().st_size
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise DDInterSnapshotError("missing_snapshot_file") from exc
    if size != file_info.size_bytes or digest != file_info.sha256:
        raise DDInterSnapshotError("snapshot_checksum_mismatch")


def _parse_row(row: dict[str, Any], source_url: str) -> _DDInterRecord:
    try:
        first_id = _validate_ddinter_id(row["DDInterID_A"])
        second_id = _validate_ddinter_id(row["DDInterID_B"])
        first = _validate_ingredient(row["Drug_A"])
        second = _validate_ingredient(row["Drug_B"])
        level = DDInterLevel(row["Level"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DDInterSnapshotError("invalid_csv_row") from exc
    if first_id == second_id:
        raise DDInterSnapshotError("self_interaction")
    return _DDInterRecord(
        drug_a_id=first_id,
        drug_a=first,
        drug_b_id=second_id,
        drug_b=second,
        level=level,
        source_urls=(source_url,),
    )


def _add_name(
    names: dict[str, set[str]],
    canonical_names: dict[str, str],
    identifier: str,
    name: str,
) -> None:
    names.setdefault(_normalize_name(name), set()).add(identifier)
    canonical_names.setdefault(identifier, name)


def _record_to_document(
    record: _DDInterRecord,
    manifest: DDInterManifest,
    accessed_at: datetime,
    *,
    query_pair: tuple[str, str] | None = None,
) -> EvidenceDocument:
    digest = hashlib.sha256(
        "|".join(sorted((record.drug_a_id, record.drug_b_id))).encode()
    ).hexdigest()[:16]
    excerpt = (
        f"DDInter 2.0 classifies the interaction between {record.drug_a} and "
        f"{record.drug_b} as {record.level.value}."
    )
    return EvidenceDocument(
        source_id=f"ddinter.record.{digest}",
        source_tier=3,
        source_type="interaction_database",
        title=f"DDInter 2.0: {record.drug_a} + {record.drug_b}",
        publisher=PUBLISHER,
        url=record.source_urls[0],
        license_name=DDINTER_LICENSE_ID,
        license_url=str(manifest.license_url),
        document_date=manifest.release_date,
        accessed_at=accessed_at,
        language="en",
        applies_to=list(query_pair) if query_pair else [record.drug_a, record.drug_b],
        section="severity",
        source_severity=record.level.value.casefold(),
        excerpt=excerpt,
        retrieval_status=RetrievalStatus.OK,
    )


def _outcome(
    first: str,
    second: str,
    checked_at: datetime,
    *,
    snapshot: _DDInterSnapshot,
    status: SearchResultStatus,
    details: str,
    complete: bool,
    documents: list[EvidenceDocument] | None = None,
) -> InteractionSearchOutcome:
    search_result = SearchResult(
        source_id=_search_source_id(first, second),
        title=f"DDInter 2.0 search ({snapshot.manifest.snapshot_version})",
        publisher=PUBLISHER,
        url=DDINTER_DOWNLOAD_PAGE,
        license_name=DDINTER_LICENSE_ID,
        license_url=str(snapshot.manifest.license_url),
        document_date=snapshot.manifest.release_date,
        accessed_at=checked_at,
        query_pairs=[f"{first} + {second}"],
        searched_sections=["DDInter 2.0 downloadable DDI CSV bundle"],
        result=status,
        details=details,
    )
    return InteractionSearchOutcome(
        provider_id="ddinter",
        ingredient_pair=(first, second),
        search_result=search_result,
        documents=documents or [],
        complete=complete,
        failed_queries=[],
    )


def _failed_outcome(
    first: str,
    second: str,
    checked_at: datetime,
) -> InteractionSearchOutcome:
    search_result = SearchResult(
        source_id=_search_source_id(first, second),
        title="DDInter 2.0 local snapshot search",
        publisher=PUBLISHER,
        url=DDINTER_DOWNLOAD_PAGE,
        license_name=DDINTER_LICENSE_ID,
        license_url="https://creativecommons.org/licenses/by-nc-sa/4.0/",
        document_date=None,
        accessed_at=checked_at,
        query_pairs=[f"{first} + {second}"],
        searched_sections=["DDInter 2.0 downloadable DDI CSV bundle"],
        result=SearchResultStatus.FAILED,
        details="Локальная выборка DDInter отсутствует или не прошла проверку.",
    )
    return InteractionSearchOutcome(
        provider_id="ddinter",
        ingredient_pair=(first, second),
        search_result=search_result,
        complete=False,
        failed_queries=["snapshot_load"],
    )


def _validate_ddinter_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"DDInter\d+", value):
        raise ValueError("invalid DDInter identifier")
    return value


def _validate_ingredient(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("ingredient must be text")
    normalized = unicodedata.normalize("NFKC", value).strip()
    if not normalized or len(normalized) > 200:
        raise ValueError("ingredient must contain 1 to 200 characters")
    if any(unicodedata.category(char).startswith("C") for char in normalized):
        raise ValueError("ingredient contains control characters")
    return " ".join(normalized.split())


def _normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", value).split())


def _pair_key(first_id: str, second_id: str) -> tuple[str, str]:
    return tuple(sorted((first_id, second_id)))


def _search_source_id(first: str, second: str) -> str:
    normalized = sorted((_normalize_name(first), _normalize_name(second)))
    digest = hashlib.sha256("|".join(normalized).encode()).hexdigest()[:16]
    return f"ddinter.search.{digest}"
