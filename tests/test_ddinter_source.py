from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from pharmacheck.contracts import (
    DrugIdentity,
    InteractionLevel,
    RetrievalStatus,
    SearchResult,
    SearchResultStatus,
)
from pharmacheck.evidence import build_evidence_package
from pharmacheck.sources import InteractionSource
from pharmacheck.sources.base import InteractionSearchOutcome
from pharmacheck.sources.ddinter import (
    DDINTER_ALIASES_FILENAME,
    DDINTER_LICENSE_ID,
    EXPECTED_FILES,
    DDInterLicenseError,
    DDInterSource,
)

CHECKED_AT = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
HEADER = ("DDInterID_A", "Drug_A", "DDInterID_B", "Drug_B", "Level")
BASE_ROWS = (
    ("DDInter1", "Alpha", "DDInter2", "Beta", "Major"),
    ("DDInter3", "Gamma", "DDInter4", "Delta", "Minor"),
)


def _write_snapshot(
    root: Path,
    *,
    rows: Iterable[tuple[str, str, str, str, str]] = BASE_ROWS,
    missing_file: str | None = None,
    conflicting_row: tuple[str, str, str, str, str] | None = None,
) -> Path:
    root.mkdir()
    rows = tuple(rows)
    files: list[dict[str, object]] = []
    for index, filename in enumerate(sorted(EXPECTED_FILES)):
        if filename == missing_file:
            continue
        path = root / filename
        file_rows = list(rows) if index == 0 else []
        if index == 1 and conflicting_row is not None:
            file_rows.append(conflicting_row)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(HEADER)
            writer.writerows(file_rows)
        payload = path.read_bytes()
        files.append(
            {
                "name": filename,
                "url": (
                    "https://ddinter2.scbdd.com/static/media/download/"
                    + filename
                ),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )

    manifest = {
        "schema_version": 1,
        "dataset_name": "DDInter 2.0",
        "snapshot_version": "synthetic-test",
        "release_date": None,
        "downloaded_at": "2026-08-08T12:00:00+00:00",
        "license_id": "CC-BY-NC-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "terms_url": "https://ddinter2.scbdd.com/terms/",
        "usage_mode": "noncommercial_mvp",
        "files": files,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_major_interaction_creates_attributed_evidence(tmp_path: Path) -> None:
    source = DDInterSource(_write_snapshot(tmp_path / "snapshot"))

    outcome = source.search_pair("alpha", "BETA", checked_at=CHECKED_AT)

    assert isinstance(source, InteractionSource)
    assert outcome.complete is True
    assert outcome.search_result.result is SearchResultStatus.INTERACTION_FOUND
    assert outcome.search_result.license_name == DDINTER_LICENSE_ID
    assert "by-nc-sa" in str(outcome.search_result.license_url)
    assert len(outcome.documents) == 1
    document = outcome.documents[0]
    assert document.source_tier == 3
    assert document.source_type == "interaction_database"
    assert document.section == "severity"
    assert document.retrieval_status is RetrievalStatus.OK
    assert document.license_name == DDINTER_LICENSE_ID
    assert "Major" in document.excerpt


def test_pair_lookup_is_symmetric(tmp_path: Path) -> None:
    source = DDInterSource(_write_snapshot(tmp_path / "snapshot"))

    forward = source.search_pair("Alpha", "Beta", checked_at=CHECKED_AT)
    reverse = source.search_pair("Beta", "Alpha", checked_at=CHECKED_AT)

    assert forward.search_result.source_id == reverse.search_result.source_id
    assert forward.documents[0].source_id == reverse.documents[0].source_id


def test_absent_pair_is_negative_only_when_both_names_are_known(
    tmp_path: Path,
) -> None:
    source = DDInterSource(_write_snapshot(tmp_path / "snapshot"))

    outcome = source.search_pair("Alpha", "Gamma", checked_at=CHECKED_AT)

    assert outcome.complete is True
    assert outcome.documents == []
    assert (
        outcome.search_result.result
        is SearchResultStatus.NO_RELEVANT_INTERACTION_FOUND
    )
    assert "не доказывает безопасность" in (outcome.search_result.details or "")
    assert "официаль" not in (outcome.search_result.details or "").casefold()


def test_unknown_name_is_inconclusive(tmp_path: Path) -> None:
    source = DDInterSource(_write_snapshot(tmp_path / "snapshot"))

    outcome = source.search_pair("Alpha", "Unknown", checked_at=CHECKED_AT)

    assert outcome.complete is False
    assert outcome.search_result.result is SearchResultStatus.INCONCLUSIVE


def test_unknown_severity_is_evidence_not_a_negative_result(tmp_path: Path) -> None:
    source = DDInterSource(
        _write_snapshot(
            tmp_path / "snapshot",
            rows=(("DDInter1", "Alpha", "DDInter2", "Beta", "Unknown"),),
        )
    )

    outcome = source.search_pair("Alpha", "Beta", checked_at=CHECKED_AT)

    assert outcome.complete is True
    assert outcome.search_result.result is SearchResultStatus.INTERACTION_FOUND
    assert "Unknown" in outcome.documents[0].excerpt


def test_commercial_mode_is_blocked_before_snapshot_access(tmp_path: Path) -> None:
    with pytest.raises(DDInterLicenseError, match="commercial"):
        DDInterSource(tmp_path / "does-not-exist", commercial_use=True)


def test_incomplete_manifest_is_failed_not_negative(tmp_path: Path) -> None:
    missing = sorted(EXPECTED_FILES)[-1]
    source = DDInterSource(
        _write_snapshot(tmp_path / "snapshot", missing_file=missing)
    )

    outcome = source.search_pair("Alpha", "Gamma", checked_at=CHECKED_AT)

    assert outcome.complete is False
    assert outcome.search_result.result is SearchResultStatus.FAILED
    assert outcome.failed_queries == ["snapshot_load"]


def test_checksum_mismatch_is_failed_not_negative(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path / "snapshot")
    target = snapshot / sorted(EXPECTED_FILES)[0]
    target.write_text("tampered", encoding="utf-8")
    source = DDInterSource(snapshot)

    outcome = source.search_pair("Alpha", "Gamma", checked_at=CHECKED_AT)

    assert outcome.complete is False
    assert outcome.search_result.result is SearchResultStatus.FAILED


def test_conflicting_duplicate_levels_fail_closed(tmp_path: Path) -> None:
    source = DDInterSource(
        _write_snapshot(
            tmp_path / "snapshot",
            conflicting_row=(
                "DDInter1",
                "Alpha",
                "DDInter2",
                "Beta",
                "Minor",
            ),
        )
    )

    outcome = source.search_pair("Alpha", "Beta", checked_at=CHECKED_AT)

    assert outcome.complete is False
    assert outcome.search_result.result is SearchResultStatus.FAILED


@pytest.mark.parametrize("value", ["", "   ", "bad\nname", "x" * 201])
def test_invalid_input_is_rejected_before_snapshot_access(
    tmp_path: Path,
    value: str,
) -> None:
    source = DDInterSource(tmp_path / "does-not-exist")

    with pytest.raises(ValueError):
        source.search_pair(value, "Valid")


def test_same_ingredient_is_rejected(tmp_path: Path) -> None:
    source = DDInterSource(tmp_path / "does-not-exist")

    with pytest.raises(ValueError, match="different"):
        source.search_pair("Alpha", " alpha ")


def _write_aliases(root, *, source="ethinyl estradiol", target="Ethinylestradiol"):
    row = {
        "catalog_ingredient": source,
        "ddinter_name": target,
        "identity_source_id": "rxnorm.rxcui.4124",
        "identity_source_url": "https://rxnav.nlm.nih.gov/REST/rxcui/4124/properties.json",
        "verification": "Synthetic spelling fixture; no clinical reference data.",
        "reviewed_on": "2026-09-28",
    }
    table = {
        "schema_version": 1,
        "usage_mode": "noncommercial_mvp",
        "scope": "runtime_and_coverage",
        "aliases": [row],
    }
    (root / DDINTER_ALIASES_FILENAME).write_text(json.dumps(table), encoding="utf-8")


def test_spelling_alias_retains_original_evidence_pair_and_source_attribution(tmp_path):
    # Severity and IDs are synthetic; only the spelling relationship is real.
    snapshot = _write_snapshot(
        tmp_path / "snapshot", rows=(("DDInter1", "Ethinylestradiol", "DDInter2", "Beta", "Minor"),)
    )
    _write_aliases(snapshot)
    source = DDInterSource(snapshot)
    pair = ("ethinyl estradiol", "Beta")
    outcome = source.search_pair(*pair, checked_at=CHECKED_AT)
    native = source.search_pair("Ethinylestradiol", "Beta", checked_at=CHECKED_AT)
    reverse = source.search_pair("Beta", "ethinyl estradiol", checked_at=CHECKED_AT)

    assert outcome.complete is True
    assert outcome.search_result.result is SearchResultStatus.INTERACTION_FOUND
    assert outcome.documents[0].source_id == native.documents[0].source_id
    assert outcome.documents[0].source_id == reverse.documents[0].source_id
    assert outcome.search_result.source_id == reverse.search_result.source_id
    assert outcome.documents[0].applies_to == list(pair)
    assert "Ethinylestradiol" in outcome.documents[0].excerpt
    assert "ethinyl estradiol -> Ethinylestradiol" in outcome.search_result.details
    assert outcome.documents[0].source_severity == "minor"
    assert outcome.documents[0].license_name == DDINTER_LICENSE_ID

    negative = InteractionSearchOutcome(
        provider_id="dailymed.label",
        ingredient_pair=pair,
        search_result=SearchResult(
            source_id="dailymed.label.search.synthetic",
            title="Synthetic negative search",
            publisher="Synthetic fixture",
            url="https://example.test/dailymed",
            accessed_at=CHECKED_AT,
            query_pairs=["ethinyl estradiol + Beta"],
            searched_sections=["drug_interactions"],
            result=SearchResultStatus.NO_RELEVANT_INTERACTION_FOUND,
        ),
        documents=[],
        complete=True,
    )
    evidence = build_evidence_package(
        request_id="spelling-alias",
        user_message="ethinyl estradiol + Beta",
        drugs=[
            DrugIdentity(
                input_name=name,
                normalized_trade_name=name,
                active_ingredients=[name],
                identity_status="resolved",
                identity_confidence="high",
            )
            for name in pair
        ],
        outcomes=[negative, outcome],
        required_provider_ids=("dailymed.label", "ddinter"),
        checked_at=CHECKED_AT,
    )
    assert evidence.search_coverage.complete is True
    assert evidence.deterministic_assessment.red_eligible is False
    assert InteractionLevel.RED not in evidence.deterministic_assessment.allowed_levels


@pytest.mark.parametrize("route", ["nasal", "ophthalmic", "topical"])
def test_route_qualified_alias_table_fails_closed(tmp_path, route):
    snapshot = _write_snapshot(tmp_path / "snapshot")
    _write_aliases(snapshot, source="Alpha", target=f"Alpha ({route})")
    outcome = DDInterSource(snapshot).search_pair("Alpha", "Beta", checked_at=CHECKED_AT)
    assert outcome.search_result.result is SearchResultStatus.FAILED
    assert outcome.complete is False
    assert outcome.documents == []


def test_unreviewed_spelling_and_route_names_remain_inconclusive(tmp_path):
    snapshot = _write_snapshot(
        tmp_path / "snapshot",
        rows=(
            ("DDInter1", "Ethinylestradiol", "DDInter2", "Beta", "Minor"),
            ("DDInter3", "Xylometazoline (nasal)", "DDInter2", "Beta", "Moderate"),
        ),
    )
    source = DDInterSource(snapshot)
    for name in ("ethinyl estradiol", "xylometazoline"):
        outcome = source.search_pair(name, "Beta", checked_at=CHECKED_AT)
        assert outcome.search_result.result is SearchResultStatus.INCONCLUSIVE
    _write_aliases(snapshot)
    outcome = DDInterSource(snapshot).search_pair("xylometazoline", "Beta", checked_at=CHECKED_AT)
    assert outcome.search_result.result is SearchResultStatus.INCONCLUSIVE


@pytest.mark.parametrize("ambiguous", [False, True])
def test_missing_or_ambiguous_alias_target_fails_closed(tmp_path, ambiguous):
    rows = (
        BASE_ROWS
        if not ambiguous
        else (
            ("DDInter1", "Ethinylestradiol", "DDInter2", "Beta", "Minor"),
            ("DDInter3", "Ethinylestradiol", "DDInter2", "Beta", "Minor"),
        )
    )
    snapshot = _write_snapshot(tmp_path / "snapshot", rows=rows)
    _write_aliases(snapshot)
    outcome = DDInterSource(snapshot).search_pair(
        "ethinyl estradiol", "Beta", checked_at=CHECKED_AT
    )
    assert outcome.search_result.result is SearchResultStatus.FAILED
    assert outcome.documents == []


def test_native_ambiguity_is_not_overridden_by_alias(tmp_path):
    snapshot = _write_snapshot(
        tmp_path / "snapshot",
        rows=(
            ("DDInter1", "ethinyl estradiol", "DDInter2", "Beta", "Minor"),
            ("DDInter3", "ethinyl estradiol", "DDInter2", "Beta", "Minor"),
            ("DDInter4", "Ethinylestradiol", "DDInter2", "Beta", "Minor"),
        ),
    )
    _write_aliases(snapshot)
    outcome = DDInterSource(snapshot).search_pair(
        "ethinyl estradiol", "Beta", checked_at=CHECKED_AT
    )
    assert outcome.search_result.result is SearchResultStatus.INCONCLUSIVE
    assert outcome.complete is False


def test_aliases_cannot_turn_one_ingredient_into_a_negative_pair(tmp_path):
    snapshot = _write_snapshot(
        tmp_path / "snapshot", rows=(("DDInter1", "Ethinylestradiol", "DDInter2", "Beta", "Minor"),)
    )
    _write_aliases(snapshot)
    with pytest.raises(ValueError, match="different after alias"):
        DDInterSource(snapshot).search_pair("ethinyl estradiol", "Ethinylestradiol")


def _write_route_alias(root):
    table = {
        "schema_version": 1,
        "usage_mode": "noncommercial_mvp",
        "scope": "runtime_and_coverage",
        "aliases": [
            {
                "kind": "route",
                "catalog_ingredient": "xylometazoline",
                "ddinter_name": "Xylometazoline (nasal)",
                "route": "nasal",
                "identity_source_id": "rxnorm.rxcui.39841",
                "identity_source_url": "https://rxnav.nlm.nih.gov/REST/rxcui/39841/properties.json",
                "verification": "Synthetic route fixture; no clinical reference data.",
                "reviewed_on": "2026-09-29",
            }
        ],
    }
    (root / DDINTER_ALIASES_FILENAME).write_text(json.dumps(table), encoding="utf-8")


def test_single_route_alias_uses_that_route_record(tmp_path):
    snapshot = _write_snapshot(
        tmp_path / "snapshot",
        rows=(("DDInter1", "Xylometazoline (nasal)", "DDInter2", "Beta", "Moderate"),),
    )
    _write_route_alias(snapshot)
    outcome = DDInterSource(snapshot).search_pair("xylometazoline", "Beta", checked_at=CHECKED_AT)
    assert outcome.search_result.result is SearchResultStatus.INTERACTION_FOUND
    assert outcome.documents[0].applies_to == ["xylometazoline", "Beta"]
    assert "Xylometazoline (nasal)" in outcome.documents[0].title
    assert "xylometazoline -> Xylometazoline (nasal)" in outcome.search_result.details


@pytest.mark.parametrize(
    "extra_name",
    # D-019: a second route or an unqualified record makes the route ambiguous.
    ["Xylometazoline (ophthalmic)", "Xylometazoline"],
)
def test_route_alias_fails_closed_unless_the_route_is_unique(tmp_path, extra_name):
    snapshot = _write_snapshot(
        tmp_path / "snapshot",
        rows=(
            ("DDInter1", "Xylometazoline (nasal)", "DDInter2", "Beta", "Moderate"),
            ("DDInter3", extra_name, "DDInter2", "Beta", "Minor"),
        ),
    )
    _write_route_alias(snapshot)
    outcome = DDInterSource(snapshot).search_pair("xylometazoline", "Beta", checked_at=CHECKED_AT)
    assert outcome.search_result.result is SearchResultStatus.FAILED
    assert outcome.documents == []
