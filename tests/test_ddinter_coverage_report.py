from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from pharmacheck.sources.ddinter import _normalize_name, load_name_aliases
from scripts.report_ddinter_coverage import build_coverage, format_markdown


def _alias(source="ethinyl estradiol", target="Ethinylestradiol"):
    return {
        "catalog_ingredient": source,
        "ddinter_name": target,
        "identity_source_id": "rxnorm.rxcui.4124",
        "identity_source_url": "https://rxnav.nlm.nih.gov/REST/rxcui/4124/properties.json",
        "verification": "Synthetic test of spelling-only validation.",
        "reviewed_on": "2026-09-28",
    }


def _snapshot(names):
    canonical = {f"DDInter{index}": name for index, name in enumerate(names, start=1)}
    name_to_ids = {}
    for identifier, name in canonical.items():
        name_to_ids.setdefault(_normalize_name(name), set()).add(identifier)
    return SimpleNamespace(canonical_names=canonical, name_to_ids=name_to_ids)


def test_coverage_counts_unique_names_and_keeps_routes_excluded():
    snapshot = _snapshot(
        [
            "Alpha",
            "Alpha (topical)",
            "Ethinylestradiol",
            "Xylometazoline (nasal)",
            "Ambiguous",
            "Ambiguous",
        ]
    )
    report = build_coverage(
        ["Alpha", "Alpha", "ethinyl estradiol", "xylometazoline", "Unlisted", "Ambiguous"],
        snapshot,
        [_alias()],
    )
    assert report["counts"] == {
        "catalog_ingredients": 5,
        "runtime_exact_coverage": 1,
        "reviewed_table_coverage": 2,
        "unresolved_after_table": 3,
        "exact": 1,
        "reviewed_spelling": 1,
        "reviewed_synonym": 0,
        "reviewed_route": 0,
        "route_only": 1,
        "ambiguous_exact": 1,
        "missing": 1,
    }
    rows = {row["catalog_ingredient"]: row for row in report["ingredients"]}
    assert rows["xylometazoline"]["matched_ddinter_id"] is None
    assert (
        rows["xylometazoline"]["excluded_route_candidates"][0]["ddinter_name"]
        == "Xylometazoline (nasal)"
    )
    assert rows["Alpha"]["status"] == "exact"
    assert rows["Alpha"]["excluded_route_candidates"]
    assert rows["Ambiguous"]["status"] == "ambiguous_exact"
    report.update(snapshot_version="synthetic", checked_at="2026-09-28", catalog_entries=6)
    markdown = format_markdown(report)
    assert "Ethinylestradiol" in markdown
    assert "Xylometazoline (nasal)" in markdown
    assert "production меняется только после публикации" in markdown


def test_report_does_not_infer_synonym_or_route_matches():
    report = build_coverage(
        ["aspirin", "xylometazoline"],
        _snapshot(["Acetylsalicylic acid", "Xylometazoline (nasal)"]),
        [],
    )
    assert report["counts"]["reviewed_table_coverage"] == 0
    assert report["counts"]["missing"] == 1
    assert report["counts"]["route_only"] == 1


@pytest.mark.parametrize("target", ["Different substance", "Ethinylestradiol (nasal)"])
def test_alias_table_rejects_substance_or_route_substitution(tmp_path, target):
    path = tmp_path / "aliases.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "usage_mode": "noncommercial_mvp",
                "scope": "runtime_and_coverage",
                "aliases": [_alias(target=target)],
            }
        )
    )
    with pytest.raises(ValueError, match="whitespace/case|route-qualified"):
        load_name_aliases(path)


def test_alias_table_rejects_duplicate_reviewed_sources(tmp_path):
    path = tmp_path / "aliases.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "usage_mode": "noncommercial_mvp",
                "scope": "runtime_and_coverage",
                "aliases": [_alias(), _alias()],
            }
        )
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_name_aliases(path)


def test_report_rejects_missing_reviewed_target_or_catalog_source():
    with pytest.raises(ValueError, match="target is missing"):
        build_coverage(["ethinyl estradiol"], _snapshot(["Alpha"]), [_alias()])
    with pytest.raises(ValueError, match="source is absent"):
        build_coverage(["Alpha"], _snapshot(["Ethinylestradiol"]), [_alias()])


def _table(tmp_path, aliases):
    path = tmp_path / "aliases.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "usage_mode": "noncommercial_mvp",
                "scope": "runtime_and_coverage",
                "aliases": aliases,
            }
        )
    )
    return path


def _synonym(**overrides):
    alias = {
        **_alias(source="aspirin", target="Acetylsalicylic acid"),
        "kind": "synonym",
        "relation": "same_ingredient",
        "catalog_rxcui": "1191",
        "ddinter_name_rxcui": "1191",
    }
    alias.update(overrides)
    return alias


def _route(**overrides):
    alias = {
        **_alias(source="xylometazoline", target="Xylometazoline (nasal)"),
        "kind": "route",
        "route": "nasal",
    }
    alias.update(overrides)
    return alias


def test_reviewed_synonym_and_route_aliases_are_counted_by_kind(tmp_path):
    aliases = load_name_aliases(_table(tmp_path, [_alias(), _synonym(), _route()]))
    report = build_coverage(
        ["ethinyl estradiol", "aspirin", "xylometazoline"],
        _snapshot(["Ethinylestradiol", "Acetylsalicylic acid", "Xylometazoline (nasal)"]),
        aliases,
    )
    counts = report["counts"]
    assert (counts["reviewed_spelling"], counts["reviewed_synonym"]) == (1, 1)
    assert (counts["reviewed_route"], counts["route_only"]) == (1, 0)
    assert counts["reviewed_table_coverage"] == 3


@pytest.mark.parametrize(
    "overrides",
    [
        # D-020: different RxCUIs are different substances (povidone-iodine/iodine).
        {"catalog_rxcui": "8611", "ddinter_name_rxcui": "5933"},
        {"relation": "looks_similar"},
        {"catalog_rxcui": "aspirin"},
        {"relation": "salt_or_form_of", "ddinter_name_rxcui": "21216"},
        {
            "relation": "salt_or_form_of",
            "ddinter_name_rxcui": "21216",
            "ddinter_ingredient_rxcui": "9999",
        },
        {"ddinter_name": "Acetylsalicylic acid (topical)"},
    ],
)
def test_synonym_alias_requires_rxnorm_identity(tmp_path, overrides):
    with pytest.raises(ValueError):
        load_name_aliases(_table(tmp_path, [_synonym(**overrides)]))


def test_salt_synonym_resolving_to_catalog_ingredient_is_accepted(tmp_path):
    alias = _synonym(
        catalog_ingredient="clavulanate",
        ddinter_name="Clavulanic acid",
        relation="salt_or_form_of",
        catalog_rxcui="48203",
        ddinter_name_rxcui="21216",
        ddinter_ingredient_rxcui="48203",
    )
    assert load_name_aliases(_table(tmp_path, [alias]))[0]["kind"] == "synonym"


@pytest.mark.parametrize(
    "overrides",
    [
        {"ddinter_name": "Oxymetazoline (nasal)"},
        {"ddinter_name": "Xylometazoline nasal spray"},
        {"route": "ophthalmic"},
        {"catalog_ingredient": "xylometazoline (nasal)"},
    ],
)
def test_route_alias_target_must_be_the_ingredient_plus_its_route(tmp_path, overrides):
    with pytest.raises(ValueError):
        load_name_aliases(_table(tmp_path, [_route(**overrides)]))


def test_unknown_alias_kind_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="kind"):
        load_name_aliases(_table(tmp_path, [{**_alias(), "kind": "guess"}]))
