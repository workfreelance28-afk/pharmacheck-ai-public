from __future__ import annotations

import json

import pytest

from scripts.apply_rxnorm_verifications import apply_verifications


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _accepted(russian_name="метформин", source_name="metformin", rxcui="6809"):
    return {
        "russian_name": russian_name,
        "source_name": source_name,
        "source_id": f"rxnorm.rxcui.{rxcui}",
        "source_url": f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/properties.json",
    }


def test_merges_accepted_mapping_and_resolves_prior_network_error(tmp_path):
    mappings = tmp_path / "mappings.json"
    first = tmp_path / "first.json"
    retry = tmp_path / "retry.json"
    _write_json(
        mappings,
        [
            {
                "russian_name": "Ибупрофен",
                "source_name": "ibuprofen",
                "source_id": "rxnorm.rxcui.5640",
                "source_url": "https://rxnav.nlm.nih.gov/REST/rxcui/5640/properties.json",
            }
        ],
    )
    _write_json(
        first,
        {
            "verified_against": "RxNorm API",
            "accepted": [],
            "errors": [{"russian_name": "метформин", "error": "network_error"}],
        },
    )
    _write_json(
        retry,
        {
            "verified_against": "RxNorm API",
            "accepted": [_accepted()],
            "errors": [],
        },
    )

    result = apply_verifications(mappings, [first, retry])

    assert result == {
        "existing_count": 1,
        "accepted_unique_count": 1,
        "added_count": 1,
        "final_count": 2,
    }
    assert json.loads(mappings.read_text(encoding="utf-8"))[1]["russian_name"] == "Метформин"


def test_refuses_unresolved_network_error(tmp_path):
    mappings = tmp_path / "mappings.json"
    report = tmp_path / "report.json"
    _write_json(mappings, [_accepted("Ибупрофен", "ibuprofen", "5640")])
    _write_json(
        report,
        {
            "verified_against": "RxNorm API",
            "accepted": [],
            "errors": [{"russian_name": "метформин", "error": "network_error"}],
        },
    )

    with pytest.raises(ValueError, match="unresolved API errors"):
        apply_verifications(mappings, [report])


def test_normalizes_official_rxnorm_name_to_catalog_case(tmp_path):
    mappings = tmp_path / "mappings.json"
    report = tmp_path / "report.json"
    _write_json(mappings, [_accepted("Ибупрофен", "ibuprofen", "5640")])
    _write_json(
        report,
        {
            "verified_against": "RxNorm API",
            "accepted": [_accepted("витамин е", "vitamin E", "11256")],
            "errors": [],
        },
    )

    apply_verifications(mappings, [report])

    saved = json.loads(mappings.read_text(encoding="utf-8"))
    assert saved[1]["source_name"] == "vitamin e"
