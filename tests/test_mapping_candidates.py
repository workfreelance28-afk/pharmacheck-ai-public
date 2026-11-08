from __future__ import annotations

import json

import pytest

from scripts.validate_mapping_candidates import CandidateErrorSafe, validate_candidates


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _verified_mapping():
    return [
        {
            "russian_name": "Ибупрофен",
            "source_name": "ibuprofen",
            "source_id": "who.inn.ibuprofen",
            "source_url": "https://extranet.who.int/soinn/",
        }
    ]


def test_qwen_candidates_remain_separate_from_verified_mappings(tmp_path):
    candidates = tmp_path / "candidates.json"
    mappings = tmp_path / "mappings.json"
    _write_json(mappings, _verified_mapping())
    _write_json(
        candidates,
        {
            "schema_version": 1,
            "generated_by": "qwen",
            "candidates": [
                {
                    "russian_name": "Ибупрофен",
                    "source_name_candidate": "ibuprofen",
                    "official_source_urls": ["https://extranet.who.int/soinn/"],
                },
                {
                    "russian_name": "Эсциталопрам",
                    "source_name_candidate": "escitalopram",
                    "official_source_urls": ["https://dailymed.nlm.nih.gov/dailymed/"],
                },
            ],
        },
    )

    report = validate_candidates(candidates, mappings)

    assert report["already_verified_count"] == 1
    assert report["new_candidate_count"] == 1
    assert report["new_candidates"] == [
        {
            "russian_name": "Эсциталопрам",
            "source_name_candidate": "escitalopram",
        }
    ]


def test_untrusted_evidence_host_is_rejected(tmp_path):
    candidates = tmp_path / "candidates.json"
    mappings = tmp_path / "mappings.json"
    _write_json(mappings, _verified_mapping())
    _write_json(
        candidates,
        {
            "schema_version": 1,
            "generated_by": "qwen",
            "candidates": [
                {
                    "russian_name": "Эсциталопрам",
                    "source_name_candidate": "escitalopram",
                    "official_source_urls": ["https://example.com/article"],
                }
            ],
        },
    )

    with pytest.raises(CandidateErrorSafe):
        validate_candidates(candidates, mappings)


def test_duplicate_russian_names_are_rejected(tmp_path):
    candidates = tmp_path / "candidates.json"
    mappings = tmp_path / "mappings.json"
    _write_json(mappings, _verified_mapping())
    candidate = {
        "russian_name": "Эсциталопрам",
        "source_name_candidate": "escitalopram",
        "official_source_urls": ["https://rxnav.nlm.nih.gov/"],
    }
    _write_json(
        candidates,
        {
            "schema_version": 1,
            "generated_by": "qwen",
            "candidates": [candidate, candidate],
        },
    )

    with pytest.raises(CandidateErrorSafe):
        validate_candidates(candidates, mappings)
