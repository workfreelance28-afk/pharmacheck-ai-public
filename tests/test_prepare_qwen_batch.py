from __future__ import annotations

import json

import pytest

from scripts.prepare_qwen_batch import prepare_batch


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_batch_excludes_verified_and_previously_processed_names(tmp_path):
    unmapped = tmp_path / "unmapped.json"
    mappings = tmp_path / "mappings.json"
    previous = tmp_path / "input_001.json"
    _write_json(
        unmapped,
        {
            "schema_version": 1,
            "source_id": "grls.export.20260807",
            "items": [
                {"russian_name": "Первое", "live_row_count": 10},
                {"russian_name": "Второе", "live_row_count": 9},
                {"russian_name": "Третье", "live_row_count": 8},
                {"russian_name": "я" * 201, "live_row_count": 8},
                {"russian_name": "Четвёртое", "live_row_count": 7},
            ],
        },
    )
    _write_json(
        mappings,
        [
            {
                "russian_name": "Первое",
                "source_name": "first",
                "source_id": "rxnorm.rxcui.1",
                "source_url": "https://rxnav.nlm.nih.gov/REST/rxcui/1/properties.json",
            }
        ],
    )
    _write_json(
        previous,
        {
            "schema_version": 1,
            "source_id": "grls.export.20260807",
            "batch_id": "001",
            "purpose": "untrusted_mapping_candidates_only",
            "russian_names": ["второе"],
        },
    )

    batch = prepare_batch(
        unmapped_path=unmapped,
        mappings_path=mappings,
        previous_batch_paths=[previous],
        batch_id="002",
        limit=2,
    )

    assert batch["russian_names"] == ["Третье", "Четвёртое"]
    assert batch["batch_id"] == "002"


def test_batch_limit_is_bounded(tmp_path):
    with pytest.raises(ValueError, match="between 1 and 100"):
        prepare_batch(
            unmapped_path=tmp_path / "unused",
            mappings_path=tmp_path / "unused",
            previous_batch_paths=[],
            batch_id="002",
            limit=101,
        )
