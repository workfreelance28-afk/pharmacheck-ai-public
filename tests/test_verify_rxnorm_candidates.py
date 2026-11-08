from __future__ import annotations

import json

from pharmacheck.sources.rxnorm import (
    RxNormConcept,
    RxNormResolution,
    RxNormSourceError,
    RxNormStatus,
)
from scripts.verify_rxnorm_candidates import verify_candidates


def _write_batch(path, candidates):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_by": "qwen",
                "candidates": candidates,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _candidate(russian_name, source_name):
    return {
        "russian_name": russian_name,
        "source_name_candidate": source_name,
        "official_source_urls": ["https://rxnav.nlm.nih.gov/"],
    }


class FakeClient:
    def __init__(self, resolutions, concepts, related=None):
        self.resolutions = iter(resolutions)
        self.concepts = concepts
        self.related = related or {}

    def resolve(self, ingredient):
        result = next(self.resolutions)
        if isinstance(result, Exception):
            raise result
        return result

    def get_concept(self, rxcui):
        return self.concepts[rxcui]

    def get_related_ingredient(self, rxcui):
        return self.related.get(rxcui)


def test_only_exact_active_english_ingredient_is_accepted(tmp_path):
    path = tmp_path / "candidates.json"
    _write_batch(
        path,
        [
            _candidate("Метформин", "metformin"),
            _candidate("Ацетилсалициловая кислота", "acetylsalicylic acid"),
            _candidate("Торговое имя", "some product"),
        ],
    )
    client = FakeClient(
        [
            RxNormResolution("metformin", RxNormStatus.MATCHED, "6809"),
            RxNormResolution("acetylsalicylic acid", RxNormStatus.MATCHED, "1191"),
            RxNormResolution("some product", RxNormStatus.MATCHED, "99"),
        ],
        {
            "6809": RxNormConcept("6809", "metformin", "IN", "ENG", "N"),
            "1191": RxNormConcept("1191", "aspirin", "IN", "ENG", "N"),
            "99": RxNormConcept("99", "some product", "SCD", "ENG", "N"),
        },
    )

    report = verify_candidates(path, client=client, delay_seconds=0)

    assert report["accepted_count"] == 1
    assert report["accepted"][0]["source_id"] == "rxnorm.rxcui.6809"
    assert [item["reason"] for item in report["rejected"]] == [
        "preferred_name_mismatch",
        "not_ingredient",
    ]


def test_not_found_and_api_errors_are_kept_separate(tmp_path):
    path = tmp_path / "candidates.json"
    _write_batch(
        path,
        [
            _candidate("Не найдено", "not found"),
            _candidate("Ошибка", "network error"),
        ],
    )
    client = FakeClient(
        [
            RxNormResolution("not found", RxNormStatus.NOT_FOUND),
            RxNormSourceError("network_error"),
        ],
        {},
    )

    report = verify_candidates(path, client=client, delay_seconds=0)

    assert report["rejected_count"] == 1
    assert report["rejected"][0]["reason"] == "not_found"
    assert report["error_count"] == 1
    assert report["errors"][0]["error"] == "network_error"


def test_precise_ingredient_is_accepted_via_its_parent_ingredient(tmp_path):
    path = tmp_path / "candidates.json"
    _write_batch(path, [_candidate("Ипратропия бромид", "ipratropium bromide")])
    client = FakeClient(
        [RxNormResolution("ipratropium bromide", RxNormStatus.MATCHED, "203212")],
        {"203212": RxNormConcept("203212", "ipratropium bromide", "PIN", "ENG", "N")},
        related={"203212": RxNormConcept("7213", "ipratropium", "IN", "ENG", "N")},
    )

    report = verify_candidates(path, client=client, delay_seconds=0)

    assert report["rejected_count"] == 0
    assert report["accepted_count"] == 1
    assert report["accepted"][0] == {
        "russian_name": "Ипратропия бромид",
        "source_name": "ipratropium",
        "source_id": "rxnorm.rxcui.7213",
        "source_url": "https://rxnav.nlm.nih.gov/REST/rxcui/7213/properties.json",
    }
    assert report["resolved_from_precise_ingredient"] == [
        {
            "russian_name": "Ипратропия бромид",
            "source_name_candidate": "ipratropium bromide",
            "precise_ingredient_name": "ipratropium bromide",
            "precise_ingredient_rxcui": "203212",
            "resolved_ingredient_name": "ipratropium",
            "resolved_ingredient_rxcui": "7213",
        }
    ]


def test_precise_ingredient_without_a_parent_is_still_rejected(tmp_path):
    path = tmp_path / "candidates.json"
    _write_batch(path, [_candidate("Неизвестное", "some salt form")])
    client = FakeClient(
        [RxNormResolution("some salt form", RxNormStatus.MATCHED, "1")],
        {"1": RxNormConcept("1", "some salt form", "PIN", "ENG", "N")},
        related={},
    )

    report = verify_candidates(path, client=client, delay_seconds=0)

    assert report["accepted_count"] == 0
    assert report["rejected_count"] == 1
    assert report["rejected"][0]["reason"] == "not_ingredient"
    assert report["resolved_from_precise_ingredient"] == []
