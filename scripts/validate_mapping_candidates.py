"""Validate untrusted ingredient-mapping candidates before human review."""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAPPINGS = PROJECT_ROOT / "data" / "ingredient_mappings.json"

TRUSTED_EVIDENCE_HOSTS = {
    "accessdata.fda.gov",
    "dailymed.nlm.nih.gov",
    "extranet.who.int",
    "grls.rosminzdrav.ru",
    "rxnav.nlm.nih.gov",
    "www.accessdata.fda.gov",
}


class CandidateErrorSafe(RuntimeError):
    """Invalid AI-generated candidate data safe to show in CLI output."""


class VerifiedIngredientMapping(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    russian_name: str = Field(min_length=1)
    source_name: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9 .'-]*$")
    source_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.:-]+$")
    source_url: HttpUrl


class MappingCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    russian_name: str = Field(min_length=1, max_length=200)
    source_name_candidate: str = Field(
        min_length=1,
        max_length=200,
        pattern=r"^[a-z0-9][a-z0-9 .'-]*$",
    )
    official_source_urls: list[HttpUrl] = Field(min_length=1, max_length=5)
    evidence_notes: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def evidence_must_use_trusted_hosts(self) -> MappingCandidate:
        hosts = {urlparse(str(url)).hostname for url in self.official_source_urls}
        if None in hosts or not hosts.issubset(TRUSTED_EVIDENCE_HOSTS):
            raise ValueError("official_source_urls contains an untrusted host")
        return self


class CandidateBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    schema_version: Literal[1]
    generated_by: Literal["qwen"]
    candidates: list[MappingCandidate] = Field(min_length=1)

    @model_validator(mode="after")
    def russian_names_must_be_unique(self) -> CandidateBatch:
        names = [
            normalize_russian_ingredient(candidate.russian_name)
            for candidate in self.candidates
        ]
        if len(names) != len(set(names)):
            raise ValueError("candidate batch contains duplicate Russian names")
        return self


def validate_candidates(candidate_path: Path, mappings_path: Path) -> dict[str, object]:
    try:
        raw = json.loads(candidate_path.read_text(encoding="utf-8"))
        batch = CandidateBatch.model_validate(raw)
        verified = _load_verified_names(mappings_path)
    except (OSError, UnicodeError, ValueError) as exc:
        raise CandidateErrorSafe("Candidate or verified mapping file is invalid.") from exc

    existing = []
    new = []
    for candidate in batch.candidates:
        normalized = normalize_russian_ingredient(candidate.russian_name)
        item = {
            "russian_name": candidate.russian_name,
            "source_name_candidate": candidate.source_name_candidate,
        }
        if normalized in verified:
            existing.append(item)
        else:
            new.append(item)
    return {
        "schema_version": batch.schema_version,
        "generated_by": batch.generated_by,
        "candidate_count": len(batch.candidates),
        "already_verified_count": len(existing),
        "new_candidate_count": len(new),
        "new_candidates": new,
    }


def normalize_russian_ingredient(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().replace("ё", "е")
    return " ".join(value.strip().strip("[]").strip().split())


def _load_verified_names(path: Path) -> set[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    mappings = [VerifiedIngredientMapping.model_validate(item) for item in raw]
    names = {normalize_russian_ingredient(mapping.russian_name) for mapping in mappings}
    if not names or len(names) != len(mappings):
        raise ValueError("verified mappings are empty or contain duplicate Russian names")
    return names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--mappings", type=Path, default=DEFAULT_MAPPINGS)
    arguments = parser.parse_args()
    try:
        report = validate_candidates(arguments.candidates, arguments.mappings)
    except CandidateErrorSafe as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
