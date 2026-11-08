"""Verify untrusted Qwen ingredient candidates against official RxNorm data."""

from __future__ import annotations

import argparse
import json
import time
import unicodedata
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from pharmacheck.sources.rxnorm import (
    RXNORM_PROPERTIES_URL,
    RxNormClient,
    RxNormConcept,
    RxNormResolution,
    RxNormSourceError,
    RxNormStatus,
)
from scripts.validate_mapping_candidates import CandidateBatch


class RxNormCandidateClient(Protocol):
    def resolve(self, ingredient: str) -> RxNormResolution: ...

    def get_concept(self, rxcui: str) -> RxNormConcept: ...

    def get_related_ingredient(self, rxcui: str) -> RxNormConcept | None: ...


def verify_candidates(
    candidate_path: Path,
    *,
    client: RxNormCandidateClient | None = None,
    delay_seconds: float = 0.1,
) -> dict[str, object]:
    """Return exact matches, rejected candidates, and transient API errors."""
    if delay_seconds < 0:
        raise ValueError("delay_seconds cannot be negative")
    try:
        raw = json.loads(candidate_path.read_text(encoding="utf-8"))
        batch = CandidateBatch.model_validate(raw)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as exc:
        raise ValueError("candidate file is invalid") from exc

    rxnorm = client or RxNormClient()
    accepted: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    resolved_from_precise_ingredient: list[dict[str, str]] = []

    for index, candidate in enumerate(batch.candidates):
        try:
            resolution = rxnorm.resolve(candidate.source_name_candidate)
            if resolution.status is not RxNormStatus.MATCHED or resolution.rxcui is None:
                rejected.append(
                    _rejection(
                        candidate.russian_name,
                        candidate.source_name_candidate,
                        resolution.status.value,
                    )
                )
                continue

            concept = rxnorm.get_concept(resolution.rxcui)
            reason = _concept_rejection_reason(
                candidate.source_name_candidate,
                concept,
            )
            if reason == "not_ingredient" and concept.tty == "PIN":
                parent = rxnorm.get_related_ingredient(concept.rxcui)
                if (
                    parent is not None
                    and parent.language == "ENG"
                    and parent.suppress == "N"
                ):
                    accepted.append(
                        {
                            "russian_name": candidate.russian_name,
                            "source_name": parent.name,
                            "source_id": f"rxnorm.rxcui.{parent.rxcui}",
                            "source_url": RXNORM_PROPERTIES_URL.format(rxcui=parent.rxcui),
                        }
                    )
                    resolved_from_precise_ingredient.append(
                        {
                            "russian_name": candidate.russian_name,
                            "source_name_candidate": candidate.source_name_candidate,
                            "precise_ingredient_name": concept.name,
                            "precise_ingredient_rxcui": concept.rxcui,
                            "resolved_ingredient_name": parent.name,
                            "resolved_ingredient_rxcui": parent.rxcui,
                        }
                    )
                    continue
            if reason is not None:
                rejected.append(
                    _rejection(
                        candidate.russian_name,
                        candidate.source_name_candidate,
                        reason,
                        official_name=concept.name,
                        rxcui=concept.rxcui,
                    )
                )
                continue

            accepted.append(
                {
                    "russian_name": candidate.russian_name,
                    "source_name": concept.name,
                    "source_id": f"rxnorm.rxcui.{concept.rxcui}",
                    "source_url": RXNORM_PROPERTIES_URL.format(rxcui=concept.rxcui),
                }
            )
        except (RxNormSourceError, ValueError) as exc:
            errors.append(
                {
                    "russian_name": candidate.russian_name,
                    "source_name_candidate": candidate.source_name_candidate,
                    "error": getattr(exc, "code", type(exc).__name__),
                }
            )
        if delay_seconds and index < len(batch.candidates) - 1:
            time.sleep(delay_seconds)

    return {
        "schema_version": 1,
        "verified_against": "RxNorm API",
        "candidate_count": len(batch.candidates),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "error_count": len(errors),
        "accepted": accepted,
        "rejected": rejected,
        "errors": errors,
        "resolved_from_precise_ingredient": resolved_from_precise_ingredient,
    }


def _concept_rejection_reason(
    candidate_name: str,
    concept: RxNormConcept,
) -> str | None:
    if _normalize_name(candidate_name) != _normalize_name(concept.name):
        return "preferred_name_mismatch"
    if concept.tty != "IN":
        return "not_ingredient"
    if concept.language != "ENG":
        return "not_english"
    if concept.suppress != "N":
        return "suppressed"
    return None


def _normalize_name(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _rejection(
    russian_name: str,
    source_name_candidate: str,
    reason: str,
    *,
    official_name: str | None = None,
    rxcui: str | None = None,
) -> dict[str, str]:
    result = {
        "russian_name": russian_name,
        "source_name_candidate": source_name_candidate,
        "reason": reason,
    }
    if official_name is not None:
        result["official_name"] = official_name
    if rxcui is not None:
        result["rxcui"] = rxcui
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delay-seconds", type=float, default=0.1)
    arguments = parser.parse_args()
    try:
        report = verify_candidates(
            arguments.candidates,
            delay_seconds=arguments.delay_seconds,
        )
    except ValueError as exc:
        parser.error(str(exc))
    arguments.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["error_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
