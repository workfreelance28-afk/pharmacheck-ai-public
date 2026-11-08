"""Merge only successfully verified RxNorm mappings into the catalog mapping file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from pharmacheck.drug_recognition import normalize_name
from scripts.import_grls_catalog import IngredientMapping


def apply_verifications(
    mappings_path: Path,
    verification_paths: list[Path],
) -> dict[str, int]:
    """Merge accepted mappings and reject incomplete or contradictory reports."""
    try:
        raw_mappings = json.loads(mappings_path.read_text(encoding="utf-8"))
        mappings = [IngredientMapping.model_validate(item) for item in raw_mappings]
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as exc:
        raise ValueError("mapping file is invalid") from exc

    existing = {normalize_name(item.russian_name): item for item in mappings}
    accepted: dict[str, IngredientMapping] = {}
    errored_names: set[str] = set()

    for report_path in verification_paths:
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("verification report is invalid") from exc
        if report.get("verified_against") != "RxNorm API":
            raise ValueError("verification report has an unexpected authority")
        for error in report.get("errors", []):
            errored_names.add(normalize_name(error["russian_name"]))
        for item in report.get("accepted", []):
            mapping = IngredientMapping.model_validate(
                {
                    **item,
                    "russian_name": _sentence_case(item["russian_name"]),
                    "source_name": item["source_name"].casefold(),
                }
            )
            key = normalize_name(mapping.russian_name)
            previous = accepted.get(key)
            if previous is not None and previous != mapping:
                raise ValueError("verification reports contain conflicting accepted mappings")
            accepted[key] = mapping

    unresolved_errors = errored_names - accepted.keys()
    if unresolved_errors:
        raise ValueError("verification reports contain unresolved API errors")

    added = 0
    for key in sorted(accepted):
        mapping = accepted[key]
        previous = existing.get(key)
        if previous is not None:
            if previous.source_name != mapping.source_name:
                raise ValueError("verified mapping conflicts with an existing mapping")
            continue
        mappings.append(mapping)
        existing[key] = mapping
        added += 1

    mappings_path.write_text(
        json.dumps(
            [item.model_dump() for item in mappings],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "existing_count": len(raw_mappings),
        "accepted_unique_count": len(accepted),
        "added_count": added,
        "final_count": len(mappings),
    }


def _sentence_case(value: str) -> str:
    return value[:1].upper() + value[1:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mappings", type=Path, required=True)
    parser.add_argument("--verification", type=Path, action="append", required=True)
    arguments = parser.parse_args()
    try:
        result = apply_verifications(arguments.mappings, arguments.verification)
    except (ValidationError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
