"""Prepare a bounded Qwen candidate batch from the local GRLS unmapped report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from scripts.validate_mapping_candidates import (
    VerifiedIngredientMapping,
    normalize_russian_ingredient,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNMAPPED = PROJECT_ROOT / "data" / "grls_unmapped_ingredients.json"
DEFAULT_MAPPINGS = PROJECT_ROOT / "data" / "ingredient_mappings.json"
DEFAULT_BATCH_DIR = PROJECT_ROOT / "data" / "qwen_batches"


class UnmappedItem(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    russian_name: str = Field(min_length=1)
    live_row_count: int = Field(ge=1)


class UnmappedReport(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    schema_version: Literal[1]
    source_id: str = Field(min_length=1)
    items: list[UnmappedItem]


class QwenBatchInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    schema_version: Literal[1]
    source_id: str = Field(min_length=1)
    batch_id: str = Field(min_length=1, pattern=r"^[0-9]{3}$")
    purpose: Literal["untrusted_mapping_candidates_only"]
    russian_names: list[str] = Field(min_length=1, max_length=100)


def prepare_batch(
    *,
    unmapped_path: Path,
    mappings_path: Path,
    previous_batch_paths: list[Path],
    batch_id: str,
    limit: int = 100,
) -> dict[str, object]:
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    if len(batch_id) != 3 or not batch_id.isdigit():
        raise ValueError("batch_id must contain exactly three digits")
    try:
        report = UnmappedReport.model_validate(_read_json(unmapped_path))
        mappings = [
            VerifiedIngredientMapping.model_validate(item)
            for item in _read_json(mappings_path)
        ]
        previous = [
            QwenBatchInput.model_validate(_read_json(path))
            for path in previous_batch_paths
        ]
    except (OSError, UnicodeError, ValueError, ValidationError) as exc:
        raise ValueError("batch source data is invalid") from exc

    excluded = {
        normalize_russian_ingredient(mapping.russian_name) for mapping in mappings
    }
    excluded.update(
        normalize_russian_ingredient(name)
        for batch in previous
        for name in batch.russian_names
    )
    selected: list[str] = []
    for item in report.items:
        if len(item.russian_name) > 200:
            continue
        normalized = normalize_russian_ingredient(item.russian_name)
        if normalized in excluded:
            continue
        selected.append(item.russian_name)
        excluded.add(normalized)
        if len(selected) == limit:
            break
    if not selected:
        raise ValueError("no new unmapped ingredients are available")

    return QwenBatchInput(
        schema_version=1,
        source_id=report.source_id,
        batch_id=batch_id,
        purpose="untrusted_mapping_candidates_only",
        russian_names=selected,
    ).model_dump()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unmapped", type=Path, default=DEFAULT_UNMAPPED)
    parser.add_argument("--mappings", type=Path, default=DEFAULT_MAPPINGS)
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH_DIR)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    previous = [
        path
        for path in sorted(arguments.batch_dir.glob("input_*.json"))
        if path.resolve() != arguments.output.resolve()
    ]
    try:
        batch = prepare_batch(
            unmapped_path=arguments.unmapped,
            mappings_path=arguments.mappings,
            previous_batch_paths=previous,
            batch_id=arguments.batch_id,
            limit=arguments.limit,
        )
    except ValueError as exc:
        parser.error(str(exc))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(batch, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
