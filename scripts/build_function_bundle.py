"""Build a deterministic Yandex Cloud Functions deployment archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from pydantic import ValidationError

from pharmacheck.drug_recognition import DrugCatalogEntry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "dist" / "pharmacheck-function.zip"
DEFAULT_DEMO_OUTPUT = PROJECT_ROOT / "dist" / "pharmacheck-function-demo.zip"

USAGE_MODE_NONCOMMERCIAL_MVP = "noncommercial_mvp"
USAGE_MODE_OFFICIAL_LABELS_DEMO = "official_labels_demo"
SUPPORTED_USAGE_MODES = frozenset(
    {USAGE_MODE_NONCOMMERCIAL_MVP, USAGE_MODE_OFFICIAL_LABELS_DEMO}
)
DEPLOYMENT_MANIFEST_NAME = "deployment_manifest.json"
INGREDIENT_NAMES_FILENAME = "ingredient_mappings.json"
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


class BundleBuildError(RuntimeError):
    """A safe build failure that prevents an invalid production archive."""


def validate_production_catalog(
    path: Path,
    manifest_path: Path | None = None,
) -> list[DrugCatalogEntry]:
    if not path.is_file():
        raise BundleBuildError(
            "Production drug catalog is missing. Provide --catalog with verified source IDs."
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("catalog root must be an array")
        entries = [DrugCatalogEntry.model_validate(item) for item in raw]
    except (OSError, UnicodeError, ValueError, ValidationError) as exc:
        raise BundleBuildError("Production drug catalog is invalid.") from exc
    if not entries:
        raise BundleBuildError("Production drug catalog is empty.")
    if any(
        entry.source_id.casefold().startswith("synthetic")
        or entry.product_id.casefold().startswith("test.")
        for entry in entries
    ):
        raise BundleBuildError("Synthetic or test catalog entries are forbidden in production.")
    manifest = manifest_path or path.with_suffix(".manifest.json")
    try:
        raw_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise BundleBuildError("Production drug catalog manifest is missing or invalid.") from exc
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if raw_manifest.get("catalog_sha256") != digest:
        raise BundleBuildError("Production drug catalog checksum does not match its manifest.")
    if raw_manifest.get("catalog_entries") != len(entries):
        raise BundleBuildError("Production drug catalog count does not match its manifest.")
    return entries


def build_bundle(
    *,
    catalog_path: Path,
    output_path: Path | None = None,
    usage_mode: str = USAGE_MODE_NONCOMMERCIAL_MVP,
) -> Path:
    if usage_mode not in SUPPORTED_USAGE_MODES:
        raise BundleBuildError(
            "usage_mode must be one of: " + ", ".join(sorted(SUPPORTED_USAGE_MODES))
        )
    include_ddinter = usage_mode == USAGE_MODE_NONCOMMERCIAL_MVP
    if output_path is None:
        output_path = DEFAULT_OUTPUT if include_ddinter else DEFAULT_DEMO_OUTPUT

    manifest_path = catalog_path.with_suffix(".manifest.json")
    validate_production_catalog(catalog_path, manifest_path)
    required = [
        PROJECT_ROOT / "index.py",
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "prompts" / "system_prompt_v1.txt",
        PROJECT_ROOT / "src" / "pharmacheck",
    ]
    if include_ddinter:
        required.append(PROJECT_ROOT / "data" / "ddinter")
        required.append(PROJECT_ROOT / "data" / INGREDIENT_NAMES_FILENAME)
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in required if not path.exists()]
    if missing:
        raise BundleBuildError(f"Required deployment files are missing: {', '.join(missing)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pharmacheck-bundle-") as temporary:
        staging = Path(temporary)
        shutil.copy2(PROJECT_ROOT / "index.py", staging / "index.py")
        shutil.copy2(PROJECT_ROOT / "requirements.txt", staging / "requirements.txt")
        ignore_patterns = ["__pycache__", "*.pyc"]
        if not include_ddinter:
            ignore_patterns.append("ddinter.py")
        shutil.copytree(
            PROJECT_ROOT / "src" / "pharmacheck",
            staging / "pharmacheck",
            ignore=shutil.ignore_patterns(*ignore_patterns),
        )
        shutil.copytree(PROJECT_ROOT / "prompts", staging / "prompts")
        (staging / "data").mkdir(exist_ok=True)
        if include_ddinter:
            shutil.copytree(PROJECT_ROOT / "data" / "ddinter", staging / "data" / "ddinter")
            # Russian display names for the private bot; the frozen demo is unchanged.
            shutil.copy2(
                PROJECT_ROOT / "data" / INGREDIENT_NAMES_FILENAME,
                staging / "data" / INGREDIENT_NAMES_FILENAME,
            )
        shutil.copy2(catalog_path, staging / "data" / "drug_catalog.json")
        shutil.copy2(manifest_path, staging / "data" / "drug_catalog.manifest.json")
        (staging / DEPLOYMENT_MANIFEST_NAME).write_text(
            json.dumps(
                {"schema_version": 1, "usage_mode": usage_mode},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        temporary_zip = staging / "bundle.zip"
        with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(staging.rglob("*")):
                if path.is_file() and path != temporary_zip:
                    name = path.relative_to(staging)
                    if not include_ddinter and (
                        name.parts[:2] == ("data", "ddinter")
                        or name == Path("pharmacheck") / "sources" / "ddinter.py"
                    ):
                        raise BundleBuildError(
                            "DDInter data or code must not be present in an "
                            "official_labels_demo archive"
                        )
                    # Fixed timestamp and mode: the archive depends only on file
                    # contents, not on when it was built (reproducible SHA-256).
                    info = zipfile.ZipInfo(name.as_posix(), date_time=ZIP_TIMESTAMP)
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o644 << 16
                    archive.writestr(info, path.read_bytes())
        shutil.copy2(temporary_zip, output_path)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=sorted(SUPPORTED_USAGE_MODES),
        default=USAGE_MODE_NONCOMMERCIAL_MVP,
    )
    arguments = parser.parse_args()
    try:
        result = build_bundle(
            catalog_path=arguments.catalog,
            output_path=arguments.output,
            usage_mode=arguments.mode,
        )
    except BundleBuildError as exc:
        parser.error(str(exc))
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
