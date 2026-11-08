from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from scripts.build_function_bundle import (
    DEPLOYMENT_MANIFEST_NAME,
    USAGE_MODE_NONCOMMERCIAL_MVP,
    USAGE_MODE_OFFICIAL_LABELS_DEMO,
    BundleBuildError,
    build_bundle,
)


def _catalog(path, *, synthetic=False):
    path.write_text(
        json.dumps(
            [
                {
                    "product_id": "test.alpha" if synthetic else "drug.alpha",
                    "trade_name": "Альфа",
                    "active_ingredients": ["alpha"],
                    "source_id": "synthetic.catalog" if synthetic else "grls.catalog.alpha",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "catalog_entries": 1,
                "catalog_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )


def test_bundle_rejects_synthetic_catalog(tmp_path):
    catalog = tmp_path / "catalog.json"
    _catalog(catalog, synthetic=True)
    with pytest.raises(BundleBuildError):
        build_bundle(catalog_path=catalog, output_path=tmp_path / "bundle.zip")


@pytest.mark.skipif(
    not (Path(__file__).parents[1] / "data/ddinter/manifest.json").is_file()
    or not (Path(__file__).parents[1] / "data/ingredient_mappings.json").is_file(),
    reason="Private noncommercial deployment inputs are not distributed",
)
def test_bundle_contains_runtime_data_and_no_test_files(tmp_path):
    catalog = tmp_path / "catalog.json"
    output = tmp_path / "bundle.zip"
    _catalog(catalog)

    build_bundle(catalog_path=catalog, output_path=output)

    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert "index.py" in names
        assert "requirements.txt" in names
        assert "pharmacheck/cloud_runtime.py" in names
        assert "prompts/system_prompt_v1.txt" in names
        assert "data/drug_catalog.json" in names
        assert "data/drug_catalog.manifest.json" in names
        assert "data/ddinter/manifest.json" in names
        assert "data/ddinter/name_aliases.json" in names
        assert "data/ingredient_mappings.json" in names
        assert not any("__pycache__" in name or name.startswith("tests/") for name in names)
        assert DEPLOYMENT_MANIFEST_NAME in names
        with archive.open(DEPLOYMENT_MANIFEST_NAME) as handle:
            manifest = json.load(handle)
        assert manifest == {
            "schema_version": 1,
            "usage_mode": USAGE_MODE_NONCOMMERCIAL_MVP,
        }


def test_demo_bundle_excludes_ddinter_entirely(tmp_path):
    catalog = tmp_path / "catalog.json"
    output = tmp_path / "demo-bundle.zip"
    _catalog(catalog)

    build_bundle(
        catalog_path=catalog,
        output_path=output,
        usage_mode=USAGE_MODE_OFFICIAL_LABELS_DEMO,
    )

    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert "index.py" in names
        assert "pharmacheck/cloud_runtime.py" in names
        assert "data/drug_catalog.json" in names
        assert not any(name.startswith("data/ddinter") for name in names)
        assert "data/ingredient_mappings.json" not in names
        assert "pharmacheck/sources/ddinter.py" not in names
        assert "pharmacheck/sources/openfda.py" in names
        assert DEPLOYMENT_MANIFEST_NAME in names
        with archive.open(DEPLOYMENT_MANIFEST_NAME) as handle:
            manifest = json.load(handle)
        assert manifest == {
            "schema_version": 1,
            "usage_mode": USAGE_MODE_OFFICIAL_LABELS_DEMO,
        }


def _assert_cold_import_succeeds(archive_path, tmp_path, *, label):
    extracted = tmp_path / f"extracted-{label}"
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extracted)

    result = subprocess.run(
        [sys.executable, "-c", "import index"],
        cwd=extracted,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"cold import of {label} archive failed:\n{result.stderr}"
    )


def test_demo_bundle_cold_imports_without_ddinter(tmp_path):
    catalog = tmp_path / "catalog.json"
    output = tmp_path / "demo-bundle.zip"
    _catalog(catalog)

    build_bundle(
        catalog_path=catalog,
        output_path=output,
        usage_mode=USAGE_MODE_OFFICIAL_LABELS_DEMO,
    )

    _assert_cold_import_succeeds(output, tmp_path, label="demo")


@pytest.mark.skipif(
    not (Path(__file__).parents[1] / "data/ddinter/manifest.json").is_file()
    or not (Path(__file__).parents[1] / "data/ingredient_mappings.json").is_file(),
    reason="Private noncommercial deployment inputs are not distributed",
)
def test_noncommercial_bundle_cold_imports(tmp_path):
    catalog = tmp_path / "catalog.json"
    output = tmp_path / "bundle.zip"
    _catalog(catalog)

    build_bundle(catalog_path=catalog, output_path=output)

    _assert_cold_import_succeeds(output, tmp_path, label="noncommercial_mvp")


def test_bundle_rejects_unsupported_mode(tmp_path):
    catalog = tmp_path / "catalog.json"
    _catalog(catalog)
    with pytest.raises(BundleBuildError):
        build_bundle(
            catalog_path=catalog,
            output_path=tmp_path / "bundle.zip",
            usage_mode="commercial_unrestricted",
        )


@pytest.mark.skipif(
    not (Path(__file__).parents[1] / "data/ddinter/manifest.json").is_file()
    or not (Path(__file__).parents[1] / "data/ingredient_mappings.json").is_file(),
    reason="Private noncommercial deployment inputs are not distributed",
)
def test_bundle_is_reproducible_regardless_of_build_time(tmp_path):
    # Entry times are fixed, so files written during the build (such as the
    # deployment manifest) cannot change the archive hash between builds.
    catalog = tmp_path / "catalog.json"
    _catalog(catalog)
    first = build_bundle(catalog_path=catalog, output_path=tmp_path / "first.zip")
    second = build_bundle(catalog_path=catalog, output_path=tmp_path / "second.zip")
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        assert {info.date_time for info in archive.infolist()} == {(1980, 1, 1, 0, 0, 0)}
