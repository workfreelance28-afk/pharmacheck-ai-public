#!/usr/bin/env python3
"""Download a verified non-commercial DDInter 2.0 CSV snapshot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

BASE_URL = "https://ddinter2.scbdd.com/static/media/download"
FILENAMES = (
    "ddinter_downloads_code_A.csv",
    "ddinter_downloads_code_B.csv",
    "ddinter_downloads_code_D.csv",
    "ddinter_downloads_code_H.csv",
    "ddinter_downloads_code_L.csv",
    "ddinter_downloads_code_P.csv",
    "ddinter_downloads_code_R.csv",
    "ddinter_downloads_code_V.csv",
)
EXPECTED_COLUMNS = (
    "DDInterID_A",
    "Drug_A",
    "DDInterID_B",
    "Drug_B",
    "Level",
)
MAX_FILE_BYTES = 12_000_000


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download DDInter 2.0 for non-commercial MVP use only."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/ddinter"),
        help="Snapshot directory (default: data/ddinter)",
    )
    parser.add_argument("--timeout", type=float, default=45.0)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(
        prefix="ddinter-sync-", dir=destination
    ) as temporary:
        temporary_dir = Path(temporary)
        for filename in FILENAMES:
            url = f"{BASE_URL}/{filename}"
            path = temporary_dir / filename
            _download(url, path, timeout=args.timeout)
            _validate_csv(path)
            payload = path.read_bytes()
            files.append(
                {
                    "name": filename,
                    "url": url,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }
            )

        downloaded_at = datetime.now(UTC)
        manifest = {
            "schema_version": 1,
            "dataset_name": "DDInter 2.0",
            "snapshot_version": f"download-{downloaded_at.date().isoformat()}",
            "release_date": None,
            "downloaded_at": downloaded_at.isoformat(),
            "license_id": "CC-BY-NC-SA-4.0",
            "license_url": (
                "https://creativecommons.org/licenses/by-nc-sa/4.0/"
            ),
            "terms_url": "https://ddinter2.scbdd.com/terms/",
            "usage_mode": "noncommercial_mvp",
            "files": files,
        }
        manifest_path = temporary_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        for filename in FILENAMES:
            os.replace(temporary_dir / filename, destination / filename)
        os.replace(manifest_path, destination / "manifest.json")

    total_bytes = sum(int(item["size_bytes"]) for item in files)
    print(
        f"DDInter snapshot ready: {len(files)} files, "
        f"{total_bytes} bytes, non-commercial use only."
    )
    return 0


def _download(url: str, destination: Path, *, timeout: float) -> None:
    request = Request(
        url,
        headers={
            "Accept": "text/csv",
            "User-Agent": "PharmaCheckAI-DDInterSync/0.1",
        },
        method="GET",
    )
    with urlopen(request, timeout=timeout) as response:
        payload = response.read(MAX_FILE_BYTES + 1)
    if len(payload) > MAX_FILE_BYTES:
        raise RuntimeError(f"DDInter file exceeds size limit: {destination.name}")
    destination.write_bytes(payload)


def _validate_csv(path: Path) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = tuple(next(reader, ()))
        first_row = tuple(next(reader, ()))
    if header != EXPECTED_COLUMNS or len(first_row) != len(EXPECTED_COLUMNS):
        raise RuntimeError(f"Unexpected DDInter CSV schema: {path.name}")


if __name__ == "__main__":
    raise SystemExit(main())
