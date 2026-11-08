"""Run the portable local quality gate before an agent handoff."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATUS_MAX_LINES = 100  # D-017: STATUS.md is rewritten each step, never a growing log
CATALOG_MANIFEST = PROJECT_ROOT / "data" / "drug_catalog.manifest.json"
# The runtime rejects a catalog older than 45 days (cloud_runtime.py), which
# silently took the bot down in September 2026. Warn well before that.
CATALOG_WARN_DAYS = 30
CATALOG_MAX_DAYS = 45


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _python() -> str:
    configured = os.environ.get("PROJECT_PYTHON")
    if configured:
        return configured
    project_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    return str(project_python) if project_python.is_file() else sys.executable


def _ruff() -> str:
    configured = os.environ.get("RUFF_BIN")
    if configured:
        return configured
    project_ruff = PROJECT_ROOT / ".venv" / "bin" / "ruff"
    if project_ruff.is_file():
        return str(project_ruff)
    discovered = shutil.which("ruff")
    if discovered:
        return discovered
    cursor_root = Path.home() / ".cursor" / "extensions"
    candidates = sorted(
        cursor_root.glob("charliermarsh.ruff-*/bundled/libs/bin/ruff"),
        reverse=True,
    )
    if candidates:
        return str(candidates[0])
    raise RuntimeError(
        "Ruff was not found. Install Ruff or set the RUFF_BIN environment variable."
    )


def check_status_length(path: Path, max_lines: int = STATUS_MAX_LINES) -> None:
    """Fail when STATUS.md is missing or has grown past its line limit."""
    if not path.is_file():
        raise RuntimeError(f"{path.name} is missing; it is required by AGENTS.md (D-017).")
    line_count = len(path.read_text(encoding="utf-8").splitlines())
    if line_count > max_lines:
        raise RuntimeError(
            f"{path.name} has {line_count} lines (limit {max_lines}). "
            "Rewrite it instead of appending; history belongs in commit messages."
        )


def check_catalog_age(
    manifest_path: Path,
    *,
    today: date | None = None,
    warn_days: int = CATALOG_WARN_DAYS,
    max_days: int = CATALOG_MAX_DAYS,
) -> str:
    """Return a status line; fail once the runtime would reject the catalog."""
    export_date = date.fromisoformat(
        json.loads(manifest_path.read_text(encoding="utf-8"))["export_date"]
    )
    age = ((today or datetime.now(UTC).date()) - export_date).days
    expires_on = (export_date + timedelta(days=max_days)).isoformat()
    if age > max_days:
        raise RuntimeError(
            f"Drug catalog export {export_date} is {age} days old; the runtime "
            f"rejects catalogs older than {max_days} days. Refresh it from GRLS."
        )
    if age >= warn_days:
        return (
            f"WARNING: drug catalog export {export_date} is {age} days old; "
            f"production stops accepting it after {expires_on}. Refresh it from GRLS."
        )
    return f"Drug catalog export {export_date} is {age} days old (valid through {expires_on})."


def main() -> int:
    if (PROJECT_ROOT / "STATUS.md").is_file():
        check_status_length(PROJECT_ROOT / "STATUS.md")
    if CATALOG_MANIFEST.is_file():
        print("+ " + check_catalog_age(CATALOG_MANIFEST), flush=True)
    _run([_python(), "-m", "pytest", "-q"])
    _run([_ruff(), "check", "src", "tests", "scripts", "index.py"])
    print("Project check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
