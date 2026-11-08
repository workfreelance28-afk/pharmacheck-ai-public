from __future__ import annotations

from datetime import date

import pytest

from scripts.project_check import PROJECT_ROOT, check_catalog_age, check_status_length


@pytest.mark.skipif(
    not (PROJECT_ROOT / "STATUS.md").is_file(),
    reason="Private operations status is not distributed",
)
def test_repository_status_file_is_within_limit():
    check_status_length(PROJECT_ROOT / "STATUS.md")


def test_status_file_over_limit_fails(tmp_path):
    status = tmp_path / "STATUS.md"
    status.write_text("line\n" * 4, encoding="utf-8")
    check_status_length(status, max_lines=4)
    status.write_text("line\n" * 5, encoding="utf-8")
    with pytest.raises(RuntimeError, match="5 lines"):
        check_status_length(status, max_lines=4)


def test_missing_status_file_fails(tmp_path):
    with pytest.raises(RuntimeError, match="missing"):
        check_status_length(tmp_path / "STATUS.md")


def _manifest(tmp_path, export_date):
    path = tmp_path / "drug_catalog.manifest.json"
    path.write_text(f'{{"export_date": "{export_date}"}}', encoding="utf-8")
    return path


def test_fresh_catalog_reports_expiry(tmp_path):
    message = check_catalog_age(_manifest(tmp_path, "2026-09-28"), today=date(2026, 10, 1))
    assert "3 days old" in message
    assert "2026-11-12" in message
    assert "WARNING" not in message


def test_catalog_warns_from_30_days(tmp_path):
    message = check_catalog_age(_manifest(tmp_path, "2026-09-28"), today=date(2026, 10, 28))
    assert message.startswith("WARNING")


def test_catalog_valid_on_last_day_and_fails_after(tmp_path):
    manifest = _manifest(tmp_path, "2026-08-07")
    assert check_catalog_age(manifest, today=date(2026, 9, 21)).startswith("WARNING")
    with pytest.raises(RuntimeError, match="46 days old"):
        check_catalog_age(manifest, today=date(2026, 9, 22))
