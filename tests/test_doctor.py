from pathlib import Path

from vadbench.doctor import collect_diagnostics


def test_doctor_is_read_only_and_reports_project(tmp_path: Path) -> None:
    before = list(tmp_path.iterdir())
    report = collect_diagnostics(tmp_path)
    assert report["project_root"] == str(tmp_path.resolve())
    assert set(report["paths"]) == {"data", "weights", "outputs", "external"}
    assert list(tmp_path.iterdir()) == before
