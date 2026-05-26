"""Phase 4 integration test: amergin run --steps benchmark,verify,evaluate."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_run_all_steps_produces_report(tmp_path: Path) -> None:
    """Full pipeline exits 0 and produces index.html + badge.json."""
    example_dir = Path(__file__).parent.parent
    env = {
        **os.environ,
        "AMERGIN_STORAGE__BENCHMARK_DATA": str(tmp_path / "benchmark-data"),
    }

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "amergin",
            "run",
            "--steps",
            "benchmark,verify,evaluate",
        ],
        cwd=str(example_dir),
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, (
        f"Command failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    # Aggregated report lives at benchmark-data/report/ in the new layout.
    report_dir = tmp_path / "benchmark-data" / "report"
    html_path = report_dir / "index.html"
    badge_path = report_dir / "badge.json"
    assert report_dir.exists(), "report/ directory not found"

    assert html_path.exists(), "index.html not produced"
    assert badge_path.exists(), "badge.json not produced"

    html = html_path.read_text(encoding="utf-8")
    assert "normalize" in html
    assert "<table" in html

    badge = json.loads(badge_path.read_text())
    assert badge["schemaVersion"] == 1
    assert "passed" in badge["message"]
