"""Phase 3 integration test: amergin run --steps benchmark,verify."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_run_benchmark_verify(tmp_path: Path) -> None:
    """Full benchmark + verify pipeline exits 0 and produces valid results.json."""
    example_dir = Path(__file__).parent.parent
    env_override = {
        "AMERGIN_STORAGE__BENCHMARK_DATA": str(tmp_path / "benchmark-data"),
    }

    import os

    env = {**os.environ, **env_override}

    result = subprocess.run(
        [sys.executable, "-m", "amergin", "run", "--steps", "benchmark,verify"],
        cwd=str(example_dir),
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, (
        f"Command failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    results_files = list(
        (tmp_path / "benchmark-data").glob("*/verify/results.json")
    )
    assert len(results_files) == 1, "results.json not found"

    results = json.loads(results_files[0].read_text())

    # The "numpy" dot-product alternative should pass on any machine
    normalize_results = results.get("normalize", {})
    any_passed = any(
        r.get("passed") is True
        for combo in normalize_results.values()
        for r in combo.values()
    )
    assert any_passed, (
        f"Expected at least one passed verify result.\nresults.json:\n"
        f"{json.dumps(results, indent=2)}"
    )
