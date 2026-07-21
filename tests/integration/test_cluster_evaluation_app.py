from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cluster_evaluation_app_exposes_help():
    result = subprocess.run(
        [sys.executable, "apps/evaluate_cluster_pruning.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
    assert "--dry-run" in result.stdout
