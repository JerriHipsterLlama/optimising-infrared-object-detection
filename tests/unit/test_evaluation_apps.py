import subprocess
import sys


def test_evaluation_apps_expose_help():
    for app in ("evaluate.py", "export.py", "benchmark.py", "report.py"):
        result = subprocess.run(
            [sys.executable, f"apps/{app}", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
