import subprocess
import sys


def test_evaluation_apps_expose_help():
    for app in ("evaluate.py", "export.py", "benchmark.py", "report.py", "evaluate_cluster_pruning.py"):
        result = subprocess.run(
            [sys.executable, f"apps/{app}", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def test_cluster_pruning_app_exposes_filterwise_mode():
    result = subprocess.run(
        [sys.executable, "apps/evaluate_cluster_pruning.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--filterwise" in result.stdout
    assert "--full-curve" in result.stdout


def test_matrix_app_exposes_config_and_dry_run():
    result = subprocess.run(
        [sys.executable, "apps/evaluate_compression_matrix.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
    assert "--dry-run" in result.stdout
