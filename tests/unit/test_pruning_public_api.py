import os
import subprocess
import sys
from pathlib import Path


def test_pruning_package_import_does_not_load_fcpts():
    source_root = Path(__file__).resolve().parents[2] / "src"
    env = {**os.environ, "PYTHONPATH": str(source_root)}
    script = (
        "import sys; import infrared_detection.compression.pruning; "
        "assert not any('.fcpts' in name or '.legacy' in name "
        "for name in sys.modules), [name for name in sys.modules if '.fcpts' in name or '.legacy' in name]"
    )

    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def test_physical_pruning_exports_remain_available():
    from infrared_detection.compression.pruning import (
        build_yolo_dependency_graph,
        compute_channel_importance,
        plan_low_importance_clusters,
        prune_yolo_channels,
        run_structural_probe,
    )

    assert all(
        callable(api)
        for api in (
            build_yolo_dependency_graph,
            compute_channel_importance,
            plan_low_importance_clusters,
            prune_yolo_channels,
            run_structural_probe,
        )
    )
